#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the per-architecture GGUF conversion fixtures in test_data/arch_fixtures/.

Run this OFFLINE, by hand, when llama.cpp gains an architecture worth covering.  It is deliberately
NOT wired into CI: OpenVINO must not depend on llama.cpp at build or test time (that dependency is
the reason the native GGUF builder exists), and a CI-time generator would make OV precommit red for
upstream llama.cpp churn that has nothing to do with the PR under test.  The committed output has no
such dependency -- see docs/testing_architecture.md.

What gets committed is the GGUF *header only*: everything before min(tensor.data_offset), i.e. the
magic, the KV metadata and the tensor table.  That is the entire input to architecture detection and
graph construction; the weight bytes that follow contribute nothing to the shape of the converted
model.  The C++ test reconstructs a loadable .gguf by appending the recorded number of zero bytes
(test_arch_conversion.cpp).  Headers are ~25 KB each and delta-compress well: all 101 cost ~120 KB
in the git pack, versus 559 MB for the full files.

The headers are also seed-independent -- verified byte-identical for `-s 1` and `-s 999` -- because
the seed only feeds the weight values.  The fixtures are therefore reproducible without pinning a
seed, and pinning one anyway (below) costs nothing.

Usage:
    # 1. Build llama.cpp's arch-fixture generator at the pinned commit:
    git -C <llama.cpp> checkout 476c01efe88aad7880a8132d5d3a415f2ca75139
    cmake -S <llama.cpp> -B <llama.cpp>/build -DLLAMA_BUILD_TESTS=ON
    cmake --build <llama.cpp>/build -j --target test-llama-archs

    # 2. Regenerate:
    python3 gen_arch_fixtures.py --llama-build <llama.cpp>/build

Requires the `gguf` Python package (llama.cpp's gguf-py) to read back the generated files.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

# Pinned for reproducibility of the committed fixtures.  Bump together with a fixture refresh, and
# say so in the commit message so the provenance stays reviewable.
LLAMA_CPP_COMMIT = "476c01efe88aad7880a8132d5d3a415f2ca75139"
# The generator's own default seed is std::random_device, i.e. non-reproducible.  Pin it: it does not
# affect the header bytes, but it keeps the whole run deterministic.
SEED = 1

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "test_data", "arch_fixtures")
MANIFEST = os.path.join(FIXTURE_DIR, "manifest.txt")

MANIFEST_HEADER = """\
# GGUF per-architecture conversion fixtures -- see gen_arch_fixtures.py (generator, run offline) and
# test_arch_conversion.cpp (consumer).  Generated from llama.cpp {commit} at seed {seed}.
#
# One line per fixture:
#   <file> <data_bytes> <expectation>
#
# <file>         header-only fixture in this directory; the loadable .gguf is <file> followed by
#                <data_bytes> zero bytes (all fixture tensors are F32, so no dequant is involved).
# <expectation>  what conversion must do, and the test asserts exactly this:
#   convert   converts cleanly.  The graph fingerprint is pinned in test_arch_conversion.cpp.
#   reject    the architecture is not on the builder's accept list, so conversion must fail with
#             that specific diagnostic -- not a crash, and not a silent wrong-graph success.
#   broken    a supported architecture that currently fails to convert: a real defect, recorded so
#             it cannot be forgotten.  The test asserts it STILL FAILS; fixing the defect turns this
#             line into a failure telling you to promote it to `convert`.
"""


def run_generator(llama_build, out_dir):
    exe = os.path.join(llama_build, "bin", "test-llama-archs")
    if not os.path.isfile(exe):
        sys.exit(f"error: {exe} not found; build the test-llama-archs target first (see docstring)")
    subprocess.run([exe, "-s", str(SEED), "-o", out_dir], check=True, stdout=subprocess.DEVNULL)
    files = sorted(glob.glob(os.path.join(out_dir, "*.gguf")))
    if not files:
        sys.exit("error: generator produced no .gguf files")
    return files


def strip_to_header(path, out_path):
    """Write everything before the first tensor's data offset; return (header_len, data_len)."""
    import gguf

    reader = gguf.GGUFReader(path)
    if not reader.tensors:
        return None
    header_len = min(t.data_offset for t in reader.tensors)
    total = os.path.getsize(path)
    with open(path, "rb") as src, open(out_path, "wb") as dst:
        dst.write(src.read(header_len))
    arch_field = reader.fields["general.architecture"]
    arch = str(bytes(arch_field.parts[-1]), "utf-8")
    return header_len, total - header_len, arch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-build", required=True, help="llama.cpp build directory containing bin/test-llama-archs")
    ap.add_argument("--keep", action="store_true", help="keep the full generated .gguf files for inspection")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="gguf_arch_fixtures_")
    try:
        files = run_generator(args.llama_build, work)
        print(f"generated {len(files)} models in {work}")

        os.makedirs(FIXTURE_DIR, exist_ok=True)
        for stale in glob.glob(os.path.join(FIXTURE_DIR, "*.gguf.hdr")):
            os.remove(stale)

        entries = []
        for path in files:
            name = os.path.basename(path) + ".hdr"
            result = strip_to_header(path, os.path.join(FIXTURE_DIR, name))
            if result is None:
                print(f"  skip {os.path.basename(path)}: no tensors")
                continue
            header_len, data_len, arch = result
            entries.append((name, data_len, arch))
            print(f"  {name}: header {header_len} B, data {data_len} B, arch {arch}")

        # Expectations are NOT inferred from the frontend here -- that would make the fixtures assert
        # whatever the frontend currently does, which is not a test.  Preserve the reviewed
        # expectation from the existing manifest and default new fixtures to `reject`, which is
        # correct for any architecture the builder's accept list does not name.  A new architecture
        # that should convert is then a deliberate, reviewable one-word manifest edit.
        previous = {}
        if os.path.exists(MANIFEST):
            with open(MANIFEST) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    previous[parts[0]] = parts[2]

        with open(MANIFEST, "w") as f:
            f.write(MANIFEST_HEADER.format(commit=LLAMA_CPP_COMMIT, seed=SEED))
            for name, data_len, _arch in sorted(entries):
                expectation = previous.get(name, "reject")
                if name not in previous:
                    print(f"  NEW fixture {name}: defaulted to `reject`; review it")
                f.write(f"{name} {data_len} {expectation}\n")

        total = sum(os.path.getsize(os.path.join(FIXTURE_DIR, n)) for n, _, _ in entries)
        print(f"\nwrote {len(entries)} headers ({total / 1024:.1f} KB raw) + manifest to {FIXTURE_DIR}")

        stale_manifest = set(previous) - {n for n, _, _ in entries}
        if stale_manifest:
            print(f"note: {len(stale_manifest)} fixture(s) disappeared upstream and were dropped: "
                  f"{sorted(stale_manifest)}")
    finally:
        if args.keep:
            print(f"full models kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

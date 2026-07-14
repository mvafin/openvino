#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Measure peak RSS during read_model (conversion) and compile_model for a GGUF file.
# Reports: GGUF file size, read_model VmHWM, compile_model VmHWM.
#
# Usage:
#   PYTHONPATH=<ov>/bin/intel64/Release/python \
#   LD_LIBRARY_PATH=<ov>/bin/intel64/Release \
#   python3 measure_memory.py model.gguf [--device CPU] [--props '{"key":"val"}']

import argparse
import json
import os
import resource
import sys
import time

import openvino as ov


def rss_mb():
    """Current process peak RSS in MB (VmHWM from /proc/self/status)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    # fallback: getrusage (less accurate, maxrss in KB on Linux)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def reset_hwm():
    """Reset VmHWM by writing 5 to /proc/self/clear_refs (requires Linux 3.3+)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5\n")
        return True
    except Exception:
        return False


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Measure GGUF frontend memory consumption")
    parser.add_argument("gguf", help="Path to .gguf model file")
    parser.add_argument("--device", default="CPU", help="Target device (default: CPU)")
    parser.add_argument("--props", default="{}", help="JSON compile properties (e.g. '{\"SNIPPETS_MODE\":\"DISABLE\"}')")
    parser.add_argument("--skip-compile", action="store_true", help="Only measure read_model")
    args = parser.parse_args()

    props = json.loads(args.props)
    gguf_mb = file_size_mb(args.gguf)

    print(f"Model : {args.gguf}")
    print(f"GGUF  : {gguf_mb:.1f} MB")
    print(f"Device: {args.device}")
    if props:
        print(f"Props : {props}")
    print()

    core = ov.Core()

    # --- read_model ---
    can_reset = reset_hwm()
    if not can_reset:
        print("[warn] /proc/self/clear_refs not writable — VmHWM is cumulative")
    rss_before_read = rss_mb()

    t0 = time.perf_counter()
    model = core.read_model(args.gguf)
    t_read = time.perf_counter() - t0

    rss_after_read = rss_mb()
    read_peak = rss_after_read  # VmHWM is process max since start / last reset

    print(f"read_model  : {t_read:.1f}s  |  VmHWM {read_peak:.0f} MB  (before {rss_before_read:.0f} MB)")
    print(f"  model ops : {len(model.get_ops())}")
    # Count constants and their total byte size
    total_const_mb = sum(
        op.get_byte_size() / (1024 * 1024)
        for op in model.get_ops()
        if op.get_type_name() == "Constant"
    )
    print(f"  const size: {total_const_mb:.1f} MB (in-graph constants)")

    if args.skip_compile:
        return

    # --- compile_model ---
    reset_hwm()
    rss_before_compile = rss_mb()

    t0 = time.perf_counter()
    compiled = core.compile_model(model, args.device, props)
    t_compile = time.perf_counter() - t0

    rss_after_compile = rss_mb()
    compile_peak = rss_after_compile

    print(f"compile     : {t_compile:.1f}s  |  VmHWM {compile_peak:.0f} MB  (before {rss_before_compile:.0f} MB)")
    print()
    print(f"Summary: GGUF {gguf_mb:.0f} MB  |  read {read_peak:.0f} MB  |  compile {compile_peak:.0f} MB")


if __name__ == "__main__":
    main()

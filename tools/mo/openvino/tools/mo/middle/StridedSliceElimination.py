# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from tools.mo.openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined
import numpy as np


class StridedSliceElimination(MiddleReplacementPattern):
    """
    If input shape equal to output shape ss can be eliminated
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for ss in graph.get_op_nodes(type='StridedSlice'):
            in_shape = ss.in_port(0).data.get_shape()
            out_shape = ss.out_port(0).get_destination().data.get_shape()
            print(f"in shape: {in_shape}; out shape {out_shape}")

            if is_fully_defined(in_shape) and is_fully_defined(out_shape) and np.array_equal(in_shape, out_shape):
                print("Eliminating SS")
                out_node_in_port = ss.out_port(0).get_destination()
                ss.out_port(0).disconnect()
                ss.in_port(0).get_connection().set_destination(out_node_in_port)

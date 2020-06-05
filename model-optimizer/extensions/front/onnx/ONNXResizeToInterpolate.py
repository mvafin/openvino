"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Mul
from extensions.ops.interpolate import Interpolate
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node, rename_node
from mo.middle.passes.convert_data_type import data_type_str_to_np
from mo.ops.shape import Shape
import numpy as np


class ONNXResizeToInterpolateFrontReplacer(FrontReplacementOp):
    op = "ONNXResize"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)

        rename_node(node, node_name + '/TBR')
        interpolate = Interpolate(graph,
                                  {'mode': node.soft_get('mode', 'nearest'), 'axes': [0, 1, 2, 3]}).create_node()
        rename_node(interpolate, node_name)
        node.in_port(0).get_connection().set_destination(interpolate.in_port(0))
        if node.is_in_port_connected(2) and node.in_port(2).get_source().node.has_valid('value') and any(
                node.in_port(2).get_source().node.value):
            shape = Shape(graph, {}).create_node()
            interpolate.in_port(0).get_connection().get_source().connect(shape.in_port(0))

            mul = Mul(graph, {}).create_node()
            shape.out_port(0).connect(mul.in_port(0))
            node.in_port(2).get_connection().set_destination(mul.in_port(1))

            model_data_type = data_type_str_to_np(graph.graph['cmd_params'].data_type)
            convert_to_float = Cast(graph, dict(dst_type=model_data_type)).create_node()
            mul.in_port(0).get_connection().insert_node(convert_to_float)
            int_np_type = np.int64 if graph.graph['cmd_params'].generate_experimental_IR_V10 else np.int32
            convert_to_int = Cast(graph, dict(dst_type=int_np_type)).create_node()

            mul.out_port(0).connect(convert_to_int.in_port(0))
            convert_to_int.out_port(0).connect(interpolate.in_port(1))
        else:
            node.in_port(3).get_connection().set_destination(interpolate.in_port(1))

        return [interpolate.id]

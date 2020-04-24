"""
 Copyright (C) 2018-2020 Intel Corporation

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
from onnx import numpy_helper
import numpy as np
from onnx.numpy_helper import to_array

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.const import Const


class ConstExtractor(FrontExtractorOp):
    op = 'prim::Constant'
    enabled = True

    @classmethod
    def extract(cls, node):
        out_type = node.pb.output().type().kind()
        if out_type == 'NoneType':
            value = None
            data_type = None
        elif out_type == 'FunctionType':
            # This is a fuction, will be extracted in different extractor
            return
        else:
            if 'value' in node.pb.attributeNames():
                if out_type == 'BoolType':
                    value = node.pb.i('value')
                    data_type = np.bool
                elif out_type == 'IntType':
                    value = node.pb.i('value')
                    data_type = np.int
                elif out_type == 'FloatType':
                    value = node.pb.f('value')
                    data_type = np.float
                elif out_type == 'TensorType':
                    value = node.pb.t('value').numpy()
                    data_type = value.dtype
                elif out_type == 'StringType' or out_type == 'DeviceObjType':
                    value = node.pb.s('value')
                    data_type = str
                else:
                    raise 'Not supported constant Type {}'.format(out_type)
            else:
                return

        attrs = {
            'data_type': data_type,
            'value': value
        }
        Const.update_node_stat(node, attrs)
        return cls.enabled


class ConstFromClassExtractor(FrontExtractorOp):
    op = 'prim::GetAttr'
    enabled = True

    @classmethod
    def extract(cls, node):
        if len(node.in_nodes()) != 1 or 'name' not in node.pb.attributeNames():
            # it is not a constant, must be handled differently
            return

        name = node.pb.s('name')
        in_class = node.in_node()

        if node.pb.output().type().kind() == 'TensorType':
            value = in_class.module.__getattr__(name).detach().numpy()
            data_type = value.dtype
        elif node.pb.output().type().kind() == 'BoolType':
            value = in_class.module._initializing
            data_type = np.bool

        node.graph.remove_edge(in_class.id, node.id)

        attrs = {
            'data_type': data_type,
            'value': value
        }
        Const.update_node_stat(node, attrs)
        return cls.enabled

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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging as log

import torch
import torchvision

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import fill_graph_with_nodes, Graph, Node
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.utils.error import Error, FrameworkError


def load_torchscript_model(file_name: str):
    try:
        torchscript_model = torch.jit.load(file_name)
    except Exception as e:
        raise FrameworkError(
            'Cannot read the model file: "{}" is incorrect TorchScript model file. Details: {}',
            file_name,
            str(e)
        ) from e

    return torchscript_model


def protobuf_attrs(pb):
    return {'pb': pb}


def extract_graph(graph, module, parent_name=''):
    # maps a tensor name to a node produced it and the node port: str -> (node_id, node_port)
    data_nodes_map = {}

    # first go through all inputs
    classes = {}
    for inp in module.graph.inputs():
        name_id = name = parent_name + inp.debugName()
        if graph.has_node(name):
            log.debug('Name {} of input node already exists, input names are duplicated.'.format(name))
            name_id = graph.unique_id(name)
            # raise Error('Name {} of input node already exists, input names are duplicated.', name)
        if inp.type().kind() == 'ClassType':
            graph.add_node(name_id, kind='op', op='Class', pb=inp, module=module)
            classes[name] = module
        elif not parent_name:
            graph.add_node(name_id, kind='op', op='Parameter', pb=inp)
        # add to a tensors map
        assert not name in data_nodes_map, 'Inconsistency between data_nodes_map and graph.nodes'
        data_nodes_map[name] = (name_id, 0)

    incoming_ids = {}
    outcoming_ids = {}
    graph_nodes = list(module.graph.nodes())
    for node in graph_nodes:
        # create an NX node
        if len(list(node.outputs())) == 0:
            # node doesn't have output, not needed in graph
            # TODO: remove nodes before that
            continue
        id = graph.unique_id(parent_name + next(node.outputs()).debugName())
        if 'GetAttr' in node.kind() and node.output().type().kind() == 'ClassType':
            parent_node_name = parent_name + node.input().debugName()
            assert parent_node_name in classes
            this_module = classes[parent_node_name].__getattr__(node.s('name'))
            # id = graph.unique_id(parent_name + node.s('name'))
            graph.add_node(id, pb=node, kind='op', op='Class', module=this_module, blocks=list(node.blocks()))
            classes[id] = this_module
        else:
            graph.add_node(id, pb=node, kind='op', op=node.kind())

            # add incoming edges based on data_nodes_map
            for dst_port, _inp in enumerate(node.inputs()):
                # should add edge inp --> id
                inp = parent_name + _inp.debugName()
                if inp not in data_nodes_map:
                    raise Error(
                        'Reference to {} is not satisfied. A node refer not existing data tensor. TorchScript model is not '
                        'consistent. Protobuf fragment: {}', inp, node)

                src_id, src_port = data_nodes_map[inp]
                if not graph.has_node(src_id):
                    if inp in incoming_ids:
                        incoming_ids[inp].append((id, dst_port))
                    else:
                        incoming_ids[inp] = [(id, dst_port)]
                    continue

                edge_attrs = {
                    'out': src_port,
                    'in': dst_port,
                    'name': inp,
                    'fw_tensor_debug_info': [(inp, inp)],
                    'in_attrs': ['in', 'name'],
                    'out_attrs': ['out', 'name'],
                    'data_attrs': ['fw_tensor_debug_info']
                }
                graph.add_edge(src_id, id, **edge_attrs)

        # add outgoing edges to data_nodes_map
        outcoming_ids = {}
        for src_port, _out in enumerate(node.outputs()):
            out = parent_name + _out.debugName()
            if out in data_nodes_map:
                log.debug("Detected reuse of blob {}.".format(out))
            outcoming_ids[src_port] = id
            data_nodes_map[out] = (id, src_port)
    return incoming_ids, outcoming_ids


class ExtractInnerGraphs(FrontReplacementSubgraph):
    enabled = False

    def pattern(self):
        return dict(nodes=[('class', dict(op='Class')),
                           ('call_method', dict(op='prim::CallMethod'))],
                    edges=[('class', 'call_method', {'in': 0, 'out': 0})])

    def replace_sub_graph(self, graph: Graph, match: dict):
        c = match['class']
        call_method = match['call_method']

        in_ids, out_ids = extract_graph(graph, c.module, parent_name=c.id + '/')
        i = 1
        in_ids_to_remove = set()
        for in_ids_for_inp in in_ids.values():
            for id, port in in_ids_for_inp:
                if i in call_method.in_edges():
                    edge = call_method.in_edge(i).copy()
                    edge['in'] = port
                    graph.add_edge(call_method.in_node(i).id, id, **edge)
                else:
                    print('foo')
            in_ids_to_remove.add(call_method.in_node(i).id)
            i += 1
        # one node may have several edges to call_mathod, so need to remove edges only once for each id
        for id in in_ids_to_remove:
            graph.remove_edge(id, call_method.id)
        for id, edge in call_method.get_outputs():
            if edge['out'] in out_ids:
                graph.add_edge(out_ids[edge['out']], id, **edge)
                graph.remove_edge(call_method.id, id)
        graph.remove_node(call_method.id)
        if not c.get_outputs():
            graph.remove_node(c.id)

        return []


class ExtractFunctions(FrontReplacementSubgraph):
    def pattern(self):
        return dict(nodes=[('constant', dict(op='prim::Constant')),
                           ('call_function', dict(op='prim::CallFunction'))],
                    edges=[('constant', 'call_function', {'in': 0, 'out': 0})])

    def replace_sub_graph(self, graph: Graph, match: dict):
        f_name = match['constant']
        call_function = match['call_function']

        assert f_name.pb.output().type().kind() == 'FunctionType'
        assert 'name' in f_name.pb.attributeNames()

        call_function.op = 'F.' + f_name.pb.s('name')
        graph.remove_node(f_name.id)
        for edge in call_function.in_edges().values():
            edge['in'] -= 1

        return []


class ExtractSelfMethods(FrontReplacementSubgraph):
    def pattern(self):
        return dict(nodes=[('class', dict(op='Class')),
                           ('get_attr', dict(op='prim::GetAttr')),
                           ('call_method', dict(op='prim::CallMethod'))],
                    edges=[('class', 'get_attr'),
                           ('class', 'call_method', {'in': 0, 'out': 0}),
                           ('get_attr', 'call_method')])

    def replace_sub_graph(self, graph: Graph, match: dict):
        c = match['class']
        call_method = match['call_method']

        # in_ids, out_ids = extract_graph(graph, c.module._c.get_debug_state(),
        #                                 parent_name=c.id + '/' + c.module.original_name + '/')
        #
        # i = 1
        # for in_ids_for_inp in in_ids.values():
        #     for id, port in in_ids_for_inp:
        #         if i in call_method.in_edges():
        #             edge = call_method.in_edge(i).copy()
        #             edge['in'] = port
        #             graph.add_edge(call_method.in_node(i).id, id, **edge)
        #         else:
        #             print('foo')
        #     graph.remove_edge(call_method.in_node(i).id, call_method.id)
        #     i += 1
        # for id, edge in call_method.get_outputs():
        #     if edge['out'] in out_ids:
        #         graph.add_edge(out_ids[edge['out']], id, **edge)
        #         graph.remove_edge(call_method.id, id)
        # graph.remove_node(call_method.id)
        # if not c.get_outputs():
        #     graph.remove_node(c.id)

        call_method.op = c.module.original_name
        call_method['module'] = c.module

        graph.remove_edge(c.id, call_method.id)
        for edge in call_method.in_edges().values():
            edge['in'] -= 1

        return []


def protobuf2nx(graph, pb):
    '''Convert proto message with ONNX model to equivalent NX representation.
    All nodes and edges are restored here as ONNX model has op/data representation,
    that means that nodes are connected via tensor names. Name of tensors are defined
    on demand in nodes, so we have a code similar to Caffe here. '''

    # Go through all nodes in the original model order (because data nodes are defined on-the-fly and order is
    # important)

    extract_graph(graph, pb)
    # for_graph_and_each_sub_graph_recursively(graph, extractor.find_and_replace_pattern)
    ExtractInnerGraphs().find_and_replace_pattern(graph)
    ExtractInnerGraphs().find_and_replace_pattern(graph)
    ExtractInnerGraphs().find_and_replace_pattern(graph)
    ExtractSelfMethods().find_and_replace_pattern(graph)
    print('foo')
    # for id, src_port in out_edges:
    #     graph.add_node(id + 'sink_port', kind='op', op='Result')

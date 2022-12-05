from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGInNode, DAGOutNode
import numpy as np
import kahypar
from pathlib import Path
from itertools import compress


##########################################################################################
#      this script utilizes `KaHyPar <https://kahypar.org>` hypergraph partitioning 
#      framework to automatically perform graph partitioning schedule which is served
#      as part of the circuit cutting workflow.
#                        developed using kahypar version 1.1.7
##########################################################################################

def kahypar_cut(circuit, num_fragments, epsilon, config_path = None):
    '''
    Utilizing `KaHyPar <https://kahypar.org>`framework for graph partitioning.

    Args:
        circuit ( qiskit `QuantumCircuit` object): The quantum circuit (DAG graph) to be cut.
        epsilon (float): Imbalance factor of the graph partitioning.
        num_fragments (int): Desired number of fragments
        config_path (str): KaHyPar's ``.ini`` config file path. Defaults to its SEA20 paper config.
        
    Returns:
        a tuple of cuts by format: ( ( wire of cutting edge, location ), ... )
    '''
    graph = circuit_to_dag(circuit)
    edges = graph._multi_graph.edge_list()
    nodes = graph._multi_graph.nodes()

    # convert graph into hMETIS hypergraph input format <http://glaros.dtc.umn.edu/gkhome/fetch/sw/hmetis/manual.pdf>
    # conforming to KaHyPar's calling signature.
    # `adjacent_nodes` : Flattened list of adjacent node indices.
    #  `edge_splits` : List of starting indices for edges in the above adjacent-nodes-list.
    adjacent_nodes = [ v for node in edges for v in node ]
    edge_splits = np.cumsum([0]+[ len(e) for e in edges]).tolist()
    num_nodes = max(adjacent_nodes) + 1
    num_edges = len(edge_splits) - 1

    # generate Hypergraph instance and use KaHyPar for hypergraph partitioning
    hypergraph = kahypar.Hypergraph(num_nodes, num_edges, edge_splits, adjacent_nodes, num_fragments)
    context = kahypar.Context()
    config_path = config_path or str(Path(__file__).parent / "cut_kKaHyPar_sea20.ini")
    context.loadINIconfiguration(config_path)
    context.setK(num_fragments)
    context.setEpsilon(epsilon)
    context.suppressOutput(True)
    kahypar.partition(hypergraph, context)

    cut_edge_mask = [hypergraph.connectivity(e) > 1 for e in hypergraph.edges()]
    # compress() ignores the extra hyperwires at the end if there is any.
    cut_edges_raw = list(compress(edges, cut_edge_mask))
    # the cut edges resolved by KaHyPar contain edges between op node and in/out node
    # which is not what we want, so we only keep the edges between op node and op node
    cut_edges = _cut_edge_filter(graph, cut_edges_raw)
    edgedata = [graph._multi_graph.get_edge_data(cut_edge[0], cut_edge[1]) for cut_edge in cut_edges]

    return _collect_cuts(edgedata, cut_edges, nodes)

def _cut_edge_filter(graph, cut_edge_raw):
    '''
    filter the cut edges, only keep the edges between op node and op node.
    '''
    cut_edges = []
    for cut_edge in cut_edge_raw:
        node0 = graph._multi_graph.get_node_data(cut_edge[0])
        node1 = graph._multi_graph.get_node_data(cut_edge[1])
        if isinstance(node0,DAGOpNode) and isinstance(node1,DAGOpNode):
            cut_edges.append((cut_edge[0],cut_edge[1]))
    
    return cut_edges

def _collect_cuts(edgedata, cut_edges, nodes):
    '''
    returns the final cuts by format: ( ( wire of cutting edge, location ), ... )
    '''
    # locate the cut edge, find the index of the cutting edge on wire sorted by order
    cut_loc = []
    for cut_edge,adj_nodes in zip(edgedata,cut_edges):
        ops = [node._node_id for node in nodes if isinstance(node, DAGOpNode) and cut_edge in node.qargs]
        cut_loc.append(ops.index(adj_nodes[1]))

    return tuple(zip(edgedata,cut_loc))

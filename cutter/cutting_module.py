from qiskit import *
from processor.processing_module import *
from processor.recombination_model import *
from cutter.kahypar_cutter import kahypar_cut
from qiskit.dagcircuit import DAGCircuit,DAGNode
from qiskit.circuit import Qubit, Clbit, AncillaQubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode, DAGInNode, DAGOutNode
import retworkx as rx
import copy
import time
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize


##########################################################################################
#       this script provides class and functions for finding optimal cut strategy
#       and cutting procedure.
#                        developed using qiskit version 0.33
##########################################################################################

class Circuit_Cutting_Task():
    '''
    The complete circuit cutting workflow module, which includes:

    '''
    def __init__(self, circuit, cuts, shots, backend, reconstruct_method = "direct"):
        self.circuit = circuit
        self.cuts = cuts
        self.shots = shots
        self.backend = backend
        self.reconstruct_method = reconstruct_method
        self.full_circuit_dist = {}
        # list of all possible measurement outcomes (bitstrings)
        self.all_bits = [ "".join(bits) for bits in itertools.product(["0","1"],
                            repeat = len(self.circuit.qubits)) ]
        # get the actual state / probability distribution for the full circuit
        self.actual_state = get_statevector(circuit)
        self.actual_dist = { "".join(bits) : abs(amp)**2
                for bits, amp in zip(self.all_bits, self.actual_state)
                if amp != 0 }
        if is_valid_cut(self.circuit,self.cuts,self.shots):
            self.cut_procedure()
            self.reconstruct_procedure()
            self.full_circuit_simulation()
            self.print_results()


    def fidelity(self, dist):
        '''
        compute the fidelity between two quantum state
        '''
        fidelity = sum( numpy.sqrt(self.actual_dist[bits] * dist[bits], dtype = complex)
                    for bits in self.all_bits
                    if self.actual_dist.get(bits) and dist.get(bits) )**2
        return fidelity.real if fidelity.imag == 0 else fidelity

    def full_circuit_simulation(self):
        '''
        compute a simulated probability distribution for the full circuit
        '''
        print('~~Full circuit simulation~~')
        self.circuit.measure_all()
        full_circuit_result = run_circuits(self.circuit, self.shots, backend = "qasm_simulator")
        for part in full_circuit_result:
            for bits, counts in part.get_counts(self.circuit).items():
                if bits not in self.full_circuit_dist:
                    self.full_circuit_dist[bits] = 0
                self.full_circuit_dist[bits] += counts / self.shots
        
        self.full_circuit_fidelity = self.fidelity(self.full_circuit_dist)
        print('Done!')
        print()

    def cut_procedure(self):
        '''
        cut a circuit into subcircuits, run each subcircuit and collect the corresponding data
        '''
        if len(self.circuit.qubits) <= 10:
            print('Full circuit:')
            print()
            print(self.circuit)
            print()
        print('~~Cutting the circuit~~')
        self.subcircuits, self.qubit_map, self.stitches = cut_circuit(self.circuit, self.cuts, self.shots)
        print('Done!')
        self.bit_permutation = bit_axis_permutation(self.subcircuits, self.qubit_map)
        print(self.bit_permutation)
        if all(len(subcircuit.circuit.qubits) <= 10 for subcircuit in self.subcircuits):
            for idx,subcircuit in enumerate(self.subcircuits):
                print(f'subcircuit {str(idx)}:')
                print(subcircuit.circuit)
                print()
        print('~~Running subcircuits and collecting data~~')
        self.subcircuit_data = collect_subcircuit_data(self.subcircuits, backend = self.backend)
        print('Done!')
        print() 

    def reconstruct_procedure(self):
        '''
        reconstruct the choi matrix of each subcircuit by a variety of methods, then recombine
        the probability distribution of full circuit
        '''
        print('~~Reconstructing the circuit~~')
        print('Using the method of "'+str(self.reconstruct_method)+'"')
        if self.reconstruct_method == "direct":
            self.choi_matrix = direct_circuit_model(self.subcircuit_data)
        elif self.reconstruct_method == "MLFT":
            self.choi_matrix = direct_circuit_model(self.subcircuit_data)
            self.choi_matrix = maximum_likelihood_model(self.choi_matrix)
        
        #  recombine the probability distribution of circuit by methods of :
        #  1. tensor-network-based method by default (method = "network")
        #  2. insert a complete basis of operators (method = "insertion")
        start = time.time()
        self.prob_distribution = recombine_circuit_models(self.choi_matrix, self.stitches, 
                                self.bit_permutation, self.subcircuits)
        end = time.time()
        self.reconstruct_time = end - start
        print('reconstruct time',self.reconstruct_time)
        #  if there are negative probabilities induced by noise, don't bother fitting
        self.prob_distribution = naive_fix(self.prob_distribution)
        #  compute the reconstruction fidelity
        self.reconstruct_fidelity = self.fidelity(self.prob_distribution)
        print('Done!')
        print()

    def print_results(self):
        '''
        output the result
        '''
        print('Results:')
        print('reconstruction fidelity:', self.reconstruct_fidelity)
        print('full circuit simulation fidelity:', self.full_circuit_fidelity)
        print()


class Cut_Solver():
    '''
    Automatically find the optimal Cut Strategy, this module served as part of the circuit cutting workflow.
    
    '''
    def __init__(self, circuit, shots, cut_constraint, cutter = "kahypar"):
        self.circuit = circuit
        self.shots = shots
        self.cutter = cutter

        self.problem = Optimal_Cut(circuit, shots, cut_constraint, cutter)
        self.algorithm = NSGA2(pop_size=20,
                  sampling=MixedVariableSampling(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  eliminate_duplicates=MixedVariableDuplicateElimination(),
                  )
        self.result = minimize(self.problem, 
                            self.algorithm,
                            ('n_gen', 200),
                            seed=1,
                            verbose=False)

    

class Optimal_Cut():
    '''
    Convert the problem of finding the optimal Cut Strategy into Multi-Objective Optimization problem,
    then solve it. 

    --------------------------------------Probelm Definition--------------------------------------------

    Variables:
        epsilon (float): imbalance factor of the graph partitioning
        num_fragments (int): number of fragments we want to cut
    
    Constraints:
        - the minimal number of qubits in one subcircuit
        - the maximum number of qubits in one subcircuit
        - the maximum depth in one subcircuit

    Objectives:
        - minimize the maximum number of cuts on one subcircuit
        - minimize the total number of cuts
        - minimize qubit number imbalance of subcircuits
        - minimize depth imbalance of subcircuits
        - minimize operation number imbalance of subcircuits

    ----------------------------------------------------------------------------------------------------

    Args:
        circuit (`QuantumCircuit`): the circuit to cut
        shots (int): number of shots
        cut_constraint (dict): the constraints of cutted subcircuits

    '''
    def __init__(self, circuit, shots, cut_constraint, cutter):
        # set up variables for searching optimal cutting parameters
        # "epsilon" : imbalance factor of the graph partitioning
        # "num_fragments" : number of fragments
        self.variables = {
            "num_fragments": np.arange(2, cut_constraint["max_size_fragments"]+1).tolist(),
            "epsilon": np.round(np.arange(0.01, 1, 0.03),2).tolist(),
            }

        self.circuit = circuit
        self.shots = shots
        self.cut_constraint = cut_constraint
        self.cutter = cutter
        self.results = {}
        self._sampling()

    def _evaluate(self, num_fragments, epsilon):
        '''
        evaluate the result of a couple of variables (epsilon, num_fragments), impose constraints
        to determine whether the results shounld be kept, then return the objective functions.
        '''
        # choose cutter to cut the circuit
        if self.cutter == "kahypar":
            cuts = kahypar_cut(self.circuit, num_fragments, epsilon)
        # check the validation of cuts
        if not is_valid_cut(self.circuit, cuts, self.shots):
            return []
        subcircuits, _, _, = cut_circuit(self.circuit, cuts, self.shots)
        min_qubits = min( len(subcircuit.circuit.qubits) for subcircuit in subcircuits )
        max_qubits = max( len(subcircuit.circuit.qubits) for subcircuit in subcircuits )
        min_depth = min( subcircuit.circuit.depth() for subcircuit in subcircuits )
        max_depth = max( subcircuit.circuit.depth() for subcircuit in subcircuits )
        # we weight two_qubit gate twice as much as single_qubit gate, since it has higher error rate
        ops_count = []
        for subcircuit in subcircuits:
            if 'cx' in subcircuit.circuit.count_ops():
                ops_count.append(subcircuit.circuit.count_ops()['cx'] + 
                    sum(subcircuit.circuit.count_ops().values()))
            else:
                ops_count.append(sum(subcircuit.circuit.count_ops().values()))
        # Inequality Constraints
        # g1 : the minimal number of qubits in one subcircuit
        # g2 : the maximum number of qubits in one subcircuit
        # g3 : the maximum depth in one subcircuit
        if min_qubits < self.cut_constraint["min_qubits"]:
            return []
        if max_qubits > self.cut_constraint["max_qubits"]:
            return []
        if max_depth > self.cut_constraint["depth"]:
            return []
        
        # Objective Functions
        # f1 : the maximum number of cuts on one subcircuit
        # f2 : the total number of cuts
        # f3 : qubit number imbalance of subcircuits
        # f4 : operation number imbalance of subcircuits
        # f5 : depth imbalance of subcircuits
        f1 = max( 4**len(subcircuit.prep_qubits) * 3**len(subcircuit.meas_qubits)
                                    for subcircuit in subcircuits )
        f2 = len(cuts)
        f3 = max_qubits - min_qubits
        f4 = max(ops_count) - min(ops_count)
        f5 = max_depth - min_depth

        return [f1, f2, f3, f4, f5]
    
    def _sampling(self):
        '''
        uniformly sample cut variables from variable space and evaluate the cutting performance,
        store the corresponding results.
        '''
        for num_frags in self.variables["num_fragments"]:
            for eps in self.variables["epsilon"]:
                if self._evaluate(num_frags,eps):
                    self.results[(num_frags,eps)] = self._evaluate(num_frags,eps)

       


##########################################################################################
# functions for cutting and checking a quantum circuit
##########################################################################################

def is_valid_cut(circuit, cuts, shots):
    '''
        check if the cut is valid. For the case of inner prep/meas qubits on one subcircuit, which means
        we need to do intermediate measurement and qubit reset operations, our reconstruction procedure
        will collapse up to now. we use valid `bit_permutation` list to check it. More general implementation
        may update in the future.
        '''
    subcircuits, _, _, = cut_circuit(circuit, cuts, shots)
    if not subcircuits:
        print("Invalid Cut!")
        return False
    return True

def _trimmed_circuit(graph, graph_idx, prep_qubits, meas_qubits, 
                        num_shots, qreg_name = "q", creg_name = "c"):
    '''
    "trim" a circuit graph (in DAG form) by eliminating unused bits.

    Returns:
        subcircuit (`SubCircuit` class): trimmed subcircuit.
        register_map (dict): a dictionary mapping old wires to new ones
    '''
    # idenify wires used in this graph
    wires = set.union(*[ set(node.qargs) for node in graph.op_nodes() ])

    # construct map from old bits to new ones
    old_qbits = [wire for wire in wires if isinstance(wire, (Qubit, AncillaQubit))]
    old_cbits = [ wire for wire in wires
                  if isinstance(wire, Clbit) ]

    new_qbits = QuantumRegister(len(old_qbits), qreg_name) if old_qbits else []
    new_cbits = ClassicalRegister(len(old_cbits), creg_name) if old_cbits else []

    registers = [ reg for reg in [ new_qbits, new_cbits ] if reg != [] ]
    trimmed_circuit = QuantumCircuit(*registers)

    register_map = list(zip(old_qbits, new_qbits)) + list(zip(old_cbits, new_cbits))
    register_map = dict(register_map)

    sub_prep_qubits = [register_map[prepq] for prepq in prep_qubits]
    sub_meas_qubits = [register_map[measq] for measq in meas_qubits]
    sub_prep_qubits = sorted(sub_prep_qubits, key=lambda qubit: qubit.index)
    sub_meas_qubits = sorted(sub_meas_qubits, key=lambda qubit: qubit.index)
    # add all operations to the trimmed circuit
    for node in graph.op_nodes():
        new_qargs = [ register_map[qbit] for qbit in node.qargs ]
        new_cargs = [ register_map[cbit] for cbit in node.cargs ]
        trimmed_circuit.append(node.op, qargs = new_qargs, cargs = new_cargs)


    subcircuit = SubCircuit(trimmed_circuit,graph_idx,sub_prep_qubits,sub_meas_qubits,num_shots)
    for map in register_map:
        register_map[map] = (graph_idx,register_map[map])

    return subcircuit, register_map

def _remove_cut_edges(graph,cuts):
    '''
    remove the cutting edges and the (DAG) graph will be splitted into subgraphs.

    Returns:
        graph (`DAGCircuit` object): cutted graph after removing the cutting edges.
        nodeids (list): the ids of node before and after a cutting edge, used for 
            identifying the preperation/measurement qubits in every subgraph.
    '''
    nodes = graph._multi_graph.nodes()
    # barriers currently interfere with splitting a graph into subgraphs
    graph.remove_all_ops_named("barrier")
    cutnodes = []
    # find neighborhood nodes of every cutting edge, remove the cutting edge between these nodes
    for cut in cuts:
        ops = [node for node in nodes if isinstance(node, DAGOpNode) and cut[0] in node.qargs]
        cutnodes.append([ops[cut[1]-1],ops[cut[1]]])
    # obtain node ids
    nodeids = [[cutnode[0]._node_id,cutnode[1]._node_id] for cutnode in cutnodes]
    # remove cutting edges
    for ids in nodeids:
        graph._multi_graph.remove_edge(ids[0],ids[1])
    
    return graph, nodeids

def _merge_subgraph(used_qubits_collect, cut_qubits, subgraphs_nodes):
    '''
    determine if it is necessary to merge some subgraphs, if all uncut subgraphs are small, do the
    merge process.

    Returns:
        merge_subgraph (bool): whether to merge subgraphs or not.
    '''
    uncut_subgraphs = [used_qubits_collect.index(used_qubits) for used_qubits in used_qubits_collect
                     if all(qbit not in cut_qubits for used_q in used_qubits for qbit in used_q)]
    if not uncut_subgraphs:
        return False
    else:
        cut_subgraphs = list(set(list(range(len(subgraphs_nodes)))) - set(uncut_subgraphs))
        # if uncut subgraphs are somehow bigger than cut subgraphs, skip the merge step
        for uncut_subgraph in uncut_subgraphs:
            if any(len(subgraphs_nodes[uncut_subgraph]) >= len(subgraphs_nodes[cut_subgraph]) 
                            for cut_subgraph in cut_subgraphs):
                return False
                
        return True
        
def _trimmed_graph(graph, cuts):
    '''
    detach subgraphs from a graph, if there are subgraphs with no cuts, glue them onto other subgraphs.

    Returns:
        rx_subgraphs (list[`PyDiGraph`]): a list of subgraphs in `retworkx.PyDiGraph` form.
    '''
    rx_graph = graph._multi_graph
    used_qubits_collect = []
    cut_qubits = [ cut[0] for cut in cuts]
    # collect subgraphs by their nodes
    subgraphs_nodes = [ list(subgraph_nodes)
                        for subgraph_nodes in rx.weakly_connected_components(rx_graph) ]
    
    # find all qubits in one subgraph
    for subgraph_nodes in rx.weakly_connected_components(rx_graph):
        nodes = [rx_graph.get_node_data(node) for node in subgraph_nodes]
        qargs = [node.qargs for node in nodes if isinstance(node,DAGOpNode)]
        used_qubits_collect.append(list(set(qargs)))

    # if there are no cuts in original graph or not necessary to merge subgraphs,
    # return all individual subgraphs directly
    merge_subgraph = _merge_subgraph(used_qubits_collect, cut_qubits, subgraphs_nodes)
    if not cuts or not merge_subgraph:
        return [rx_graph.subgraph(rx_subgraph_nodes) for rx_subgraph_nodes in subgraphs_nodes]

    # join all uncut subgraphs(free nodes) together if exist, and wipe them out from original graph
    free_nodes = []
    pop_idx = []
    for idx, used_qubits in enumerate(used_qubits_collect):
            if all( qbit not in cut_qubits for used_q in used_qubits for qbit in used_q ):
                free_nodes.extend(subgraphs_nodes[idx])
                pop_idx.append(idx) 
    if free_nodes:
        # remove all free nodes from original graph
        pop_idx.reverse()
        for idx in pop_idx:
            subgraphs_nodes.pop(idx)
        # if collected uncut subgraph are bigger than cut subgraphs, return it with all other cut subgraphs
        if any( len(free_nodes) >= len(subgraph) for subgraph in subgraphs_nodes ):
            subgraphs_nodes.append(free_nodes)
            return [rx_graph.subgraph(rx_subgraph_nodes) for rx_subgraph_nodes in subgraphs_nodes]
        else:
            # reorganize trimmed subgraphs by appending the joint uncut subgraph(free nodes) to 
            # smallest subgraph in the rest
            subgraph_len = [ len(subgraph) for subgraph in subgraphs_nodes ]
            sort_subgraph_len = sorted(subgraph_len)
            # merge two smallest subgraphs
            subgraphs_nodes[subgraph_len.index(sort_subgraph_len[0])].extend(free_nodes)
    
    return [rx_graph.subgraph(rx_subgraph_nodes) for rx_subgraph_nodes in subgraphs_nodes]

def _disjoint_graph(graph, nodeids, cuts):
    '''
    generate subgraphs (new DAG graph) which are separated from a graph and store the information of 
    the preperation/measurement qubits and originla stitches.

    Returns:
        subgraphs (list[`DAGCircuit`]): a list of subgraphs.
        prep_qubits (list): a list of preperation qubits of subgraphs.
        meas_qubits (list): a list of measurement qubits of subgraphs.
        stitches (dict): a dictionary indicate the connectivity between subgraphs.
    '''
    rx_subgraphs = _trimmed_graph(graph, cuts)
    # convert subgraphs of nodes to circuit graphs
    subgraphs = []
    prep_qubits = []
    meas_qubits = []
    for rx_subgraph in rx_subgraphs:
        # make a copy of the full graph, and remove nodes not in this subgraph
        subgraph = copy.deepcopy(graph)
        for node in subgraph.op_nodes():
            if not any( DAGNode.semantic_eq(node, rx_node)
                        for rx_node in rx_subgraph.nodes() if isinstance(rx_node, DAGOpNode) ):
                subgraph.remove_op_node(node)

        # identify the preperation/measurement qubits in every subgraph, check the cut validation,
        # return empty dataset if invalid
        sub_prep_qubits, sub_meas_qubits, invalid = _prep_meas_identify(subgraph, nodeids, cuts)
        if invalid:
            return [], [], [], {}
        # save the preperation/measurement qubits and pass to SubCircuit parameters
        prep_qubits.append(sub_prep_qubits)
        meas_qubits.append(sub_meas_qubits)
        # ignore trivial subgraphs 
        if len(subgraph.op_nodes()) == 0: continue

        subgraphs.append(subgraph)

    # store all original stitches in a cut-up circuit, in dictionary format:
    # { <exit wire> : <init wire> }
    stitches = _find_original_stitches(subgraphs, nodeids, cuts)

    return subgraphs, prep_qubits, meas_qubits , stitches

def _prep_meas_identify(subgraph, nodeids, cuts):
    '''
    identify the preperation/measurement qubits in every subgraph by finding the retained node
    around cutting edge, meanwhile check if it is a valid cut (see comments of `is_valid_cut`),
    if not, return empty lists.

    Returns:
        sub_prep_qubits (list): a list of preperation qubits in this subcircuit.
        sub_meas_qubits (list): a list of measurement qubits in this subcircuit.
        invalid (bool): whether the cut is invalid or not
    '''
    # if there are no cuts in graph, skip this routine
    if not cuts:
        return [], [], False
    # collect preperation/measurement qubits in every subgraph
    sub_prep_qubits = []
    sub_meas_qubits = []
    for node in subgraph.op_nodes():
        for nodeid, wire in zip(nodeids, cuts):
            ops = [node._node_id for node in subgraph.op_nodes()
                    if wire[0] in node.qargs]
            if nodeid[0] == node._node_id:
                # measurement node should be the last node on a qubit, otherwise it is invalid
                if node._node_id != ops[-1]:
                    return [], [], True
                sub_meas_qubits.append(wire[0])
            if nodeid[1] == node._node_id:
                # preperation node should be the first node on a qubit, otherwise it is invalid
                if node._node_id != ops[0]:
                    return [], [], True
                sub_prep_qubits.append(wire[0])
    
    return sub_prep_qubits, sub_meas_qubits, False

def _find_original_stitches(subgraphs, nodeids, cuts):
    '''
    find original stitches in a cut-up circuit.
    
    Returns:
        { (subcircuit_id0, qubit_in_original_circuit) : (subcircuit_id1, qubit_in_original_circuit) }
    '''
    stitches_prep = []
    stitches_meas = []
    for nodeid, wire in zip(nodeids, cuts):
        for subid, subgraph in enumerate(subgraphs):
            for node in subgraph.op_nodes():
                if nodeid[0] == node._node_id:
                    stitches_prep.append((subid, wire[0]))
                if nodeid[1] == node._node_id:
                    stitches_meas.append((subid, wire[0]))

    return dict(zip(stitches_prep,stitches_meas))

def _collect_stitches(stitches_data, wire_path_map):
    '''
    collect all stitches among subcircuits.
    
    Returns:
        { <exit wire in subcircuit> : <init wire in subcircuit> }
    '''
    stitches = {}
    for prep_qubit, meas_qubit in stitches_data.items():
        for qubit_map in wire_path_map[prep_qubit[1]]:
            if qubit_map[0] == prep_qubit[0]:
                new_prep = qubit_map[1]
            if qubit_map[0] == meas_qubit[0]:
                new_meas = qubit_map[1]

        stitches[(prep_qubit[0],new_prep)] = (meas_qubit[0],new_meas)

    return stitches

def cut_circuit(circuit, cuts, num_shots):
    '''
    cut a quantum circuit into subcircuits, and store the information of subcircuits, qubit map,
    and stitches that indicate the connectivity between subcircuits.

    Args:
        circuit (`QuantumCircuit` object): the circuit need to be cut.
        cuts (list): a list of cut points in the form of (`Qubit`, location), where location is 
            the index of the cutting edge on corresponding qubit.
        num_shots (int): the number of shots on circuit.

    Returns:
        subcircuits (list[`SubCircuit`]): a list of subcircuits.
        wire_path_map (dict): a dictionary mapping qubits in the original circuit to
            a list of qubits in subcircuits, in the form of
            { < qubit of original circuit > : [ ( < qubit of subcircuit_1  >, < qubit of subcircuit_2 > ) ] }
        stitches (dict): stitches that indicate the connectivity between subcircuits, in the form of
            { <exit wire in subcircuit> : <init wire in subcircuit> }
    '''
    # convert circuit into DAG 
    graph = circuit_to_dag(circuit.copy())
    circuit_wires = circuit.qubits + circuit.clbits

    # remove the cutting edges and obtain the node ids connected a cutting edge
    graph, nodeids = _remove_cut_edges(graph, cuts)

    # divide a graph into subgraphs, generate new DAG graph instances for subgraphs
    # and save the preperation/measurement qubits in every subgraph
    subgraphs, prep_qubits, meas_qubits, ori_stitches = _disjoint_graph(graph, nodeids, cuts)
    # check the validation of graph partitioning, if not, return empty dataset
    if not subgraphs:
        return [], {}, {}
    # identify unused qubits, if there are, raise WARNING
    unused_qubits = []
    for qubit in circuit.qubits:
        qubit_found = any(qubit in node.qargs for node in graph.topological_op_nodes())
        if qubit_found: continue
        unused_qubits.append(qubit)
    if unused_qubits:
        print("WARNING: some qubits are entirely unused")
        print("unused qubits:",unused_qubits)

    # the shots on every subcircuit should be total_shots/num_variants
    variants_shots = num_shots // fragment_variants(cuts)
    # if the number of shots is too small, fix it to 1000
    variants_shots = max(variants_shots,1000)

    # store subcircuits data (in SubCircuit class) and qubit maps from full_circuit to subcircuits
    subcircuits, subgraph_wire_maps = zip(*[_trimmed_circuit(subgraph, subidx, prep_qubit, meas_qubit, variants_shots,
                                f'sub_{str(subidx)}q', f'sub_{str(subidx)}c') for subgraph, subidx, prep_qubit, meas_qubit in 
                                zip(subgraphs, range(len(subgraphs)), prep_qubits, meas_qubits)])

    wire_path_map = { circuit_wire : tuple(subgraph_wire_map[circuit_wire] for subgraph_wire_map in 
                        subgraph_wire_maps if circuit_wire in subgraph_wire_map.keys() ) 
                        for circuit_wire in circuit_wires }
    
    # obtain stitches among subcircuits
    stitches = _collect_stitches(ori_stitches,wire_path_map)
    
    return subcircuits, wire_path_map, stitches

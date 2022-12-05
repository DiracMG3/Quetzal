from qiskit import *
import numpy

##########################################################################################
#                  this script is used for generating quantum circuit
#                       developed using qiskit version 0.33
##########################################################################################

def random_unitary(qubits):
    return qiskit.quantum_info.random.random_unitary(2**qubits)

def ghz_circuit(qubits):
    '''
    construt a circuit that prepares a multi-qubit GHZ state
    '''
    qreg = QuantumRegister(qubits, "q") # quantum register
    circuit = QuantumCircuit(qreg) # initialize a trivial circuit

    # the GHZ circuit itself
    circuit.h(circuit.qubits[0])
    for qq in range(len(circuit.qubits)-1):
        circuit.cx(circuit.qubits[qq], circuit.qubits[qq+1])

    return circuit

def random_cascade_circuit(qubits, layers, seed = None):
    '''
    construct a cascade circuit of random local 2-qubit gates
    '''
    if seed is not None: numpy.random.seed(seed)
    qreg = QuantumRegister(qubits, "q")
    circuit = QuantumCircuit(qreg)

    # the random unitary circuit itself
    for _ in range(layers):
        for qq in range(len(circuit.qubits)-1):
            circuit.append(random_unitary(2), [ qreg[qq], qreg[qq+1] ])

    return circuit

def random_dense_circuit(qubits, layers, seed = None):
    '''
    construct a dense circuit of random local 2-qubit gates
    '''
    if seed is not None: numpy.random.seed(seed)
    qreg = QuantumRegister(qubits, "q")
    circuit = QuantumCircuit(qreg)

    # the random unitary circuit itself
    for _ in range(layers):
        for odd_links in range(2):
            for qq in range(odd_links, qubits-1, 2):
                circuit.append(random_unitary(2), [ qreg[qq], qreg[qq+1] ])

    return circuit

def random_clustered_circuit(qubits, layers, cluster_connectors, seed = None):
    '''
    construct a dense circuit of random local 2-qubit gates
    '''
    if seed is not None: numpy.random.seed(seed)
    qreg = QuantumRegister(qubits, "q")
    circuit = QuantumCircuit(qreg)

    clusters = len(cluster_connectors)+1
    boundaries = [ -1 ] + cluster_connectors + [ qubits ]

    def intra_cluster_gates():
        for cc in range(clusters):
            cluster_qubits = qreg[ boundaries[cc]+1 : boundaries[cc+1]+1 ]
            circuit.append(random_unitary(len(cluster_qubits)), cluster_qubits)

    def inter_cluster_gates():
        for cc in range(clusters-1):
            connecting_qubits = qreg[ boundaries[cc+1] : boundaries[cc+1]+2 ]
            circuit.append(random_unitary(2), connecting_qubits)

    # the random unitary circuit itself
    intra_cluster_gates()
    for _ in range(layers):
        inter_cluster_gates()
        intra_cluster_gates()

    return circuit


def build_circuit_with_cuts(circuit_type, layers, qubits, fragments, seed = 0):
    '''
    build a named circuit with cuts
    '''
    cut_qubits = [ qubits*ff//fragments-1 for ff in range(1,fragments) ]

    if circuit_type == "GHZ":
        circuit = ghz_circuit(qubits)
        cuts = [ (circuit.qubits[idx * qubits//fragments], 1)
                 for idx in range(1,fragments) ]

    elif circuit_type == "cascade":
        circuit = random_cascade_circuit(qubits, layers, seed)
        cuts = [ (circuit.qubits[qubit], loc)
                 for qubit in cut_qubits
                 for loc in range(1,2*layers) ]

    elif circuit_type == "dense":
        circuit = random_dense_circuit(qubits, layers, seed)
        cuts = [ (circuit.qubits[qubit], loc)
                 for qubit in cut_qubits
                 for loc in range(1,2*layers) ]

    elif circuit_type == "clustered":
        circuit = random_clustered_circuit(qubits, layers, cut_qubits, seed)
        cuts = [ (circuit.qubits[qubit], 1 + 2*layer + side)
                 for qubit in cut_qubits
                 for layer in range(layers)
                 for side in [ 0, 1 ] ]

    else:
        raise TypeError("circuit type not recognized: {circuit_type}")

    return circuit, cuts

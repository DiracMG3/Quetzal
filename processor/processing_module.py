from qiskit.ignis.verification.tomography import process_tomography_circuits
import qiskit,itertools,ast,numpy


##########################################################################################
#       this script provides the class and functions for processing subcircuits data
#                        developed using qiskit version 0.33
##########################################################################################
prep_state_keys = { "Pauli" : [ "Zp", "Zm", "Xp", "Yp" ],
                    "SIC" : [ "S0", "S1", "S2", "S3" ] }
meas_state_keys = { "Pauli" : [ "Zp", "Zm", "Xp", "Xm", "Yp", "Ym" ] }

class SubCircuit():
    '''
    A dataclass to store the information of subcircuits after circuit cutting.

    Args:
        circuit (qiskit `QuantumCircuit` object): the QuantumCircuit instance to store the circuit 
            data of a subcircuit.
        sub_id (int): the index of subcircuit.
        prep_qubits (Sequence[Qubit]): Sequence of `Qubit` indicating the preperation qubits on this
            subcircuit.
        meas_qubits (Sequence[Qubit]): Sequence of `Qubit` indicating the measurement qubits on this
            subcircuit.
        shots (int): total shots for sampling on this subcircuit 
    
    Attrs:
        circuit : return the `QuantumCircuit` object of subcircuit.
        sub_id : return the index of subcircuit.
        prep_qubits : return the preperation qubits on this subcircuit.
        meas_qubits : return the measurement qubits on this subcircuit.
        shots : return total shots.

    Methods:
        partial_tomography() : perform partial quantum process tomography on corresponding
            prep/meas qubits, return tomography raw data.

    '''
    def __init__(self, circuit, sub_id, prep_qubits, meas_qubits, shots):
        self.circuit = circuit
        self.sub_id = sub_id
        self.prep_qubits = prep_qubits
        self.meas_qubits = meas_qubits
        self.shots = shots

    def partial_tomography(self, prep_basis = "SIC",
                       backend = "qasm_simulator", monitor_jobs = False):
        '''
        perform partial quantum process tomography on this subcircuit 
        and return the corresponding raw data
        '''
        if self.prep_qubits is None: self.prep_qubits = []
        if self.meas_qubits is None: self.meas_qubits = []
        if self.prep_qubits == "all": self.prep_qubits = self.circuit.qubits
        if self.meas_qubits == "all": self.meas_qubits = self.circuit.qubits
        total_qubit_num = len(self.circuit.qubits)

        prep_qubits = list(map(qubit_index, self.prep_qubits))
        meas_qubits = list(map(qubit_index, self.meas_qubits))

        # define preparation states and measurement bases
        prep_states = prep_state_keys[prep_basis]
        # To implement "I" measurement, we directly measure the qubit in the Z-basis and do classical post-processing.
        meas_ops = [ "Z", "X", "Y" ]

        # collect preparation / measurement labels for all circuit variants
        def _reorder(label, positions):
            return tuple( label[positions.index(qq)] for qq in range(len(label)) )

        def _full_label(part_label, part_qubits, pad):
            qubit_order = list(part_qubits) + [ qq for qq in range(total_qubit_num)
                                                if qq not in part_qubits ]
            full_pad = [pad] * ( total_qubit_num - len(part_qubits) )
            return _reorder(list(part_label) + full_pad, qubit_order)

        def _full_prep_label(prep_label):
            return _full_label(prep_label, prep_qubits, prep_states[0])

        def _full_meas_label(meas_label):
            return _full_label(meas_label, meas_qubits, "Z")

        prep_labels = list(itertools.product(prep_states, repeat = len(self.prep_qubits)))
        meas_labels = list(itertools.product(meas_ops, repeat = len(self.meas_qubits)))

        # define full preparation / measurment labels on *all* qubits
        full_prep_labels = list(map(_full_prep_label, prep_labels))
        full_meas_labels = list(map(_full_meas_label, meas_labels))

        # collect circuit variants for peforming tomography
        tomo_circuits = process_tomography_circuits(self.circuit, self.circuit.qubits,
                                        prep_basis = prep_basis,
                                        prep_labels = full_prep_labels,
                                        meas_labels = full_meas_labels)

        circuit_result = run_circuits(tomo_circuits, self.shots, backend, monitor_jobs = monitor_jobs)

        print('tomography done!')
        return circuit_result
        
        

##########################################################################################
# functions for organizing and collecting associated data
##########################################################################################

def qubit_index(qubit):
    '''
    convert qubit objects to qubit indices (i.e. in a quantum register)
    '''
    return qubit if type(qubit) is int else qubit.index

def naive_fix(dist):
    '''
    fix the fragment data by discard the negative probability and normalize the distribution
    '''
    norm = sum( value for value in dist.values() if value >= 0 )
    return { bits : value / norm for bits, value in dist.items() if value >= 0 }

def get_statevector(circuit):
    '''
    get the quantum state prepared by a circuit
    '''
    simulator = qiskit.Aer.get_backend("statevector_simulator")
    sim_job = qiskit.execute(circuit, simulator)
    return sim_job.result().get_statevector(circuit)

def run_circuits(circuits, shots, backend = "qasm_simulator",
                 max_hardware_shots = 8192, monitor_jobs = False):
    '''
    run circuits and get the resulting probability distributions
    '''

    # get results from a single run
    def _results(shots, backend):
        tomo_job = qiskit.execute(circuits, backend = backend, shots = shots)
        if monitor_jobs: qiskit.tools.monitor.job_monitor(tomo_job)
        return tomo_job.result()

    if type(backend) is str:
        # if the backend is a string, simulate locally with a qiskit Aer backend
        backend = qiskit.Aer.get_backend(backend)
        backend._configuration.max_shots = shots
        return [ _results(shots, backend) ]

    else:
        # otherwise, we're presumably running on hardware,
        #   so only run as many shots at a time as we're allowed
        max_shot_repeats = shots // max_hardware_shots
        shot_remainder = shots % max_hardware_shots
        shot_sequence = [ max_hardware_shots ] * max_shot_repeats \
                      + [ shot_remainder ] * ( shot_remainder > 0 )
        return [ _results(shots, backend) for shots in shot_sequence ]

def fragment_variants(cuts):
    '''
    Return:
        The total munber of variants.

    Note the reconstruction of a single cut (channel) contains 4 sub-channels of minimal (see arXiv:2012.02333 or arXiv:2005.12702),
    which needs at least 4 state preperations and 3 measrements.
    '''
    return 4**len(cuts) * 3**len(cuts)

def label_to_matrix(label):
    '''
    convert string label to a matrix
    '''
    if label == "I": return numpy.eye(2)

    bases = qiskit.ignis.verification.tomography.basis
    if label[0] in [ "X", "Y", "Z" ]:
        matrix = bases.paulibasis.pauli_preparation_matrix
        if len(label) == 1:
            return matrix(f"{label}p") - matrix(f"{label}m")
        else:
            return matrix(label)
    if label[0] == "S":
        return bases.sicbasis.sicpovm_preparation_matrix(label)

    raise ValueError(f"label not recognized: {label}")

def target_labels_to_matrix(targets):
    '''
    convert a tuple of preparation / measurement labels into a choi matrix element
    '''
    try:
        prep_labels, meas_labels = targets["prep"], targets["meas"]
    except:
        prep_labels, meas_labels = targets
    prep_matrix = numpy.array(1)
    meas_matrix = numpy.array(1)
    for label in prep_labels:
        prep_matrix = numpy.kron(label_to_matrix(label), prep_matrix)
    for label in meas_labels:
        meas_matrix = numpy.kron(label_to_matrix(label), meas_matrix)
    return numpy.kron(prep_matrix.T, meas_matrix)

def to_projector(vector):
    '''
    convert a statevector into a density operator
    '''
    return numpy.outer(numpy.conjugate(vector), vector)

def bit_axis_permutation(subcircuits, wire_path_map):
    '''
    the qubits generated in subcircuits may not order in the original way as
    in primitive full circuit. We need to get the permutation to apply to the
    tensor factors of a united distribution
    '''
    bit_permutation = []
    for idx, subcircuit in enumerate(subcircuits):
        for qubit in subcircuit.circuit.qubits:
            if qubit not in subcircuit.meas_qubits:
                bit_permutation.extend(qubit_index(list(wire_path_map.keys()).index(ori_qubit))
                for ori_qubit in wire_path_map if (idx, qubit) in wire_path_map[ori_qubit])

    return bit_permutation

def organize_tomography_data(raw_data_collection, prep_qubits, meas_qubits, prep_basis):
    '''
    organize raw tomography data into a dictionary of dictionaries.

    mapping:
        bitstrings on the "final" qubits
        --> prepared / measured state labels
        --> observed counts

    Returns:
        { <final bitstring> : { <prepared_measured_states> : <counts> } }
    '''
    prep_qubits = list(map(qubit_index, prep_qubits))
    meas_qubits = list(map(qubit_index, meas_qubits))

    # split a bitstring on all qubits into:
    #   (1) a bitstring on the "middle" qubits that are associated with a cut, and
    #   (2) a bitstring on the "final" qubits that are *not* associated with a cut
    def _split_bits(bits):
        # the final output of bitstring is from right to left in qiskit, 
        # so we need to reverse the order to fit the postprocessing
        qubits = bits[::-1]
        mid_bits = "".join([ qubits[idx] for idx in meas_qubits ])
        fin_bits = "".join([ bit for pos, bit in enumerate(qubits)
                                if pos not in meas_qubits ])
        
        return mid_bits, fin_bits

    organized_data = {}
    for raw_data in raw_data_collection:
        for result in raw_data.results:
            name = result.header.name
            meas_counts = raw_data.get_counts(name)
            full_prep_label, full_meas_label = ast.literal_eval(name)
            prep_label = tuple( full_prep_label[qubit] for qubit in prep_qubits )
            meas_label = tuple( full_meas_label[qubit] for qubit in meas_qubits )
            for bits, counts in meas_counts.items():
                meas_bits, final_bits = _split_bits(bits)
                meas_state = tuple( basis + ( "p" if outcome == "0" else "m" )
                                    for basis, outcome in zip(meas_label, meas_bits) )
                count_label = ( prep_label, meas_state )
                if final_bits not in organized_data:
                    organized_data[final_bits] = {}
                organized_data[final_bits][count_label] = counts

    # add zero count data for output strings with missing prep/meas combinations
    prep_labels = itertools.product(prep_state_keys[prep_basis], repeat = len(prep_qubits))
    meas_states = itertools.product(meas_state_keys["Pauli"], repeat = len(meas_qubits))
    count_labels = list(itertools.product(prep_labels, meas_states))
    for bits in organized_data:
        if len(organized_data[bits]) == len(count_labels): continue
        for count_label in count_labels:
            if count_label not in organized_data[bits]:
                organized_data[bits][count_label] = 0

    print('collect tomo data done!')
    return organized_data

def collect_subcircuit_data(subcircuits, backend = "qasm_simulator",
                          prep_basis = "SIC", monitor_jobs = False):
    '''
    perform process tomography on all subcircuits and return the corresponding data
    '''
    subcirc_raw_data = [ subcircuit.partial_tomography(prep_basis = prep_basis,
                       backend = backend, monitor_jobs = monitor_jobs)
                      for subcircuit in subcircuits ]


    for idx, subcircuit in enumerate(subcircuits):
        print(f'subcircuit {idx}:')
        print('prep_qubits')
        print(subcircuit.prep_qubits)
        print('meas_qubits')
        print(subcircuit.meas_qubits)
        print()

    return [ organize_tomography_data(raw_data,
                                      subcircuit.prep_qubits,
                                      subcircuit.meas_qubits,
                                      prep_basis = prep_basis)
             for raw_data , subcircuit in zip(subcirc_raw_data, subcircuits) ]


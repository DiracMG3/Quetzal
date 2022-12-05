import numpy, scipy , tensornetwork
from processor.processing_module import *


########################################################################################################
#               this script collects some models for circuit recombination, including:
#                1. `direct_circuit_model` used by the original circuit cutting work 
#                           (Phys. Rev. Lett. 125, 150504 (2020).)
#                2. `maximum_likelihood_model` used by (npj Quantum Inf 7, 64 (2021).)
########################################################################################################

def direct_circuit_model(tomography_data, discard_poor_data = False, rank_cutoff = 1e-8):
    '''
    use tomography data to build a "naive" model (i.e. choi matrix) for a subcircuit.
    `tomography_data` should be a dictionary of dictionaries.

    Args:
        tomography_data (dict): { <final bitstring> : { <prepared_measured_states> : <counts> } }

    Returns:
        choi_matrix (dict): choi matrix data of different final bitstrings
    '''
    # if we were given a list of data sets, build a model for each data set in the list
    if type(tomography_data) is list:
        return [ direct_circuit_model(data_set, discard_poor_data, rank_cutoff)
                 for data_set in tomography_data ]
    
    # build a block-diagonal choi matrix from experiment data,
    #   where each block corresponds to a unique bitstring
    #   on the "final" outputs of a fragent
    choi_matrix = {}
    total_shots = 0  # used to normalize the choi matrix
    for final_bits, fixed_bit_data in tomography_data.items():
        prep_meas_states, state_counts = zip(*fixed_bit_data.items())
        prep_labels, meas_labels = zip(*prep_meas_states)
        prep_qubit_num = len(prep_labels[0])
        meas_qubit_num = len(meas_labels[0])
        if discard_poor_data:
            # if our system of equations defining this block of the choi matrix
            #   is underdetermined, don't bother fitting
            degrees_of_freedom = 4**( prep_qubit_num + meas_qubit_num )
            if len(fixed_bit_data) < degrees_of_freedom:
                print(f"discarding {sum(state_counts)} counts that define" +
                      " an underdetermined system of equations")
                continue

        # total number of cut qubits
        cut_qubit_num = prep_qubit_num + meas_qubit_num

        # collect data for fitting procedure, in which we will find a vector choi_fit
        #   that minimizes | state_matrix.conj() @ choi_fit - state_counts |
        state_matrix = numpy.array([ target_labels_to_matrix(states).flatten()
                                     for states in prep_meas_states ])
        state_counts = numpy.array(list(state_counts))

        # TODO: add count-adjusted weights to fitting procedure
        choi_fit = scipy.linalg.lstsq(state_matrix.conj(), state_counts, cond = rank_cutoff)[0]

        # save the fitted choi matrix
        choi_matrix[final_bits] = choi_fit.reshape(2**cut_qubit_num, 2**cut_qubit_num)

        total_shots += sum(state_counts)

    # normalize the choi matrix: tr(\tilde\Lambda) = 2**(#[quantum inputs])
    variants = 4 ** prep_qubit_num * 3 ** meas_qubit_num
    shots_per_variant = total_shots / variants
    choi_matrix = {bits : mat/shots_per_variant for bits, mat in choi_matrix.items()}

    print('direct choi process done!')
    return choi_matrix


def maximum_likelihood_model(choi_matrix):
    '''
    this method finds the closest nonnegative choi matrix to a "naive" one, is used in 
    (npj Quantum Inf 7, 64 (2021)) named `maximum-likelihood fragment tomography (MLFT),
    which is a generalization of `Maximum-likelihood State tomography (MLST)`
    (Phys. Rev. Lett. 108, 070502 (2012)) from state tomography to process tomography.
    '''
    # if we were given a list of models,
    #   then build maximum likelihood model for each data set in the list
    if type(choi_matrix) is list:
        return [ maximum_likelihood_model(mat) for mat in choi_matrix ]

    # diagonalize each block of the choi matrix
    choi_eigs = {}
    choi_vecs = {}
    for final_bits, choi_block in choi_matrix.items():
        choi_eigs[final_bits], choi_vecs[final_bits] = scipy.linalg.eigh(choi_block)

    # find the eigenvalues of the closest nonnegative choi matrix
    all_eigs = numpy.concatenate(list(choi_eigs.values()))
    eig_order = numpy.argsort(all_eigs)
    sorted_eigs = all_eigs[eig_order]
    dim = len(sorted_eigs)
    for idx in range(dim):
        val = sorted_eigs[idx]
        if val >= 0: break
        sorted_eigs[idx] = 0
        sorted_eigs[idx+1:] += val / ( dim - (idx+1) )
    reverse_order = numpy.arange(dim)[numpy.argsort(eig_order)]
    all_eigs = sorted_eigs[reverse_order]

    # organize eigenvalues back into their respective blocks
    num_blocks = len(choi_eigs)
    block_size = dim // num_blocks
    all_eigs = numpy.reshape(all_eigs, (num_blocks, block_size))
    choi_eigs = {bits: all_eigs[idx, :] for idx, bits in enumerate(choi_eigs.keys())
                if numpy.count_nonzero(all_eigs[idx, :]) != 0}


    # reconstruct choi matrix from eigenvalues / eigenvectors
    return { bits : sum( val * to_projector(choi_vecs[bits][:,idx])
                         for idx, val in enumerate(vals) if val > 0 )
             for bits, vals in choi_eigs.items() }


##########################################################################################
# functions for recombining collected subcircuits data
##########################################################################################

def _recombine_using_insertions(frag_models, stitches, bit_permutation, fragments):
    '''
    recombine subcircuits data by inserting a complete basis of operators
    '''
    frag_num = len(frag_models)
    print(bit_permutation)
    # identify permutation to apply to recombined fragment output
    final_bit_pieces = [ list(choi.keys()) for choi in frag_models ]

    combined_dist = {}
    for stitch_ops in itertools.product(["I","Z","X","Y"], repeat = len(stitches)):
        frag_ops = { idx : { "prep" : {} , "meas" : {} }
                     for idx in range(frag_num) }
        for stitch_op, stitch_qubits in zip(stitch_ops, stitches.items()):
            meas_frag_qubit, prep_frag_qubit = stitch_qubits
            meas_frag, meas_qubit = meas_frag_qubit
            prep_frag, prep_qubit = prep_frag_qubit
            meas_idx = fragments[meas_frag].meas_qubits.index(meas_qubit)
            prep_idx = fragments[prep_frag].prep_qubits.index(prep_qubit)
            frag_ops[meas_frag]["meas"][meas_idx] = stitch_op
            frag_ops[prep_frag]["prep"][prep_idx] = stitch_op

        frag_ops = [ frag_ops[idx] for idx in range(frag_num) ]
        def _ops_to_labels(ops_dict):
            labels = {}
            labels["prep"] \
                = tuple( ops_dict["prep"][idx] for idx in range(len(ops_dict["prep"])) )
            labels["meas"] \
                = tuple( ops_dict["meas"][idx] for idx in range(len(ops_dict["meas"])) )
            return labels

        frag_labels = list(map(_ops_to_labels, frag_ops))
        frag_mats = list(map(target_labels_to_matrix, frag_labels))

        for frag_bits in itertools.product(*final_bit_pieces):
            joined_bits = "".join(frag_bits)
            permuted_bits = "".join([ joined_bits[bit_permutation.index(order)] 
                                        for order in range(len(bit_permutation)) ])
            final_bits = "".join(permuted_bits[::-1])
            frag_vals = [ mat.flatten().conj() @ choi[bits].flatten()
                          for choi, bits, mat
                          in zip(frag_models, frag_bits, frag_mats) ]
            val = numpy.product(frag_vals).real
            try:
                combined_dist[final_bits] += val
            except:
                combined_dist[final_bits] = val

    return combined_dist

def _recombine_using_networks(frag_models, stitches, bit_permutation, fragments):
    '''
    recombine subcircuits data by building and contracting tensor networks
    '''
    frag_num = len(frag_models)
    print(bit_permutation)
    # identify permutation to apply to recombined fragment output
    final_bit_pieces = [ list(choi.keys()) for choi in frag_models ]

    combined_dist = {}
    for frag_bits in itertools.product(*final_bit_pieces):
        joined_bits = "".join(frag_bits)
        permuted_bits = "".join([ joined_bits[bit_permutation.index(order)] 
                            for order in range(len(bit_permutation)) ])
        final_bits = "".join(permuted_bits[::-1])
        nodes = {}
        for choi, bits, idx in zip(frag_models, frag_bits, range(frag_num)):
            matrix = choi[bits]
            qubits =  (len(bin(matrix.shape[0]))-3)
            tensor = matrix.reshape((2,)*2*qubits)

            prep_qubits = len(fragments[idx].prep_qubits)
            prep_axes_bra = reversed(range(prep_qubits))
            meas_axes_ket = reversed(range(prep_qubits,qubits))
            prep_axes_ket = reversed(range(qubits,qubits+prep_qubits))
            meas_axes_bra = reversed(range(qubits+prep_qubits,2*qubits))
            prep_axes = numpy.array(list(zip(prep_axes_bra, prep_axes_ket))).flatten()
            meas_axes = numpy.array(list(zip(meas_axes_ket, meas_axes_bra))).flatten()
            tensor = tensor.transpose(list(prep_axes) + list(meas_axes))
            tensor = tensor.reshape((4,)*qubits)

            nodes[idx] = tensornetwork.Node(tensor)

        for meas_frag_qubit, prep_frag_qubit in stitches.items():
            meas_frag, meas_qubit = meas_frag_qubit
            prep_frag, prep_qubit = prep_frag_qubit

            meas_qubit_idx = fragments[meas_frag].meas_qubits.index(meas_qubit)
            prep_qubit_idx = fragments[prep_frag].prep_qubits.index(prep_qubit)

            prep_axis = prep_qubit_idx
            meas_axis = len(fragments[meas_frag].prep_qubits) + meas_qubit_idx
            nodes[meas_frag][meas_axis] ^ nodes[prep_frag][prep_axis]

        val = tensornetwork.contractors.greedy(nodes.values()).tensor.real
        try:
            combined_dist[final_bits] += val
        except:
            combined_dist[final_bits] = val

    return combined_dist

def recombine_circuit_models(*args, method = "network", **kwargs):
    print('recombining')
    if method == "network":
        recombination_method = _recombine_using_networks
    elif method == "insertion":
        recombination_method = _recombine_using_insertions
    else:
        raise ValueError("recombination method {method} not recognized")
    combined_dist = recombination_method(*args, **kwargs)
    combined_norm = sum(combined_dist.values())
    return { bits : val / combined_norm for bits, val in combined_dist.items() }

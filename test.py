from qiskit import *
from processor.processing_module import *
from cutter.cutting_module import *
from cutter.kahypar_cutter import *
from processor.circuit_generator import *
from qiskit.circuit import Qubit
from qiskit.circuit.library import *
import matplotlib.pyplot as plt
from qiskit.circuit.random import random_circuit



# initial configurations
simulation_backend = "qasm_simulator"
num_shots = 10**6 # number of shots
cut_constraint = {"max_size_fragments": 10, 
                  "min_qubits": 2,
                  "max_qubits": 15,
                  "depth": 100,
                }


##########################################################################################
# test a circuit cutting task
##########################################################################################
#circuit, cuts = build_circuit_with_cuts(circuit_type, layers, qubits, frag_num, seed)
circ = random_circuit(20, 3, max_operands=2, seed = 23)
circ = circ.decompose()
circ1 = QFT(14, approximation_degree=11, do_swaps=False)
circ1 = circ1.decompose()
#circ1.draw('mpl')
#plt.show()
#cuts = [(Qubit(QuantumRegister(5, 'q'), 2),2),(Qubit(QuantumRegister(5, 'q'), 3),3)]
cuts = kahypar_cut(circ, 7, 0.88)
#print(cuts)
#print('len of cuts:')
#print(len(cuts))
task1 = Circuit_Cutting_Task(circ, cuts, num_shots, simulation_backend)
#problem = Optimal_Cut(circ,num_shots,cut_constraint,"kahypar")
#print(problem.results)

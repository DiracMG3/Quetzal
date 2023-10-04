from qiskit import *
from processor.processing_module import *
from cutter.cutting_module import *
from cutter.kahypar_cutter import *
from processor.circuit_generator import *
from qiskit.circuit.library import *
from qiskit.circuit.random import random_circuit
import time
import psutil, os


# initial configurations
simulation_backend = "qasm_simulator"
num_shots = 10**5 # number of shots
cut_constraint = {"max_size_fragments": 15, 
                  "min_qubits": 2,
                  "max_qubits": 20,
                  "depth": 100,
                }


##########################################################################################
# test a circuit cutting task
##########################################################################################
num_qubits = 20

t1 = time.time()
circ = random_circuit(num_qubits, 3, max_operands=2, seed = 24)
circ = circ.decompose()

#circ1 = QFT(num_qubits, approximation_degree=23, do_swaps=False)
#circ1 = circ1.decompose()

#print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
#print()

# a complete circuit cutting task workflow
task1 = Circuit_Cutting_Task(circ, cut_constraint, num_shots, simulation_backend)
task1.find_optimal_cut()
if len(task1.circuit.qubits) <= task1.limited_qubits:
    task1.statevector_simulation()
task1.cut_procedure()
task1.circuit_synthesis()
task1.reconstruct_procedure()
task1.full_circuit_simulation()
task1.compute_fidelity()
task1.print_results()

t2 = time.time()
used_time = t2 -t1
#print("总程序耗时: ", used_time)
#print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
info = psutil.virtual_memory()
#print( u'电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024) )
#print(u'当前使用的总内存占比：',info.percent)
#print(u'cpu个数：',psutil.cpu_count())



	

  
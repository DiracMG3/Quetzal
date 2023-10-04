import copy
import numpy
import random
import math
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
from qiskit.providers.fake_provider import FakeTokyo

graph = []  # 定义耦合图的集合
quantum_circuit = []  # 定义逻辑电路
result_circuit = []  # 定义实际电路
qubits = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 定义逻辑比特映射,qubits每一位的位置代表一个逻辑比特，每一位的值代表映射到物理比特的位置。例如qubits[1]=0表示逻辑比特1映射到了物理比特0
positions = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  # 定义耦合图上的点的映射,未被映射的点被标记为-1
distance_table = numpy.zeros((20, 20), int)
layers = []  # 定义函数的层
pi = math.pi


def tokyo(): # 定义tokyo的耦合图
    graph.clear()
    graph.append([0, 1])
    graph.append([1, 2])
    graph.append([2, 3])
    graph.append([3, 4])
    graph.append([0, 5])
    graph.append([1, 6])
    graph.append([2, 7])
    graph.append([3, 8])
    graph.append([3, 9])
    graph.append([4, 8])
    graph.append([4, 9])
    graph.append([5, 6])
    graph.append([6, 7])
    graph.append([7, 8])
    graph.append([8, 9])
    graph.append([5, 10])
    graph.append([5, 11])
    graph.append([6, 10])
    graph.append([6, 11])
    graph.append([7, 12])
    graph.append([7, 13])
    graph.append([8, 12])
    graph.append([8, 13])
    graph.append([10, 11])
    graph.append([11, 12])
    graph.append([12, 13])
    graph.append([13, 14])
    graph.append([10, 15])
    graph.append([11, 16])
    graph.append([11, 17])
    graph.append([12, 16])
    graph.append([13, 18])
    graph.append([13, 19])
    graph.append([14, 18])
    graph.append([14, 19])
    graph.append([15, 16])
    graph.append([16, 17])


def mapping():  # 将逻辑比特映射到物理电路
    for i in range(len(qubits)):  # 遍历所有逻辑比特的映射
        qubits[i] = i  # 仅实现简单映射
        positions[qubits[i]] = i
    # print('positions为',positions)


def circuit_generation(num_gate, num_qubit):
    quantum_circuit = []
    nobg = 0
    for i in range(num_gate):
        gate = random.choices(['RX', 'RY', 'RZ', 'CX'], [1, 1, 1, 2])
        if gate[0] == 'RX' or gate[0] == 'RY' or gate[0] == 'RZ':
            gate.append(random.randint(0, num_qubit - 1))
            quantum_circuit.append(gate)
        else:
            while 1:
                a = random.randint(0, num_qubit - 1)
                b = random.randint(0, num_qubit - 1)
                if a != b:
                    break
            gate.append(a)
            gate.append(b)
            quantum_circuit.append(gate)
            nobg = nobg + 1
    print('随机生成的逻辑电路的双比特门个数为：', nobg)
    return quantum_circuit


def greed_search_length(start, end): #找到最短路径
    queue = [] #存储所有可能的路径
    route = [] #当前路径
    route.append(start)
    queue.append(route)
    length = 0 #路径长度
    next_nodes = [] #路径下一步拓展的点的集合

    def contains(list, node):
        for n in list:
            if n == node:
                return False
        return True

    while queue:
        route = queue[0]
        del queue[0]
        current = route[-1] #-1表示最末尾
        if current == end: #表示已经找到终点，结束循环
            length = len(route)
            break
        else:
            next_nodes.clear()
            for edge in graph:
                if edge[0] == current and contains(route, edge[1]):
                    next_nodes.append(edge[1])
                if edge[1] == current and contains(route, edge[0]):
                    next_nodes.append(edge[0])

            for next_node in next_nodes:
                #print('本轮拓展节点为', next_node)
                route1 = copy.deepcopy(route) #深拷贝
                #print('此时route1为',route1)
                route1.append(next_node)
                #print('此时route1为',route1)
                queue.append(route1)
                #print('此时queue为', queue)
    return length
def build_distance_table():
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i != j:
                distance_table[i][j] = greed_search_length(i, j)
    # print(distance_table) #输出两点距离矩阵


def routing(qc):
    nos = 0 #添加交换门的数量
    result_circuit.clear()  # 在遍历前清空结果电路
    def greed_search_route(start, end, obstacle):  # 找到最短路径
        solutions = []  # 定义为点与点之间最短路径的集合
        queue = []  # 存储所有可能的路径
        route = []  # 当前路径
        route.append(start)
        queue.append(route)
        length = 0  # 路径长度
        next_nodes = []  # 路径下一步拓展的点的集合
        solutions.clear()  # 每次寻找前清空上一次的结果

        def contains(list, node):
            for n in list:
                if n == node:
                    return False
            return True

        while queue:
            route = queue[0]
            del queue[0]
            current = route[-1]
            if current == end:  # 表示已经找到终点，结束循环
                length = len(route)
                solutions.append(route)
                # print('跳出循环时solutions为', solutions)
                break
            else:
                next_nodes.clear()
                for edge in graph:
                    if edge[0] == current and contains(route, edge[1]) and contains(route, obstacle):
                        next_nodes.append(edge[1])
                    if edge[1] == current and contains(route, edge[0]) and contains(route, obstacle):
                        next_nodes.append(edge[0])

                for next_node in next_nodes:
                    # print('本轮拓展节点为', next_node)
                    route1 = copy.deepcopy(route)  # 深拷贝
                    # print('此时route1为',route1)
                    route1.append(next_node)
                    # print('此时route1为',route1)
                    queue.append(route1)
                    # print('此时queue为', queue)

        while queue and len(queue[0]) == length:
            if queue[0][-1] == end:
                solutions.append(queue[0])
                # print('此时solutions为', solutions)
            del queue[0]
        # print('最终最短路径为', solutions)
        return solutions[0]
    def swap(node1, node2):  # node1、node2表示拓扑图上需要交换的两个物理比特所代表的逻辑比特映射
        positions[node1], positions[node2] = positions[node2], positions[node1]  # 交换物理比特的映射
        qubits[positions[node1]], qubits[positions[node2]] = positions.index(positions[node1]), positions.index(
            positions[node2])  # 交换逻辑比特的映射
        result_circuit.append(['swap', node1, node2])

    for gate in qc:
        # 表示是单比特门，则直接执行
        if gate[0] != 'iswap':
            result_circuit.append(gate)
        # 双比特门
        else:
            done = 0
            for edge in graph:
                # 如果双比特门的映射在耦合图上相邻的话，则直接执行
                if (qubits[gate[1]] == edge[0] and qubits[gate[2]] == edge[1]) or (qubits[gate[1]] == edge[1] and qubits[gate[2]] == edge[0]):
                    result_circuit.append(gate)
                    done = 1
            if done == 1:
                continue
            # 如果双比特门的映射在耦合图上不相邻的话，则添加SWAP门改变映射
            swap_route = {}
            swap_sum = 0
            number = 0
            for edge in graph:
                # number = 0初始化,对第一条边进行操作
                if number == 0:
                    sum1 = distance_table[qubits[gate[1]]][edge[0]] + distance_table[qubits[gate[2]]][edge[1]]
                    sum2 = distance_table[qubits[gate[1]]][edge[1]] + distance_table[qubits[gate[2]]][edge[0]]
                    swap_sum = min(sum1, sum2)
                    if swap_sum == sum1:
                        swap_route[0] = greed_search_route(qubits[gate[1]], edge[0], qubits[gate[2]])
                        swap_route[1] = greed_search_route(qubits[gate[2]], edge[1], qubits[gate[1]])
                    else:
                        swap_route[0] = greed_search_route(qubits[gate[1]], edge[1], qubits[gate[2]])
                        swap_route[1] = greed_search_route(qubits[gate[2]], edge[0], qubits[gate[1]])
                    number = 1
                # 找出距离和最小的路径
                else:
                    sum1 = distance_table[qubits[gate[1]]][edge[0]] + distance_table[qubits[gate[2]]][edge[1]]
                    sum2 = distance_table[qubits[gate[1]]][edge[1]] + distance_table[qubits[gate[2]]][edge[0]]
                    sum = min(sum1, sum2)
                    if sum < swap_sum:
                        swap_sum = sum
                        if swap_sum == sum1:
                            swap_route[0] = greed_search_route(qubits[gate[1]], edge[0], qubits[gate[2]])
                            swap_route[1] = greed_search_route(qubits[gate[2]], edge[1], qubits[gate[1]])
                        else:
                            swap_route[0] = greed_search_route(qubits[gate[1]], edge[1], qubits[gate[2]])
                            swap_route[1] = greed_search_route(qubits[gate[2]], edge[0], qubits[gate[1]])
            change = swap_route[0]
            for i in range(len(change) - 1):  # 进行交换操作
                swap(change[i], change[i + 1])
                nos +=1
            change = swap_route[1]
            for i in range(len(change) - 1):  # 进行交换操作
                swap(change[i], change[i + 1])
                nos +=1
            result_circuit.append(gate)
    return nos


def sa(qc, qubit = qubits, position = positions): # 模拟退火优化
    Tmax = 100
    Tmin = 1
    d = 0.98
    R = 100
    qubits = qubit
    positions = position

    def calculatingcost(gate):
        sum = 0
        for i in range(gate, gate+50):
            while gate > len(qc):
                break
            if qc[gate] == 'iswap':
                sum = sum + distance_table[qc[gate][1]][qc[gate][2]]*(50-i)/50
        return sum


    T = Tmax
    cost = float('inf')
    while T>=Tmin:
        i = 1
        while i <= R:
            i = i + 1
            qubits1 = qubits # 临时保留初始映射
            positions1 = positions
            swap = random.choice(graph) # 随机添加一个交换
            positions[swap[0]], positions[swap[1]] = positions[swap[1]], positions[swap[0]]  # 交换物理比特的映射
            qubits[positions[swap[0]]], qubits[positions[swap[1]]] = positions.index(positions[swap[0]]), positions.index(positions[swap[1]])
            ncost = calculatingcost(0)
            if ncost < cost:
                cost = ncost
            else:
                p = random.random()
                if p > math.exp((cost-ncost)/T): # 按一定概率接受较差的解
                    cost = ncost
                else:
                    qubits = qubits1
                    positions = positions1
        T = T*d
    return qubits


def convert_qiskit_circuit_to_circuit(circ):
    for g in circ:
        gate = []
        if g[0].name == 'cx': # CNOT门的情况,同时将cnot门拆成单比特门和iswap门的组合
            quantum_circuit.append(['rz', g[1][0].index, [-pi / 2]])
            quantum_circuit.append(['rx', g[1][1].index, [pi / 2]])
            quantum_circuit.append(['rz', g[1][1].index, [pi / 2]])
            quantum_circuit.append(['iswap', g[1][0].index, g[1][1].index])
            quantum_circuit.append(['rx', g[1][0].index, [pi / 2]])
            quantum_circuit.append(['iswap', g[1][0].index, g[1][1].index])
            quantum_circuit.append(['rz', g[1][1].index, [pi / 2]])
        else: # 单比特门的情况
            gate.append(g[0].name) # 得到单比特门的类型
            gate.append(g[1][0].index) # 得到单比特门作用的逻辑比特
            gate.append(g[0].params) # 得到旋转角
            quantum_circuit.append(gate)

def convert_circuit_to_qiskit_circuit():
    result = QuantumCircuit(20)
    for g in result_circuit:
        if g[0] == 'iswap':
            result.iswap(g[1], g[2])
        if g[0] == 'rx':
            result.rx(g[2][0], g[1])
        if g[0] == 'ry':
            result.ry(g[2][0], g[1])
        if g[0] == 'rz':
            result.rz(g[2][0], g[1])
        if g[0] == 'swap':
            result.iswap(g[1], g[2])
            result.rx(-pi / 2, g[2])
            result.iswap(g[1], g[2])
            result.rx(-pi / 2, g[1])
            result.iswap(g[1], g[2])
            result.rx(-pi / 2, g[2])
    print(result)

def compile(quantum_circuit, qubits):
    tokyo()  # 生成tokyo耦合图
    build_distance_table()  # 根据耦合图生成距离表
    mapping()  # 将逻辑比特按一一映射映射到物理电路
    qc = quantum_circuit
    nos = routing(qc)
    # print('添加swap门的个数为：', nos)
    # print('————————将电路反向执行————————')
    bestqubits = qubits  # 针对循环的情况找到最优的初始映射
    bestresult = nos
    for i in range(20):
        quantum_circuit = quantum_circuit[::-1]
        if i % 2 == 1:
            result = routing(qc)
            if result < bestresult:
                bestresult = result
                bestqubits = qubits

    qubits = bestqubits  # 找到循环中耗费交换门个数最少的映射情况
    n = 0
    for i in qubits:  # 更新物理比特映射
        positions[i] = n
        n = n + 1

    # print('————————对初始映射进行模拟退火————————')
    qubits = sa(qc)
    n = 0
    for i in qubits:  # 更新物理比特映射
        positions[i] = n
        n = n + 1
    nos = routing(qc)
    # print('经过编译后的电路为：')
    # print(result_circuit)

def synthesis(quantum_circuit, backend= FakeTokyo() ):
    pass_ = Unroller(['id', 'u', 'u1', 'u2', 'u3', 'cx'])
    pm = PassManager(pass_)
    new_circ = pm.run(quantum_circuit)
    new_circuit = transpile(new_circ, backend=backend)
    return new_circuit

if __name__ == '__main__':
    circ = QuantumCircuit(3) # 测试用电路，可自行定义更复杂电路
    circ.cx(0, 2)
    circ.rx(pi/2, 0)
    circ.ry(pi/2, 0)
    circ.rz(pi/2, 0)


    convert_qiskit_circuit_to_circuit(circ) # 接口函数1将输入的QuantumCircuit类转换成编译用格式，circ即为定义好的QuantumCircuit类电路
    compile(quantum_circuit, qubits) # 中间的编译过程
    convert_circuit_to_qiskit_circuit() # 接口函数2将编译后的电路转换为QuantumCircuit类电路，最终输出结果电路图。




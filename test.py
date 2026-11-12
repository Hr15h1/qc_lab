import numpy as np
from numpy import pi
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt


def qft_rotations(circuit, n):
    if n == 0:
        return circuit

    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi / 2 ** (n - qubit), qubit, n)

    qft_rotations(circuit, n)


def swap_registers(circuit, n):
    for qubit in range(n // 2):
        circuit.swap(qubit, n - qubit - 1)
    return circuit


def qft(circuit, n):
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit


n = 4


qc = QuantumCircuit(n)
qft_rotations(qc, n)
fig = qc.draw(output='mpl')
plt.show()


qc = QuantumCircuit(n)
qft(qc, n)
fig = qc.draw(output='mpl')
plt.show()


bits = [0, 1, 1, 0]
qc = QuantumCircuit(n)
for i, bit in enumerate(bits):
    if bit:
        qc.x(i)
fig = qc.draw(output='mpl')
plt.show()


sim = AerSimulator()
qc_init = qc.copy()
qc_init.save_statevector()
state_vector = sim.run(transpile(qc_init, sim)).result().get_statevector()
plot_bloch_multivector(state_vector)
plt.show()


qft(qc, n)
# qc.measure_all()
fig = qc.draw(output='mpl')
plt.show()


sim = AerSimulator()
qc_state = qc.copy()
qc_state.save_statevector()
statevector = sim.run(transpile(qc_state, sim)).result().get_statevector()


plot_bloch_multivector(statevector)
plt.show()


qc_qft_measure = qc.copy()
qc_qft_measure.measure_all()
job = sim.run(transpile(qc_qft_measure, sim), shots=2048)
result = job.result().get_counts()
plot_histogram(result, figsize=(15, 6))
plt.show()


inverse_qft = qft(QuantumCircuit(n), n).inverse()
qc.compose(inverse_qft, inplace=True)


qc.decompose()
fig = qc.draw(output='mpl')
plt.show()


qc_final = qc.copy()
qc_final.measure_all()
job1 = sim.run(transpile(qc_final, sim), shots=2048)
result1 = job1.result().get_counts()
plot_histogram(result1, figsize=(7, 6))
plt.show()

"""
  neuron.py

Module that defines a quantum neuron with two qubits

Dependencies:
-

Since:
- 01/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

def neuron(qc,weight1,weight2,weight3,weight4,first_qbit_index,first_classical_bit_index):
    """
    Quantum circuit for a neuron.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit_index (int): The first qubit of the three qubits to be used in the neuron.
    first_classical_bit_index (int): The first classical bit of the three classical bits to be used in the neuron.
    weight1,weight2,weight3,weight4 (float): The weights of the inputs to the neuron.
    """
    
    qc.ry(weight1,first_qbit_index)
    qc.ry(weight2,first_qbit_index+1)
    qc.cx(first_qbit_index,first_qbit_index+1)

    qc.ry(weight3,first_qbit_index)
    qc.ry(weight4,first_qbit_index+1)
    qc.cx(first_qbit_index,first_qbit_index+1)

    qc.x(first_qbit_index)
    qc.x(first_qbit_index+1)
    qc.ccx(first_qbit_index,first_qbit_index+1,first_qbit_index+2)
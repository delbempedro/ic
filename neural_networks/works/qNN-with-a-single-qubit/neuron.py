"""
  neuron.py

Module that defines a quantum neuron with a single qubit

Dependencies:
-

Since:
- 01/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

def neuron(qc,input1_value,input2_value,weight1,weight2,weight3,qbit_index,classical_bit_index):
    """
    Applies a quantum neuron operation to the given quantum circuit.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to which the quantum neuron operation is applied.
    input1_value (float): The value of the first input of the neuron.
    input2_value (float): The value of the second input of the neuron.
    weight1 (float): The weight of the first input of the neuron.
    weight2 (float): The weight of the second input of the neuron.
    weight3 (float): The bias of the neuron.
    qbit_index (int): The index of the qbit to which the quantum neuron operation is applied.
    classical_bit_index (int): The index of the classical bit to which the quantum neuron operation is applied.
    """

    qc.h(qbit_index)

    qc.rz(input1_value*weight1+weight3,qbit_index)
    qc.rx(input2_value*weight2+weight3,qbit_index)

    #qc.measure(qbit_index,classical_bit_index)
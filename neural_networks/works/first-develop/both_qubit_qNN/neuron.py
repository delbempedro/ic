"""
  neuron.py

Module that defines a single quantum neuron

Dependencies:
-

Since:
- 01/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

def one_qubit_neuron(qc,input1_value,input2_value,weight1,weight2,weight3,qbit_index):
    """
    Quantun circuit for a neuron with one qubit.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to which the quantum neuron operation is applied.
    input1_value (float): The value of the first input of the neuron.
    input2_value (float): The value of the second input of the neuron.
    weight1 (float): The weight of the first input of the neuron.
    weight2 (float): The weight of the second input of the neuron.
    weight3 (float): The bias of the neuron.
    qbit_index (int): The index of the qbit to which the quantum neuron operation is applied.
    """

    qc.h(qbit_index)

    qc.rz(input1_value*weight1+weight3,qbit_index)
    qc.rx(input2_value*weight2+weight3,qbit_index)

def two_qubit_neuron(qc,weight1,weight2,weight3,weight4,first_qbit_index):
    """
    Quantum circuit for a neuron with two qubits.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit_index (int): The first qubit of the three qubits to be used in the neuron.
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
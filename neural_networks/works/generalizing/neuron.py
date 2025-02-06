"""
  neuron.py

Module that defines a single quantum neuron

Dependencies:
-

Since:
- 02/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

def single_qubit_neuron(qc,inputs,weights,number_of_bits=2,qbit_index=0):
    """
    Applies a quantum neuron operation to the given quantum circuit.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to which the quantum neuron operation is applied.
    inputs (list of floats): The values of the inputs to the neuron.
    weight1 (float): The weight of the first input of the neuron.
    weight2 (float): The weight of the second input of the neuron.
    weight3 (float): The bias of the neuron.
    qbit_index (int): The index of the qbit to which the quantum neuron operation is applied.
    """

    qc.h(qbit_index)

    last_gate = ""
    for index in range(number_of_bits):
        
        if last_gate == "rz":
            qc.rx(inputs[index]*weights[index]+weights[-1],qbit_index)
            last_gate = "rx"
        elif last_gate == "rx":
            qc.ry(inputs[index]*weights[index]+weights[-1],qbit_index)
            last_gate = "ry"
        else:
            qc.rz(inputs[index]*weights[index]+weights[-1],qbit_index)
            last_gate = "rz"

    """qc.rz(inputs[0]*weight1+weight3,qbit_index)
    qc.rx(inputs[1]*weight2+weight3,qbit_index)"""

def multi_qubit_neuron(qc,weight1,weight2,weight3,weight4,first_qbit_index,first_classical_bit_index):
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
"""
  neuron.py

Module that defines a single quantum neuron

Dependencies:
-

Since:
- 11/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

def neuron(qc,input1_value,input2_value,weight1,weight2,first_qbit_index,first_classical_bit_index):
    """
    Quantum circuit for a sum of simple adder.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit_index (int): The first qubit of the three qubits to be used in the simple adder.
    first_classical_bit_index (int): The first classical bit of the three classical bits to be used in the simple adder.
    
    """
    static_angle = 0

    qc.h(first_qbit_index)
    qc.h(first_qbit_index+1)

    #qc.cu(input1_value,static_angle,static_angle,static_angle,[first_qbit_index],[first_qbit_index+1])
    qc.crx(input1_value,[first_qbit_index],[first_qbit_index+1])
    qc.x(first_qbit_index+1)
    #qc.cu(input2_value,static_angle,static_angle,static_angle,[first_qbit_index],[first_qbit_index+1])
    qc.crx(input2_value,[first_qbit_index],[first_qbit_index+1])

    #qc.cu(weight1,static_angle,static_angle,static_angle,[first_qbit_index],[first_qbit_index+1]).inverse()
    qc.crx(weight1,[first_qbit_index],[first_qbit_index+1]).inverse()
    qc.x(first_qbit_index+1)
    #qc.cu(weight2,static_angle,static_angle,static_angle,[first_qbit_index],[first_qbit_index+1]).inverse()
    qc.crx(weight2,[first_qbit_index],[first_qbit_index+1]).inverse()

    qc.h(first_qbit_index)
    qc.h(first_qbit_index+1)
    qc.x(first_qbit_index)
    qc.x(first_qbit_index+1)
    qc.ccx(first_qbit_index,first_qbit_index+1,first_qbit_index+2)
    qc.measure(first_qbit_index+2,first_classical_bit_index)

def bin_neuron(qc,input1_value,input2_value,weight1,weight2,first_qbit_index,first_classical_bit_index):
    """
    Quantum circuit for a neuron.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit_index (int): The first qubit of the three qubits to be used in the neuron.
    first_classical_bit_index (int): The first classical bit of the three classical bits to be used in the neuron.
    
    """
    if input1_value:
        qc.x(first_qbit_index)
    if input2_value:
        qc.x(first_qbit_index+1)

    qc.h(first_qbit_index)
    qc.h(first_qbit_index+1)

    qc.z(first_qbit_index+1)
    qc.cp(weight1,first_qbit_index,first_qbit_index+1)

    qc.z(first_qbit_index+1)
    qc.z(first_qbit_index)
    qc.cp(weight2,first_qbit_index,first_qbit_index+1)

    qc.h(first_qbit_index)
    qc.h(first_qbit_index+1)

    qc.x(first_qbit_index)
    qc.x(first_qbit_index+1)
    qc.ccx(first_qbit_index,first_qbit_index+1,first_qbit_index+2)
    qc.measure(first_qbit_index+2,first_classical_bit_index)

    #qc.cu(weight1,static_angle,static_angle,static_angle,[first_qbit_index],[first_qbit_index+1]).inverse()

def bin_neuron2(qc,input1_value,input2_value,weight1,weight2,weight3,first_qbit_index,first_classical_bit_index):
    """
    Quantum circuit for a neuron.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit_index (int): The first qubit of the three qubits to be used in the neuron.
    first_classical_bit_index (int): The first classical bit of the three classical bits to be used in the neuron.
    
    """
    if input1_value:
        qc.x(first_qbit_index)
    if input2_value:
        qc.x(first_qbit_index+1)

    qc.h(first_qbit_index)
    qc.h(first_qbit_index+1)

    qc.rz(weight1,first_qbit_index+1)
    qc.cz(first_qbit_index,first_qbit_index+1)

    qc.cz(first_qbit_index,first_qbit_index+1)
    qc.rz(weight2,first_qbit_index)
    qc.rz(weight3,first_qbit_index+1)

    qc.h(first_qbit_index)
    qc.h(first_qbit_index+1)

    qc.x(first_qbit_index)
    qc.x(first_qbit_index+1)
    qc.ccx(first_qbit_index,first_qbit_index+1,first_qbit_index+2)
    qc.measure(first_qbit_index+2,first_classical_bit_index)

def bin_neuron3(qc,input1_value,input2_value,weight1,weight2,weight3,weight4,first_qbit_index,first_classical_bit_index):
    """
    Quantum circuit for a neuron.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit_index (int): The first qubit of the three qubits to be used in the neuron.
    first_classical_bit_index (int): The first classical bit of the three classical bits to be used in the neuron.
    
    """
    if input1_value:
        qc.x(first_qbit_index)
    if input2_value:
        qc.x(first_qbit_index+1)

    qc.rz(weight1,first_qbit_index)
    qc.rz(weight2,first_qbit_index+1)
    qc.cx(first_qbit_index,first_qbit_index+1)

    qc.rz(weight3,first_qbit_index)
    qc.rz(weight4,first_qbit_index+1)
    qc.cx(first_qbit_index,first_qbit_index+1)

    """qc.x(first_qbit_index)
    qc.x(first_qbit_index+1)
    qc.ccx(first_qbit_index,first_qbit_index+1,first_qbit_index+2)
    qc.measure(first_qbit_index+2,first_classical_bit_index)"""
    qc.measure(first_qbit_index+1,first_classical_bit_index)
"""
  simple_adder.py

Module that create the struct simple adder in a quantum circuit.

This module is used by the current_circuit.py module to
run some circuits with needs simple adder operations.

Since:
- 09/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""


def adder(qc,first_qbit_index,first_classical_bit_index):
    """
    Quantum circuit for a sum of simple adder.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit_index (int): The first qubit of the three qubits to be used in the simple adder.
    first_classical_bit_index (int): The first classical bit of the three classical bits to be used in the simple adder.
    
    """
    #mesuare all qbits 0 and 1
    qc.measure(first_qbit_index,first_classical_bit_index)
    qc.measure(first_qbit_index+1,first_classical_bit_index+1)

    #apply CNOT on qbits 0 and 1 with qbit 2 as control
    qc.cx(first_qbit_index,first_qbit_index+2)
    qc.cx(first_qbit_index+1,first_qbit_index+2)

    #mesuare qbit 3 with contain the result of the sum of the qbits 0 and 1
    qc.measure(first_qbit_index+2,first_classical_bit_index+2)

def simple_carry_out(qc,first_qbit_index,first_classical_bit_index):
    """
    Quantum circuit for a carry out of a simple adder.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit_index (int): The first qubit of the four qubits to be used in the carry out.
    first_classical_bit_index (int): The first classical bit of the four classical bits to be used in the carry out.
    
    """
    #apply CCNOT on qbits 0 and 1 with qbit 4 as control
    qc.ccx(first_qbit_index,first_qbit_index+1,first_qbit_index+4)

    #measure qbit 4 with contain the carry out of the sum of the qbits 0 and 1
    qc.measure(first_qbit_index+4,first_classical_bit_index+3)

def simple_adder(qc,qbit1_value,qbit2_value,first_qbit_index,first_classical_bit_index):
    """
    Quantum circuit for a simple adder.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    qbit1_value (int): The value of the first qbit to be used in the simple adder.
    qbit2_value (int): The value of the second qbit to be used in the simple adder.
    first_qbit_index (int): The first qubit of the four qubits to be used in the carry out.
    first_classical_bit_index (int): The first classical bit of the four classical bits to be used in the carry out.
    
    """
    #invert the values of the qbits if the input is not 0 (as expected)
    if qbit1_value == 1:
        qc.x(first_qbit_index)
    if qbit2_value == 1:
        qc.x(first_qbit_index+1)

    #call the simple adder and carry out functions to do the simple adder
    adder(qc,first_qbit_index,first_classical_bit_index)
    simple_carry_out(qc,first_qbit_index,first_classical_bit_index)
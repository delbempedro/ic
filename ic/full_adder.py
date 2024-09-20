"""
  full_affer.py

Module that create the struct full adder in a quantum circuit.

This module is used by the current_circuit.py module to
run some circuits with needs full adder operations.

Since:
- 09/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do the necessary imports
import math

def half_adder(qc,first_qbit_index,first_classical_bit_index,carry_in):
    """
    Quantum circuit for a half adder.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit_index (int): The first qubit of the three qubits to be used in the half adder.
    first_classical_bit_index (int): The first classical bit of the four classical bits to be used in the half adder.
    carry_in (bool): If the carry in is used in the full adder.
    
    """
    #mesuare all qbits 0 if carry in are used because that implies it is the first full adder
    if carry_in: qc.measure(first_qbit_index,first_classical_bit_index)

    #mesuare all qbits 0, 1 and 2
    qc.measure(first_qbit_index+1,first_classical_bit_index+1)
    qc.measure(first_qbit_index+2,first_classical_bit_index+2)

    #apply CNOT on qbits 1 and 2 with qbit 3 as control
    qc.cx(first_qbit_index+1,first_qbit_index+3)
    qc.cx(first_qbit_index+2,first_qbit_index+3)

    #apply CNOT on qbits 0 and 3 with qbit 4 as control
    qc.cx(first_qbit_index,first_qbit_index+4)
    qc.cx(first_qbit_index+3,first_qbit_index+4)

    #mesuare qbit 4 with contain the result of the sum of the qbits 0, 1 and 2
    qc.measure(first_qbit_index+4,first_classical_bit_index+3)

def carry_out(qc,first_qbit_index,first_classical_bit_index):
    """
    Quantum circuit for a carry out.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit_index (int): The first qubit of the eight qubits to be used in the carry out.
    first_classical_bit_index (int): The first classical bit of the five classical bits to be used in the carry out.
    
    """
    #apply CCNOT on qbits 1 and 2 with qbit 5 as control
    qc.ccx(first_qbit_index+1,first_qbit_index+2,first_qbit_index+5)
    #apply CCNOT on qbits 0 and 3 with qbit 6 as control
    qc.ccx(first_qbit_index,first_qbit_index+3,first_qbit_index+6)

    #invert qbits 5 and 6
    qc.x(first_qbit_index+5)
    qc.x(first_qbit_index+6)

    #apply CCNOT on qbits 5 and 6 with qbit 7 as control
    qc.ccx(first_qbit_index+5,first_qbit_index+6,first_qbit_index+7)

    #reversible inversion of qbit 7
    qc.rx(math.pi,first_qbit_index+7)

    #measure qbit 7 with contain the carry out of the sum of the qbits 0, 1 and 2
    qc.measure(first_qbit_index+7,first_classical_bit_index+4)

def full_adder(qc,qbit1_value,qbit2_value,first_qbit_index,first_classical_bit_index,carry_in):
    """
    Quantum circuit for a full adder.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    qbit1_value (int): The value of the first qbit to be used in the full adder.
    qbit2_value (int): The value of the second qbit to be used in the full adder.
    carry_in (int): If the carry in is used in the full adder.
    first_qbit_index (int): The first qubit of the eight qubits to be used in the full adder.
    first_classical_bit_index (int): The first classical bit of the five classical bits to be used in the full adder.
    
    """
    #invert the values of the qbits if the input is not 0 (as expected)
    if qbit1_value:
        qc.x(first_qbit_index+1)
    if qbit2_value:
        qc.x(first_qbit_index+2)
    if carry_in:
        qc.x(first_qbit_index)

    #call the half adder and carry out functions to do the full adder
    half_adder(qc,first_qbit_index,first_classical_bit_index,carry_in)
    carry_out(qc,first_qbit_index,first_classical_bit_index)
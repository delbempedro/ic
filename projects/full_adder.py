#do the necessary imports
import math

def half_adder(qc,first_qbit,first_classical_bit):
    """
    Quantum circuit for a half adder.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit (int): The first qubit of the three qubits to be used in the half adder.
    first_classical_bit (int): The first classical bit of the four classical bits to be used in the half adder.
    
    """
    #mesuare all qbits 0, 1 and 2
    qc.measure(first_qbit,first_classical_bit+0)
    qc.measure(first_qbit+1,first_classical_bit+1)
    qc.measure(first_qbit+2,first_classical_bit+2)

    #apply CNOT on qbits 1 and 0 with qbit 3 as control
    qc.cx(first_qbit+1,first_qbit+3)
    qc.cx(first_qbit,first_qbit+3)

    #apply CNOT on qbits 2 and 3 with qbit 4 as control
    qc.cx(first_qbit+2,first_qbit+4)
    qc.cx(first_qbit+3,first_qbit+4)

    #mesuare qbit 4 with contain the result of the sum of the qbits 0, 1 and 2
    qc.measure(first_qbit+4,first_classical_bit+3)

def carry_out(qc,first_qbit,first_classical_bit):
    """
    Quantum circuit for a carry out.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit (int): The first qubit of the eight qubits to be used in the carry out.
    first_classical_bit (int): The first classical bit of the five classical bits to be used in the carry out.
    
    """
    #apply CCNOT on qbits 1 and 0 with qbit 5 as control
    qc.ccx(first_qbit+1,first_qbit,first_qbit+5)
    #apply CCNOT on qbits 2 and 3 with qbit 6 as control
    qc.ccx(first_qbit+2,first_qbit+3,first_qbit+6)

    #invert qbits 5 and 6
    qc.x(first_qbit+5)
    qc.x(first_qbit+6)

    #apply CcNOT on qbits 5 and 6 with qbit 7 as control
    qc.ccx(first_qbit+5,first_qbit+6,first_qbit+7)

    #reversible inversion of qbit 7
    qc.rx(math.pi,first_qbit+7)

    #measure qbit 7 with contain the carry out of the sum of the qbits 0, 1 and 2
    qc.measure(first_qbit+7,first_classical_bit+4)

def full_adder(qc,value1,value2,carry_in,first_qbit,first_classical_bit):
    """
    Quantum circuit for a full adder.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    value1 (int): The value of the first qbit to be used in the full adder.
    value2 (int): The value of the second qbit to be used in the full adder.
    carry_in (int): The value of the third qbit to be used in the full adder.
    first_qbit (int): The first qubit of the eight qubits to be used in the carry out.
    first_classical_bit (int): The first classical bit of the five classical bits to be used in the carry out.
    
    """
    #invert the values of the qbits if they are 1
    if value1 == 1:
        qc.x(first_qbit+1)
    if value2 == 1:
        qc.x(first_qbit+2)
    if carry_in == 1:
        qc.x(first_qbit)

    #call the half adder and carry out functions to do the full adder
    half_adder(qc,first_qbit,first_classical_bit)
    carry_out(qc,first_qbit,first_classical_bit)
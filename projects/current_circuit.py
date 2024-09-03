"""
  current_circuit.py

Module that defines the quantun current quantum circuit.

To run the main() function, call `julia main.jl` on the terminal.

Dependencies:
- Uses the full_adder.py module to create the struct full adder in a quantum circuit.

Since:
- 09/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do the necessary imports
from qiskit import QuantumCircuit
from full_adder import *

def current_circuit():
    """
    Create a quantum circuit with 16 qubits and 10 classical bits.
    
    The quantum circuit is composed of two full adders. The first full adder
    adds the qubits 0, 1 and 2 and stores the result in the classical bits 0, 1 and 2.
    The second full adder adds the qubits 7, 8 and 9 and stores the result in the classical bits 5, 6 and 7.
    
    Parameters:
    
    Returns:
        The quantum circuit with the two full adders.
    """
    #create quantum circuit with 7 qbits and 4 classical bits
    qc = QuantumCircuit(16,10)
    full_adder(qc,1,1,1,0,0)
    full_adder(qc,1,1,0,7,5)
    return qc
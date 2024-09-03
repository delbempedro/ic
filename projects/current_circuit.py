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
    
    The quantum circuit is composed of two full adders.
    
    Parameters:
    
    Returns:
        The quantum circuit with the two full adders.
    """
    #create quantum circuit with 7 qbits and 4 classical bits
    qc = QuantumCircuit(16,10)
    full_adder(qc,1,1,1,0,0)
    full_adder(qc,1,1,0,7,5)
    return qc
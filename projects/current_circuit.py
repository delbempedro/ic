"""
  current_circuit.py

Module that defines the quantun current quantum circuit.

Dependencies:
- Uses the full_adder.py module to create the class current_circuit which contains the struct full adder in a quantum circuit.

Since:
- 09/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do the necessary imports
from qiskit import QuantumCircuit # type: ignore
from full_adder import *
from simple_adder import *

class current_circuit():
    
    def __init__(self):
        """
        Create a new quantum circuit with two full adders.

        The circuit is built with 16 qubits and 10 classical bits.
        The first full adder is defined with qbits 0, 1 and 2 and classical bit 0 and 1.
        The second full adder is defined with qbits 7, 5 and 6 and classical bit 2 and 3.
        """
        self._num_of_qbits = 4
        self._num_of_classical_bits = 4
        self._qc = QuantumCircuit(self._num_of_qbits,self._num_of_classical_bits)

        #defines the circuit
        simple_adder(self._qc,1,1,0,0)
        #full_adder(self._qc,1,1,0,0,carry_in=True)
        #full_adder(self._qc,1,1,7,5)
        #full_adder(self._qc,1,1,14,10)
    
    def get_current_circuit(self):
        """
        Get the current quantum circuit.
        
        Returns:
        The current quantum circuit (QuantumCircuit).
        """
        return self._qc

    def get_num_of_qbits(self):
        """
        Get the number of qbits in the quantum circuit.
        
        Returns:
        The number of qbits in the quantum circuit (int).
        """
        return self._num_of_qbits

    def get_num_of_classical_bits(self):
        """
        Get the number of classical bits in the quantum circuit.
        
        Returns:
        The number of classical bits in the quantum circuit (int).
        """
        return self._num_of_classical_bits

    
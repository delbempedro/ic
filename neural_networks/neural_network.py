"""
  current_circuit.py

Module that defines the quantum current quantum circuit.

Dependencies:
- Uses current_circuit.py module to define and run the current quantum circuit
- Uses qiskit_ibm_runtime module to defines service of the quantum circuit

Since:
- 11/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do qiskit necessary imports
from current_circuit import *
from qiskit_ibm_runtime import QiskitRuntimeService # type: ignore

#defines service
service = QiskitRuntimeService()

#defines the input, weights and exprected output
input1 = 1.0
input2 = 1.0
weight1 = 0.5
weight2 = 0.5
expected_output = int(input1)^int(input2)

#initializes the quantum circuit
qc = current_circuit(3,1)

#adds the neuron to the circuit
qc.add_neuron(input1,input2,weight1,weight2,0,0)

#runs (simulates) the circuit and save the result
output = int(list(qc.run_circuit("3",service).keys())[0])
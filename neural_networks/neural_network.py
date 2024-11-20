"""
  current_circuit.py

Module that defines the quantum current quantum circuit.

Dependencies:
- Uses current_circuit.py module to define and run the current quantum circuit
- Uses qiskit_ibm_runtime module to defines service of the quantum circuit
- Uses math module to get the infinity value

Since:
- 11/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do qiskit necessary imports
from current_circuit import *
from qiskit_ibm_runtime import QiskitRuntimeService # type: ignore
import math

#defines service
service = QiskitRuntimeService()

#defines the inputs
input1 = 1.0
input2 = 1.0

#defines the weights
weight1 = 0.5
weight2 = 0.5

#defines the expected output
expected_output = int(input1)^int(input2)

#defines the threshold
threshold = 0.5

#defines the learning rate
learning_rate = 0.01

#initializes the error with infinity value
error = math.inf

#defines the bias and bias weight
"""bias = 1
bias_weight = 0.5"""

#defines the activation function
def activation_function(neuron_output):
    if neuron_output >= threshold:
        return 1
    else:
        return 0

iterations = 0
while not error == 0:
    iterations += 1

    #initializes the quantum circuit
    qc = current_circuit(3,1)

    #adds the neuron to the circuit
    qc.add_bin_neuron(input1,input2,weight1,weight2,0,0)

    #runs (simulates) the circuit and save the result
    neuron_output = int(list(qc.run_circuit("3",service).keys())[0])

    output = activation_function(neuron_output)

    error = expected_output - output

    print(f"Output: {output}")
    print(f"Error: {error}")
    print(f"Neuron output: {neuron_output}")
    print(f"Iterations: {iterations}")

    if not error == 0:
        print(f"Old weight1: {weight1}")
        print(f"Old weight2: {weight2}")
        print(f"Old bias_weight: {bias_weight}")
        weight1 += (learning_rate * input1 * error)
        weight2 += (learning_rate * input2 * error)
        bias_weight += (learning_rate * bias * error)
        print("I'm learning")
        print(f"New weight1: {weight1}")
        print(f"New weight2: {weight2}")
        print(f"New bias_weight: {bias_weight}")

print(f"Expected output: {expected_output}")
print(f"Output: {output}")
print(f"Error: {error}")
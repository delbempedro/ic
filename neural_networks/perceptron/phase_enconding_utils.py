"""
  phase_enconding_utils.py

Module that defines the util functions to train the quantum neural network.

Dependencies:
- Uses current_circuit module to define the neuron quantum circuit

Since:
- 04/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do my biblioteca necessary imports
from neuron_circuit import *

def generate_phase_qNN_circuit(inputs,parameters,number_of_inputs,number_of_inputs_per_qubit=3):
    """
    Generate a quantum neural network circuit (qNN) with a single neuron.

    Parameters:
    input1_value (float): The value of the first input of the neuron.
    input2_value (float): The value of the second input of the neuron.
    parameters (list of floats): The parameters of the neuron, in order: first input weight, second input weight, bias.
    number_of_inputs (int): The number of qbits in the circuit.
    number_of_inputs_per_qubit (int): The number of inputs per qubit.

    Returns:
    The qNN circuit (current_circuit).
    """

    #create the qNN circuit
    if number_of_inputs <= number_of_inputs_per_qubit:
        qNN = neuron_circuit(1,1)
    elif number_of_inputs%2: #if the number of qubits is even
        qNN = neuron_circuit(number_of_inputs//number_of_inputs_per_qubit+2,1)
    else: #if the number of qubits is odd
        qNN = neuron_circuit(number_of_inputs//number_of_inputs_per_qubit+1,1)

    qNN.add_phase_qubit_neuron(inputs, parameters, number_of_inputs=number_of_inputs, number_of_inputs_per_qubit=number_of_inputs_per_qubit) #add the neuron
    qNN.get_current_circuit().measure_all() #measure all qubits

    #return the circuit
    return qNN

def phase_qNN_compute_error(counts,expected_output):
    """
    Compute the error between the actual outputs and the expected outputs.

    Parameters:
    counts (list of dictionaries): The counts of the outputs of the quantum circuit.
    expected_outputs (list of floats): The expected outputs of the quantum circuit.

    Returns:
    The error (float).
    """

    #compute number of shots
    number_of_shots = sum(counts[0].values())
    
    #initialize error with 0
    error = 0

    #initialize total tests with 0
    total_tests = 0

    #compute error for each count
    for count in counts:
        for key,value in count.items():
            if key[-1] != str(expected_output):
                error += value
            total_tests += value

    #normalize error
    error = error/total_tests

    #return error
    return error



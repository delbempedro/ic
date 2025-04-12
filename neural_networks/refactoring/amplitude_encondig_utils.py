"""
  amplitude_enconding_utils.py

Module that defines the util functions to train the quantum neural network.

Dependencies:
- Uses qiskit.circuit.library module to define TwoLocal circuit
- Uses math module to define pi
- Uses current_circuit module to define the neuron quantum circuit

Since:
- 04/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""
#do qiskit necessary imports
from qiskit.circuit.library import TwoLocal #type: ignore

#do necessary imports
from math import pi

#do my biblioteca necessary imports
from neuron_circuit import *

def circuit_copy(initial_circuit, number_of_inputs):
    """
    Creates a quantum circuit that duplicates the given initial circuit with additional qubits.

    This function constructs a new quantum circuit by duplicating the specified number of qubits 
    from the initial circuit and adding two additional qubits to the circuit. The resulting circuit 
    reproduces the input values using controlled-NOT operations to copy the state of the first two 
    qubits to the additional qubits.

    Parameters:
    initial_circuit (QuantumCircuit): The initial quantum circuit to be copied.
    number_of_inputs (int): The number of qubits to duplicate from the initial circuit.

    Returns:
    QuantumCircuit: A new quantum circuit with duplicated qubits and additional operations to copy the input.
    """

    circuit_copy = QuantumCircuit(number_of_inputs*2) #duplicate the number of qubits
    circuit_copy = circuit_copy.compose(initial_circuit, qubits=list(range(0,number_of_inputs))) #first half with the initial_states_2b_circ
    circuit_copy.barrier() #to visually separate circuit components

    for index in range(number_of_inputs):
        circuit_copy.x(number_of_inputs+index)

    for index in range(number_of_inputs):
        circuit_copy.cx(index,number_of_inputs+index)

    for index in range(number_of_inputs):
        circuit_copy.x(number_of_inputs+index)

    return circuit_copy

#Generates a circuit with all the possible inputs
def all_inputs_circuit(number_of_inputs):
    """
    Generates a quantum circuit that produces all the possible inputs for a given number of qubits.

    The circuit is constructed using the TwoLocal method. The parameters are set to predefined values 
    to generate all possible inputs.

    Parameters:
    num_qubits (int): The number of qubits in the circuit.

    Returns:
    QuantumCircuit: The quantum circuit that generates all the possible inputs.
    """

    entanglement_circuit = TwoLocal(number_of_inputs, "rx", "cz", entanglement="linear", reps=1) #generates a circuit that mixture two qubits - an entanglement circuit
    parameters = [pi] * number_of_inputs + [pi / 2] * (number_of_inputs) #previously chosen/learned parameters to generate all the possible inputs
    parameter_dict = dict(zip(entanglement_circuit.parameters, parameters)) #assigns the parameters to the circuit
    initial_states_circuit = entanglement_circuit.assign_parameters(parameter_dict) #a circuit that can generate all the possible inputs

    #return the circuit
    return initial_states_circuit

def generate_amplitude_qNN_circuit(parameters_of_entanglement_circuit,number_of_inputs=2):
    """
    Generates a quantum circuit that produces all the possible inputs and a quantum neural network with two neurons.

    The circuit is constructed by using the TwoLocal method. The parameters are set to the previously chosen/learned parameters to generate all the possible inputs.
    The TwoLocal method generates a circuit that mixture two qubits - an entanglement circuit.
    A quantum neural network with two neurons is added to the circuit, by using the add_bin_neuron3 method of the current_circuit class.
    The parameters of the quantum neural network are set to the previously chosen/learned parameters.

    Parameters:
    parameters_of_entanglement_circuit (list): A list of parameters for the U and controlled-phase (cp) gates.
    number_of_inputs (int): The number of inputs.

    Returns:
    quantum_circuit (QuantumCircuit): The quantum circuit with all the possible inputs and a quantum neural network with two neurons.
    """

    number_of_qubits_required = number_of_inputs*2+1
    qNN = neuron_circuit(number_of_qubits_required,1) #create the qNN circuit
    auxiliary_circuit = all_inputs_circuit(number_of_inputs) #copy the all inputs circuit
    duplicate_circuit = circuit_copy(auxiliary_circuit, number_of_inputs) #duplicate the all inputs circuit
    qNN.get_current_circuit().append(duplicate_circuit, list(range(0,number_of_inputs*2))) #add the all inputs circuit
    qNN.add_amplitude_qubit_neuron(parameters_of_entanglement_circuit, number_of_inputs=number_of_inputs) #add the neuron
    qNN.get_current_circuit().measure_all()

    return qNN

def amplitude_qNN_compute_error(inputs,expected_outputs,counts,number_of_inputs=2):
    """
    Compute the error of the given quantum circuit.

    The error is computed by counting the number of mistakes in the outputs of the quantum circuit.
    The output of the quantum circuit is in the form of a string of length 5, where the first two
    characters are the inputs and the last character is the output. The error is the sum of the
    number of mistakes in the outputs of the quantum circuit divided by the total number of tests.

    Parameters:
    inputs (list): A list containing pairs of input values for the neuron.
    expected_outputs (list): A list of expected output values for each input pair.
    counts (list): A list of dictionaries, where each dictionary represents the counts of the outputs of the quantum circuit.
    number_of_inputs (int): The number of qbits in the quantum circuit.

    Returns:
    float: The error of the quantum circuit.
    """

    #define the statistics dictionary
    statistics = {}
    for i in range(2**number_of_inputs):
        binary_key = format(i, f'0{number_of_inputs}b')
        statistics[binary_key] = [0, 0]
    
    #get the total number of tests
    total_tests = 0

    # Processa as contagens
    for count in counts: #for each count
        for key, value in count.items(): #for each key and value

            #define the inputs and the output
            inputs = ''
            for bit in range(number_of_inputs, number_of_inputs*2):
                inputs += str(key[bit])
            output = int(key[number_of_inputs*2])

            #update the statistics
            statistics[inputs][output] += value

            #update the total number of tests
            total_tests += value

    #compute the error
    error = total_tests
    for input in list(statistics.keys()):
        error -= statistics[input][expected_outputs[input]]
        
    error /= total_tests

    return error

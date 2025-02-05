"""
  qNN.py

Module that defines the quantum current quantum circuit.

Dependencies:
- Uses qiskit.circuit.library module to generate a circuit with all the possible inputs
- Uses qiskit.primitives module to run the circuit

Since:
- 01/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#Do qiskit necessary imports 
from qiskit.circuit.library import TwoLocal #type: ignore
from qiskit.primitives import StatevectorSampler #type: ignore

#Do my biblioteca necessary imports
from current_circuit_two_qNN import *

#Do necesary other imports
import numpy as np #type: ignore
from math import pi

#Generates a circuit with all the possible inputs
def all_inputs_circuit():
    """
    Generates a quantum circuit that produces all the possible inputs.

    The circuit is constructed by using the TwoLocal method. The parameters are set to the previously chosen/learned parameters to generate all the possible inputs.

    Returns:
    quantum_circuit (QuantumCircuit): The quantum circuit that generates all the possible inputs.
    """

    two_qubits_entanglement_circuit = TwoLocal(2, "rx", "cz", entanglement="linear", reps=1) #generates a circuit that mixture two qubits - an entanglement circuit
    parameters = [pi, pi, pi/2, pi/2] #previously chosen/learned parameters to generate all the possible inputs
    parameter_dict = dict(zip(two_qubits_entanglement_circuit.parameters, parameters)) #assigns the parameters to the circuit
    initial_states_circuit = two_qubits_entanglement_circuit.assign_parameters(parameter_dict) #a circuit that can generate all the possible inputs

    #return the circuit
    return initial_states_circuit

def circuit_copy(initial_circuit, number_of_qubits):
    """
    Creates a quantum circuit that duplicates the given initial circuit with additional qubits.

    This function constructs a new quantum circuit by duplicating the specified number of qubits 
    from the initial circuit and adding two additional qubits to the circuit. The resulting circuit 
    reproduces the input values using controlled-NOT operations to copy the state of the first two 
    qubits to the additional qubits.

    Parameters:
    initial_circuit (QuantumCircuit): The initial quantum circuit to be copied.
    number_of_qubits (int): The number of qubits in the initial circuit.

    Returns:
    QuantumCircuit: A new quantum circuit with duplicated qubits and additional operations to copy the input.
    """

    circuit_copy = QuantumCircuit(number_of_qubits+2) #duplicate the number of qubits
    circuit_copy = circuit_copy .compose(initial_circuit, qubits=list(range(0,number_of_qubits))) #first half with the initial_states_2b_circ
    circuit_copy.barrier() #to visually separate circuit components

    #a layer to copy/reproduce the generated inputs in qubits 0 and 1
    circuit_copy.x(2)    #change value of qubit 2 from 0 to 1
    circuit_copy.x(3)    #change value of qubit 3 from 0 to 1
    circuit_copy.cx(0,2) #qb0 ''AND'' 1 (or NOT qb0) to copy qubit 0 to qubit 2
    circuit_copy.cx(1,3) #qb1 ''AND'' 1 (or NOT qb1) to copy qubit 1 to qubit 3
    circuit_copy.x(2)    #NOT of qubit 2 => qubit 2 equal to equal qubit 0
    circuit_copy.x(3)    #NOT of qubit 3 => qubit 3 equal to equal qubit 1

    return circuit_copy 

def qNN_circuit(all_inputs_circuit, parameters_of_entanglement_circuit):
    """
    Generates a quantum circuit that produces all the possible inputs and a quantum neural network with two neurons.

    The circuit is constructed by using the TwoLocal method. The parameters are set to the previously chosen/learned parameters to generate all the possible inputs.
    The TwoLocal method generates a circuit that mixture two qubits - an entanglement circuit.
    A quantum neural network with two neurons is added to the circuit, by using the add_bin_neuron3 method of the current_circuit class.
    The parameters of the quantum neural network are set to the previously chosen/learned parameters.

    Parameters:
    all_inputs_circuit (QuantumCircuit): The quantum circuit that generates all the possible inputs.
    parameters_of_entanglement_circuit (list): A list of parameters for the U and controlled-phase (cp) gates.

    Returns:
    quantum_circuit (QuantumCircuit): The quantum circuit with all the possible inputs and a quantum neural network with two neurons.
    """

    qNN = current_circuit(5,1) #create the qNN circuit
    auxiliary_circuit = all_inputs_circuit.copy() #copy the all inputs circuit
    duplicate_circuit = circuit_copy(auxiliary_circuit, 2) #duplicate the all inputs circuit
    qNN.get_current_circuit().append(duplicate_circuit, [0, 1, 2, 3]) #add the all inputs circuit
    qNN.add_four_angle_neuron(*parameters_of_entanglement_circuit, 2, 0) #add the neuron
    qNN.get_current_circuit().measure_all()

    return qNN

def error(counts):
    """
    Compute the error of the given quantum circuit.

    The error is computed by counting the number of mistakes in the outputs of the quantum circuit.
    The output of the quantum circuit is in the form of a string of length 5, where the first two
    characters are the inputs and the last character is the output. The error is the sum of the
    number of mistakes in the outputs of the quantum circuit divided by the total number of tests.

    Parameters:
    counts (list): A list of dictionaries, where each dictionary represents the counts of the outputs of the quantum circuit.

    Returns:
    float: The error of the quantum circuit.
    """

    #define the statistics dictionary
    statistics = {"00": [0,0], "01": [0,0], "10": [0,0], "11": [0,0]} #defines the statistics dictionary
    
    #get the total number of tests
    total_tests = 0

    for count in counts: #for each count
        for key,value in count.items(): #extract the key and value
            inputs = str(key[2])+str(key[3])
            output = int(key[4])
            statistics[inputs][output] = statistics[inputs][output] + value
            total_tests = total_tests + value

    #compute the error
    error = statistics["00"][1] + statistics["01"][0] + statistics["10"][0] + statistics["11"][1]
    error = error / total_tests

    #return the error
    return error

def exaustive_grid_search(grid_grain=4,number_of_runs=1):
    """
    Perform an exaustive search of the parameter space to find the optimal parameters for the quantum neural network.

    Parameters:
    grid_grain (int): The number of points in the grid to search.
    number_of_runs (int): The number of times the circuit is run for each point in the grid.

    Returns:
    The optimal parameters (list of floats) and the total error (float) of the optimal parameters.
    """
    final_parameters = []
    final_error = 1
    
    for i in np.linspace(0, np.pi, grid_grain):
        for j in np.linspace(0, np.pi, grid_grain):
            for k in np.linspace(0, np.pi, grid_grain):
                for l in np.linspace(0, np.pi, grid_grain):
                    
                    counts = qNN_circuit(all_inputs_circuit(), [i, j, k, l]).evaluate(number_of_runs=number_of_runs)
                    current_error = error(counts)

                    if current_error < final_error:
                        final_error = current_error
                        final_parameters = [i, j, k, l]

    return final_parameters, final_error

"""final_parameters, final_error = exaustive_grid_search(grid_grain=10)
print(final_error)
print(final_parameters)
all_inputs_circuit = all_inputs_circuit()
#final_parameters = [1.3963, 1.3963, 1.7453, 1.7453]
qNN_circuit = qNN_circuit(all_inputs_circuit, final_parameters)
qNN_circuit.print_circuit()
counts = evaluate_quantum_circuit(qNN_circuit.get_current_circuit(), number_of_runs=10)
error_counts = error(counts)
print(error_counts)"""
"""
  qNN-functions.py

Module that defines the functions to run the qNN with a single qubit.

Dependencies:
- Uses current_circuit.py module to define the quantum current quantum circuit
- Uses qiskit.circuit.library module to generate a circuit with all the possible inputs

Since:
- 01/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do my biblioteca necessary imports
from current_circuit import *

#do other necessary imports
import numpy as np #type: ignore

def generate_qNN_circuit(inputs,parameters):
    """
    Generate a quantum neural network circuit (qNN) with a single neuron.

    Parameters:
    input1_value (float): The value of the first input of the neuron.
    input2_value (float): The value of the second input of the neuron.
    parameters (list of floats): The parameters of the neuron, in order: first input weight, second input weight, bias.

    Returns:
    The qNN circuit (current_circuit).
    """
    qNN = current_circuit(1,1) #create the qNN circuit
    qNN.add_neuron(*inputs, *parameters, 0, 0) #add the neuron
    qNN.get_current_circuit().measure_all()

    return qNN

def compute_error(counts,expected_output):
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

    #compute error for each count
    for count in counts:
        if expected_output in count:
            error = number_of_shots - count[expected_output]
        else:
            error = number_of_shots

    #normalize error
    error = error/number_of_shots

    #return error
    return error

def compute_total_error(inputs,expected_outputs,parameters,number_of_runs=1):
    """
    Compute the total error for a set of inputs and expected outputs.

    Parameters:
    inputs (list of lists): A list containing pairs of input values for the neuron.
    expected_outputs (list of floats): A list of expected output values for each input pair.
    parameters (list of floats): The parameters of the neuron, including weights and bias.
    number_of_runs (int): The number of times the circuit is run.

    Returns:
    The total error (float) across all input pairs.
    """

    #initialize total error
    total_error = 0

    #apply qNN circuit to each input
    for interation in range(len(inputs)):

        qNN_circuit = generate_qNN_circuit(inputs[interation],parameters) #generate circuit
        counts = qNN_circuit.evaluate(number_of_runs=number_of_runs) #run circuit
        total_error += compute_error(counts,expected_outputs[interation]) #add error

    #normalize total error
    total_error = total_error/len(inputs)

    #return total error
    return total_error

def exaustive_search(inputs,expected_outputs,grid_grain=5,number_of_runs=1):
    """
    Perform an exaustive search of the parameter space to find the optimal parameters for the given inputs and expected outputs.

    Parameters:
    inputs (list of lists): A list containing pairs of input values for the neuron.
    expected_outputs (list of floats): A list of expected output values for each input pair.
    grid_grain (int): The number of points in the grid to search.
    number_of_runs (int): The number of times the circuit is run.

    Returns:
    The optimal parameters (list of floats) and the total error (float) of the optimal parameters.
    """

    #initialize final error
    final_error = 1

    #initialize final parameters
    final_parameters = [0,0,0]

    #exaustive search
    for weight1 in np.linspace(-np.pi, np.pi, grid_grain):
        for weight2 in np.linspace(-np.pi, np.pi, grid_grain):
            for weight3 in np.linspace(-np.pi, np.pi, grid_grain):

                #compute total error
                parameters = [weight1, weight2, weight3]
                current_error = compute_total_error(inputs,expected_outputs,parameters,number_of_runs=number_of_runs)

                #update final error
                if current_error < final_error:
                    final_error = current_error
                    final_parameters = parameters

    #return final parameters
    return final_parameters, final_error
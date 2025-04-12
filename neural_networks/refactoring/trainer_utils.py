"""
  trainer_utils.py

Module that defines the util functions to train the quantum neural network.

Dependencies:
- Uses qiskit.circuit.library module to define TwoLocal circuit
- Uses typing module to define List
- Uses functools.reduce module to reduce the logic gates
- Uses math module to define pi
- Uses operator module to define the logic gates
- Uses random module to get random numbers
- Uses numpy module to define pi
- Uses current_circuit module to define the neuron quantum circuit

Since:
- 04/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""
#do necessary imports
from typing import List
from functools import reduce
import operator
import random
import numpy as np # type: ignore


#do my biblioteca necessary imports
from amplitude_encondig_utils import *
from phase_enconding_utils import *

def compute_expected_outputs(inputs: List[List[int]], logic_gate: str = "XOR") -> List[str]:
    """
    Compute the expected outputs for a given list of inputs and a specified logic gate.

    Parameters:
    inputs (List[List[int]]): A list of lists containing the input values for each set of bits.
    logic_gate (str): The logic gate to compute the expected outputs for. Default is "XOR". 

    Returns:
    List[str]: A list of strings representing the expected outputs for each set of bits.
    """

    #define the operator to compute the expected outputs
    if logic_gate == "XOR":
        op = operator.xor
    elif logic_gate == "AND":
        op = operator.and_
    elif logic_gate == "OR":
        op = operator.or_
    elif logic_gate == "NAND":
        op = operator.and_
    elif logic_gate == "NOR":
        op = operator.or_
    elif logic_gate == "XNOR":
        op = operator.xor
    else:
        raise ValueError("Logic gate must be 'XOR', 'AND' or 'OR'")
    
    #compute the expected outputs
    result_dict = {}
    for row in inputs:#for each input

        input_key = ''.join(map(str, row))
        result = reduce(op, row)
        result_dict[input_key] = result

    #return the expected outputs
    return result_dict

    #return [str(reduce(op, row)) for row in inputs]

def phase_qNN_evaluate(inputs,expected_outputs,parameters,number_of_runs=1,number_of_shots=1024,number_of_inputs=2,type_of_run="simulation",number_of_inputs_per_qubit=3):
    """
    Compute the total error for a set of inputs and expected outputs.

    Parameters:
    inputs (list of lists): A list containing pairs of input values for the neuron.
    expected_outputs (list of floats): A list of expected output values for each input pair.
    parameters (list of floats): The parameters of the neuron, including weights and bias.
    number_of_runs (int): The number of times the circuit is run.
    number_of_shots (int): The number of shots to run the circuit.
    number_of_inputs (int): The number of qbits in the circuit.
    type_of_run (str): The type of run to use.
    number_of_inputs_per_qubit (int): The number of inputs per qubit.

    Returns:
    The total error (float) across all input pairs.
    """

    #initialize total error
    total_error = 0

    #define list of expected outputs
    list_of_expected_outputs = list(expected_outputs.values())

    #apply qNN circuit to each input
    for interation in range(len(inputs)):

        qNN_circuit = generate_phase_qNN_circuit(inputs[interation],parameters,number_of_inputs,number_of_inputs_per_qubit=number_of_inputs_per_qubit) #generate circuit
        counts = qNN_circuit.evaluate(number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run) #run circuit
        total_error += phase_qNN_compute_error(counts,list_of_expected_outputs[interation]) #add error

    #normalize total error
    total_error = total_error/len(inputs)

    #return total error
    return total_error

def amplitude_qNN_evaluate(inputs,expected_outputs,parameters,number_of_runs=1,number_of_shots=1024,number_of_inputs=2,type_of_run="simulation"):
    """
    Generate amplitude qNN circuit and compute error
    """
    #compute counts
    counts = generate_amplitude_qNN_circuit(parameters, number_of_inputs=number_of_inputs).evaluate(number_of_runs=number_of_runs,number_of_shots=number_of_shots,type_of_run=type_of_run)

    return amplitude_qNN_compute_error(inputs, expected_outputs, counts, number_of_inputs=number_of_inputs)
    

def phase_qNN_compute_gradient(parameters, inputs, expected_outputs, number_of_inputs, number_of_runs, number_of_shots, type_of_run, epsilon=1e-3, number_of_inputs_per_qubit=3):
    """
    Compute the gradient of the error function using finite differences for phase encoding case.
    """

    gradient = np.zeros(len(parameters))

    for i in range(len(parameters)):
        perturbed_params = parameters.copy()

        # Compute f(x + epsilon)
        perturbed_params[i] += epsilon
        error_plus = phase_qNN_evaluate(
            inputs, expected_outputs, perturbed_params, number_of_runs=number_of_runs, 
            number_of_shots=number_of_shots, number_of_inputs=number_of_inputs, 
            type_of_run=type_of_run, number_of_inputs_per_qubit=number_of_inputs_per_qubit
        )

        # Compute f(x - epsilon)
        perturbed_params[i] -= 2 * epsilon
        error_minus = phase_qNN_evaluate(
            inputs, expected_outputs, perturbed_params, number_of_runs=number_of_runs, 
            number_of_shots=number_of_shots, number_of_inputs=number_of_inputs, 
            type_of_run=type_of_run, number_of_inputs_per_qubit=number_of_inputs_per_qubit
        )

        # Compute numerical gradient
        gradient[i] = (error_plus - error_minus) / (2 * epsilon)

    return gradient

def amplitude_qNN_compute_gradient(parameters, inputs, expected_outputs, number_of_inputs, number_of_runs, number_of_shots, type_of_run, epsilon=1e-3, save_history=False):
    """
    Compute the gradient of the error function using finite differences for amplitude encoding case.

    Parameters:
    parameters (list of floats): Current parameters of the circuit.
    inputs (list of lists): Input data.
    expected_outputs (list of floats): Expected outputs.
    number_of_inputs (int): Number of qubits.
    number_of_runs (int): Number of times the circuit is run.
    number_of_shots (int): Number of shots.
    type_of_run (str): Type of quantum run.
    epsilon (float): Small perturbation for numerical gradient computation.

    Returns:
    list of floats: Gradient vector.
    """

    gradient = np.zeros(len(parameters))

    for i in range(len(parameters)):
        perturbed_params = parameters.copy()

        # Compute f(x + epsilon)
        perturbed_params[i] += epsilon
        counts_plus = generate_amplitude_qNN_circuit(perturbed_params, number_of_inputs=number_of_inputs).evaluate(
            number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run
        )
        error_plus = amplitude_qNN_compute_error(inputs, expected_outputs, counts_plus, number_of_inputs=number_of_inputs)

        # Compute f(x - epsilon)
        perturbed_params[i] -= 2 * epsilon
        counts_minus = generate_amplitude_qNN_circuit(perturbed_params, number_of_inputs=number_of_inputs).evaluate(
            number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run
        )
        error_minus = amplitude_qNN_compute_error(inputs, expected_outputs, counts_minus, number_of_inputs=number_of_inputs)

        # Compute numerical gradient
        gradient[i] = (error_plus - error_minus) / (2 * epsilon)

    return gradient

def random_parameters(tipe_of_enconding=None, number_of_inputs=2):
    """
    Generate random parameters for a quantum neural network.

    Parameters:
    tipe_of_enconding (str): Type of enconding, either "phase" or "amplitude".
    number_of_inputs (int): Number of inputs.

    Returns:
    list of floats: Random parameters.
    """
    #define size of list of random parameters based on enconding and number of inputs
    if tipe_of_enconding == "phase":
        size = number_of_inputs + 1
    elif tipe_of_enconding == "amplitude":
        size = number_of_inputs * 2
    else:
        raise ValueError("Invalid type of enconding.")
        
    return np.random.uniform(-np.pi, np.pi, size=size)
    

def update_if_better(parameters, current_error, final_parameters, final_error):
    """
    Update the final parameters and error if the current parameters result in a better error.

    Parameters:
    parameters (list of floats): The current parameters.
    current_error (float): The current error.
    final_parameters (list of floats): The final parameters.
    final_error (float): The final error.

    Returns:
    tuple: The final parameters and error.
    """
    if current_error < final_error:
        return parameters, current_error
    return final_parameters, final_error

def select_parents(population, errors):
    """
    Select two parents from the population based on their fitness.

    Parameters:
    population (list): List of individuals in the population.
    errors (list): List of errors of the individuals in the population.

    Returns:
    tuple: Two parents selected from the population.
    """

    #compute each fitness
    fitness = [1 / (1 + e) for e in errors]

    #sum all fiteness
    total = sum(fitness)

    #compute probabilities
    probabilities = [f / total for f in fitness]

    return random.choices(population, probabilities, k=2)

def crossover(p1, p2):
    """
    Perform a crossover between two parents to generate two offspring.

    The crossover is done by selecting a random point in the parents and swapping the parameters after that point.

    Parameters:
    p1 (list): First parent.
    p2 (list): Second parent.

    Returns:
    tuple: Two offspring generated by crossover.
    """
    #select random point
    point = random.randint(1, len(p1) - 1)
    
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def mutate(individual, mutation_rate):
    """
    Mutate an individual with a given mutation rate.

    The mutation is done by adding a random value between -pi/10 and pi/10 to each parameter of the individual with a given probability.

    Parameters:
    individual (list): List of parameters of the individual.
    mutation_rate (float): Probability of mutation for each parameter.

    Returns:
    list: List of parameters of the mutated individual.
    """
    return [x + random.uniform(-np.pi / 10, np.pi / 10) if random.random() < mutation_rate else x for x in individual]

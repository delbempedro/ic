"""
  qNN.py

Module that defines the quantum current quantum circuit.

Dependencies:
- Uses qiskit.circuit.library module to generate a circuit with all the possible inputs
- Uses qiskit.primitives module to run the circuit

Since:
- 02/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do qiskit necessary imports
from qiskit.circuit.library import TwoLocal #type: ignore

#do my biblioteca necessary imports
from current_circuit import *

#do other necessary imports
import numpy as np #type: ignore
from math import pi
import itertools
from typing import List
from functools import reduce
import operator

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

def generate_single_qubit_qNN_circuit(inputs,parameters,number_of_bits,number_of_inputs_per_qubit=3):
    """
    Generate a quantum neural network circuit (qNN) with a single neuron.

    Parameters:
    input1_value (float): The value of the first input of the neuron.
    input2_value (float): The value of the second input of the neuron.
    parameters (list of floats): The parameters of the neuron, in order: first input weight, second input weight, bias.
    number_of_bits (int): The number of qbits in the circuit.
    number_of_inputs_per_qubit (int): The number of inputs per qubit.

    Returns:
    The qNN circuit (current_circuit).
    """

    #create the qNN circuit
    if number_of_bits <= number_of_inputs_per_qubit:
        qNN = current_circuit(1,1)
    elif number_of_bits%2: #if the number of qubits is even
        qNN = current_circuit(number_of_bits//number_of_inputs_per_qubit+2,1)
    else: #if the number of qubits is odd
        qNN = current_circuit(number_of_bits//number_of_inputs_per_qubit+1,1)

    qNN.add_single_qubit_neuron(inputs, parameters, number_of_bits=number_of_bits, number_of_inputs_per_qubit=number_of_inputs_per_qubit) #add the neuron
    qNN.get_current_circuit().measure_all() #measure all qubits

    #return the circuit
    return qNN

#Generates a circuit with all the possible inputs
def all_inputs_circuit(number_of_bits):
    """
    Generates a quantum circuit that produces all the possible inputs for a given number of qubits.

    The circuit is constructed using the TwoLocal method. The parameters are set to predefined values 
    to generate all possible inputs.

    Parameters:
    num_qubits (int): The number of qubits in the circuit.

    Returns:
    QuantumCircuit: The quantum circuit that generates all the possible inputs.
    """

    entanglement_circuit = TwoLocal(number_of_bits, "rx", "cz", entanglement="linear", reps=1) #generates a circuit that mixture two qubits - an entanglement circuit
    parameters = [pi] * number_of_bits + [pi / 2] * (number_of_bits) #previously chosen/learned parameters to generate all the possible inputs
    parameter_dict = dict(zip(entanglement_circuit.parameters, parameters)) #assigns the parameters to the circuit
    initial_states_circuit = entanglement_circuit.assign_parameters(parameter_dict) #a circuit that can generate all the possible inputs

    #return the circuit
    return initial_states_circuit

def circuit_copy(initial_circuit, number_of_bits):
    """
    Creates a quantum circuit that duplicates the given initial circuit with additional qubits.

    This function constructs a new quantum circuit by duplicating the specified number of qubits 
    from the initial circuit and adding two additional qubits to the circuit. The resulting circuit 
    reproduces the input values using controlled-NOT operations to copy the state of the first two 
    qubits to the additional qubits.

    Parameters:
    initial_circuit (QuantumCircuit): The initial quantum circuit to be copied.
    number_of_bits (int): The number of qubits to duplicate from the initial circuit.

    Returns:
    QuantumCircuit: A new quantum circuit with duplicated qubits and additional operations to copy the input.
    """

    circuit_copy = QuantumCircuit(number_of_bits*2) #duplicate the number of qubits
    circuit_copy = circuit_copy.compose(initial_circuit, qubits=list(range(0,number_of_bits))) #first half with the initial_states_2b_circ
    circuit_copy.barrier() #to visually separate circuit components

    for index in range(number_of_bits):
        circuit_copy.x(number_of_bits+index)

    for index in range(number_of_bits):
        circuit_copy.cx(index,number_of_bits+index)

    for index in range(number_of_bits):
        circuit_copy.x(number_of_bits+index)

    """#a layer to copy/reproduce the generated inputs in qubits 0 and 1
    circuit_copy.x(2)    #change value of qubit 2 from 0 to 1
    circuit_copy.x(3)    #change value of qubit 3 from 0 to 1
    circuit_copy.cx(0,2) #qb0 ''AND'' 1 (or NOT qb0) to copy qubit 0 to qubit 2
    circuit_copy.cx(1,3) #qb1 ''AND'' 1 (or NOT qb1) to copy qubit 1 to qubit 3
    circuit_copy.x(2)    #NOT of qubit 2 => qubit 2 equal to equal qubit 0
    circuit_copy.x(3)    #NOT of qubit 3 => qubit 3 equal to equal qubit 1"""

    return circuit_copy

def generate_multi_qubit_qNN_circuit(parameters_of_entanglement_circuit,number_of_bits=2):
    """
    Generates a quantum circuit that produces all the possible inputs and a quantum neural network with two neurons.

    The circuit is constructed by using the TwoLocal method. The parameters are set to the previously chosen/learned parameters to generate all the possible inputs.
    The TwoLocal method generates a circuit that mixture two qubits - an entanglement circuit.
    A quantum neural network with two neurons is added to the circuit, by using the add_bin_neuron3 method of the current_circuit class.
    The parameters of the quantum neural network are set to the previously chosen/learned parameters.

    Parameters:
    parameters_of_entanglement_circuit (list): A list of parameters for the U and controlled-phase (cp) gates.
    number_of_bits (int): The number of qbits in the circuit.

    Returns:
    quantum_circuit (QuantumCircuit): The quantum circuit with all the possible inputs and a quantum neural network with two neurons.
    """

    number_of_qubits_required = number_of_bits*2+1
    qNN = current_circuit(number_of_qubits_required,1) #create the qNN circuit
    auxiliary_circuit = all_inputs_circuit(number_of_bits) #copy the all inputs circuit
    duplicate_circuit = circuit_copy(auxiliary_circuit, number_of_bits) #duplicate the all inputs circuit
    qNN.get_current_circuit().append(duplicate_circuit, list(range(0,number_of_bits*2))) #add the all inputs circuit
    qNN.add_multi_qubit_neuron(parameters_of_entanglement_circuit, number_of_bits=number_of_bits) #add the neuron
    qNN.get_current_circuit().measure_all()

    return qNN

def single_qubit_compute_error(counts,expected_output):
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

def single_qubit_compute_total_error(inputs,expected_outputs,parameters,number_of_runs=1,number_of_shots=1024,number_of_bits=2,type_of_run="simulation",number_of_inputs_per_qubit=3):
    """
    Compute the total error for a set of inputs and expected outputs.

    Parameters:
    inputs (list of lists): A list containing pairs of input values for the neuron.
    expected_outputs (list of floats): A list of expected output values for each input pair.
    parameters (list of floats): The parameters of the neuron, including weights and bias.
    number_of_runs (int): The number of times the circuit is run.
    number_of_shots (int): The number of shots to run the circuit.
    number_of_bits (int): The number of qbits in the circuit.
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

        qNN_circuit = generate_single_qubit_qNN_circuit(inputs[interation],parameters,number_of_bits,number_of_inputs_per_qubit=number_of_inputs_per_qubit) #generate circuit
        counts = qNN_circuit.evaluate(number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run) #run circuit
        total_error += single_qubit_compute_error(counts,list_of_expected_outputs[interation]) #add error

    #normalize total error
    total_error = total_error/len(inputs)

    #return total error
    return total_error

def multi_qubit_compute_error(inputs,expected_outputs,counts,number_of_bits=2):
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
    number_of_bits (int): The number of qbits in the quantum circuit.

    Returns:
    float: The error of the quantum circuit.
    """

    #define the statistics dictionary
    statistics = {}
    for i in range(2**number_of_bits):
        binary_key = format(i, f'0{number_of_bits}b')
        statistics[binary_key] = [0, 0]
    
    #get the total number of tests
    total_tests = 0

    # Processa as contagens
    for count in counts: #for each count
        for key, value in count.items(): #for each key and value

            #define the inputs and the output
            inputs = ''
            for bit in range(number_of_bits, number_of_bits*2):
                inputs += str(key[bit])
            output = int(key[number_of_bits*2])

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

def single_qubit_qNN_exaustive_search(inputs,expected_outputs,grid_grain=5,number_of_runs=1,number_of_shots=1024,number_of_bits=2,type_of_run="simulation",number_of_inputs_per_qubit=3):
    """
    Perform an exaustive search of the parameter space to find the optimal parameters for the given inputs and expected outputs.

    Parameters:
    inputs (list of lists): A list containing pairs of input values for the neuron.
    expected_outputs (list of floats): A list of expected output values for each input pair.
    grid_grain (int): The number of points in the grid to search.
    number_of_runs (int): The number of times the circuit is run.
    number_of_shots (int): The number of shots to run the circuit.
    number_of_bits (int): The number of qbits in the circuit.
    type_of_run (str): The type of run to use.
    number_of_inputs_per_qubit (int): The number of inputs per qubit.

    Returns:
    The optimal parameters (list of floats) and the total error (float) of the optimal parameters.
    """

    #initialize final error
    final_error = 1

    #initialize final parameters
    final_parameters = [0]*(number_of_bits+1)

    #initialize grid
    grid = np.linspace(-np.pi, np.pi, grid_grain)

    #exaustive search
    for parameters in itertools.product(grid, repeat=(number_of_bits+1)):
        
        #compute total error
        current_error = single_qubit_compute_total_error(inputs, expected_outputs, parameters, number_of_runs=number_of_runs, number_of_shots=number_of_shots, number_of_bits=number_of_bits, type_of_run=type_of_run, number_of_inputs_per_qubit=number_of_inputs_per_qubit)

        #update final error and final parameters
        if current_error < final_error:
            final_error = current_error
            final_parameters = list(parameters)

        if current_error < 1/4:
            break

    #return final parameters
    return final_parameters, final_error

def multi_qubit_qNN_exaustive_search(inputs,expected_outputs,grid_grain=4,number_of_runs=1,number_of_shots=1024,number_of_bits=2,type_of_run="simulation"):
    """
    Perform an exaustive search of the parameter space to find the optimal parameters for the quantum neural network.

    Parameters:
    inputs (list of lists): A list containing pairs of input values for the neuron.
    expected_outputs (list of floats): A list of expected output values for each input pair.
    grid_grain (int): The number of points in the grid to search.
    number_of_runs (int): The number of times the circuit is run for each point in the grid.
    number_of_shots (int): The number of shots to run the circuit.
    number_of_bits (int): The number of qbits in the circuit.
    type_of_run (str): The type of run to be used in the circuit.

    Returns:
    The optimal parameters (list of floats) and the total error (float) of the optimal parameters.
    """

    #initialize final error
    final_error = 1

    #initialize final parameters
    final_parameters = [0]*number_of_bits

    #initialize grid
    grid = np.linspace(-np.pi, np.pi, grid_grain)

    #exaustive search
    for parameters in itertools.product(grid, repeat=number_of_bits*2):
                    
        counts = generate_multi_qubit_qNN_circuit(parameters,number_of_bits=number_of_bits).evaluate(number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run)
        current_error = multi_qubit_compute_error(inputs,expected_outputs,counts,number_of_bits=number_of_bits)

        if current_error < final_error:
            final_error = current_error
            final_parameters = list(parameters)

    return final_parameters, final_error

def multi_qubit_qNN_gradient_descent(inputs, expected_outputs, 
                                     learning_rate=0.1, max_iterations=1000, 
                                     number_of_runs=1, number_of_shots=1024, 
                                     number_of_bits=2, type_of_run="simulation",
                                     tolerance=1e-6):
    """
    Perform gradient descent optimization to find the optimal parameters for the quantum neural network.

    Parameters:
    inputs (list of lists): A list containing pairs of input values for the neuron.
    expected_outputs (list of floats): A list of expected output values for each input pair.
    learning_rate (float): The learning rate for gradient descent.
    max_iterations (int): The maximum number of iterations for the gradient descent.
    number_of_runs (int): The number of times the circuit is run for each set of parameters.
    number_of_shots (int): The number of shots to run the circuit.
    number_of_bits (int): The number of qubits in the circuit.
    type_of_run (str): The type of run to be used in the circuit.
    tolerance (float): The stopping criterion for convergence.

    Returns:
    The optimal parameters (list of floats) and the total error (float) of the optimal parameters.
    """

    # Inicializa parâmetros aleatórios dentro do intervalo [-π, π]
    parameters = np.random.uniform(-np.pi, np.pi, number_of_bits * 2)
    
    # Função para calcular o gradiente numericamente
    def compute_gradient(params, epsilon=1e-4):
        gradient = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            counts_plus = generate_multi_qubit_qNN_circuit(params_plus, number_of_bits=number_of_bits).evaluate(
                number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run)
            counts_minus = generate_multi_qubit_qNN_circuit(params_minus, number_of_bits=number_of_bits).evaluate(
                number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run)

            error_plus = multi_qubit_compute_error(inputs, expected_outputs, counts_plus, number_of_bits=number_of_bits)
            error_minus = multi_qubit_compute_error(inputs, expected_outputs, counts_minus, number_of_bits=number_of_bits)

            gradient[i] = (error_plus - error_minus) / (2 * epsilon)

        return gradient

    #Initializes final number of interations
    final_number_of_iterations = max_iterations

    # Loop do Gradient Descent
    for iteration in range(max_iterations):
        counts = generate_multi_qubit_qNN_circuit(parameters, number_of_bits=number_of_bits).evaluate(
            number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run)
        
        current_error = multi_qubit_compute_error(inputs, expected_outputs, counts, number_of_bits=number_of_bits)

        gradient = compute_gradient(parameters)

        # Atualiza os parâmetros
        parameters -= learning_rate * gradient

        # Verifica critério de convergência
        if np.linalg.norm(gradient) < tolerance:
            #print(f"Converged in {iteration} iterations.")
            final_number_of_iterations = iteration
            break

    return list(parameters), current_error, final_number_of_iterations

def compute_gradient(parameters, inputs, expected_outputs, number_of_bits, number_of_runs, number_of_shots, type_of_run, epsilon=1e-3):
    """
    Compute the gradient of the error function using finite differences.

    Parameters:
    parameters (list of floats): Current parameters of the circuit.
    inputs (list of lists): Input data.
    expected_outputs (list of floats): Expected outputs.
    number_of_bits (int): Number of qubits.
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
        counts_plus = generate_multi_qubit_qNN_circuit(perturbed_params, number_of_bits=number_of_bits).evaluate(
            number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run
        )
        error_plus = multi_qubit_compute_error(inputs, expected_outputs, counts_plus, number_of_bits=number_of_bits)

        # Compute f(x - epsilon)
        perturbed_params[i] -= 2 * epsilon
        counts_minus = generate_multi_qubit_qNN_circuit(perturbed_params, number_of_bits=number_of_bits).evaluate(
            number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run
        )
        error_minus = multi_qubit_compute_error(inputs, expected_outputs, counts_minus, number_of_bits=number_of_bits)

        # Compute numerical gradient
        gradient[i] = (error_plus - error_minus) / (2 * epsilon)

    return gradient


def multi_qubit_qNN_gradient_descent2(inputs, expected_outputs, number_of_bits=2, number_of_runs=1, number_of_shots=1024, 
                                     type_of_run="simulation", learning_rate=0.1, max_iterations=1000, tolerance=0.25):
    """
    Optimize the quantum neural network parameters using gradient descent.

    Parameters:
    inputs (list of lists): Input data.
    expected_outputs (list of floats): Expected outputs.
    number_of_bits (int): Number of qubits.
    number_of_runs (int): Number of times the circuit is run.
    number_of_shots (int): Number of shots.
    type_of_run (str): Type of quantum run.
    learning_rate (float): Step size for gradient descent.
    max_iterations (int): Maximum number of iterations.
    tolerance (float): Stopping criterion for small error changes.

    Returns:
    list of floats: Optimal parameters.
    float: Final error value.
    int: Number of iterations.
    """

    # Initialize parameters randomly within [-pi, pi]
    parameters = np.random.uniform(-np.pi, np.pi, size=number_of_bits * 2)
    
    # Compute initial error
    counts = generate_multi_qubit_qNN_circuit(parameters, number_of_bits=number_of_bits).evaluate(
        number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run
    )
    final_error = multi_qubit_compute_error(inputs, expected_outputs, counts, number_of_bits=number_of_bits)

    #Initializes final number of interations
    final_number_of_iterations = max_iterations

    for iteration in range(max_iterations):
        # Compute gradient
        gradient = compute_gradient(parameters, inputs, expected_outputs, number_of_bits, number_of_runs, number_of_shots, type_of_run)

        # Update parameters
        parameters -= learning_rate * gradient

        # Compute new error
        counts = generate_multi_qubit_qNN_circuit(parameters, number_of_bits=number_of_bits).evaluate(
            number_of_runs=number_of_runs, number_of_shots=number_of_shots, type_of_run=type_of_run
        )
        new_error = multi_qubit_compute_error(inputs, expected_outputs, counts, number_of_bits=number_of_bits)

        # Check for convergence
        if new_error < tolerance:
            #print(f"Converged after {iteration+1} iterations.")
            final_number_of_iterations = iteration
            break

        final_error = new_error

    return list(parameters), final_error, final_number_of_iterations

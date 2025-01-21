"""
  my-version-2-a-simply-qNN.py

Module that defines the quantum current quantum circuit.

Dependencies:
- Uses qiskit module to define a quantum circuit
- Uses qiskit_ibm_runtime module to defines service of the quantum circuit

Since:
- 11/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""
#Do qiskit necessary imports
from qiskit import QuantumCircuit #type: ignore
from qiskit.circuit.library import TwoLocal #type: ignore
from qiskit.primitives import StatevectorSampler #type: ignore
from qiskit.result import QuasiDistribution #type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService #type: ignore

#Do numpy necessary imports
import numpy as np #type: ignore

#Generates a circuit with all the possible inputs
def all_inputs_circuit():
    """
    Generates a quantum circuit that produces all the possible inputs.

    The circuit is constructed by using the TwoLocal method. The parameters are set to the previously chosen/learned parameters to generate all the possible inputs.

    Returns:
    quantum_circuit (QuantumCircuit): The quantum circuit that generates all the possible inputs.
    """

    two_qubits_entanglement_circuit = TwoLocal(2, "rx", "cz", entanglement="linear", reps=1) #generates a circuit that mixture two qubits - an entanglement circuit
    parameters = [np.pi, np.pi, np.pi/2, np.pi/2] #previously chosen/learned parameters to generate all the possible inputs
    parameter_dict = dict(zip(two_qubits_entanglement_circuit.parameters, parameters)) #assigns the parameters to the circuit
    initial_states_circuit = two_qubits_entanglement_circuit.assign_parameters(parameter_dict) #a circuit that can generate all the possible inputs

    #return the circuit
    return initial_states_circuit

def xor_circuit_with_1_parameter(all_inputs_circuit, parameters_of_entanglement_circuit, position_of_qubit0, position_of_qubit1):
    """
    Constructs a quantum XOR circuit using the provided input circuit and entanglement parameters.

    This function takes an existing quantum circuit that generates all possible inputs and applies
    a sequence of quantum gates to configure it as an XOR gate. The circuit is parameterized by the
    provided entanglement parameters which control the rotations and phase shifts on the specified qubits.

    Parameters:
    all_inputs_circuit (QuantumCircuit): The quantum circuit producing all possible inputs.
    parameters_of_entanglement_circuit (list): A list of parameters for the U and controlled-phase (cp) gates.
    position_of_qubit0 (int): The index of the first qubit to be used in the XOR operation.
    position_of_qubitb1 (int): The index of the second qubit to be used in the XOR operation.

    Returns:
    QuantumCircuit: The modified quantum circuit configured as an XOR gate.
    """

    qxor = all_inputs_circuit.copy() #copy the all inputs circuit
    qxor.barrier() #to visually separate circuit components
    qxor.cp(*parameters_of_entanglement_circuit, position_of_qubit1, position_of_qubit0)
  
    return qxor

def xor_circuit_with_3_parameters(all_inputs_circuit, parameters_of_entanglement_circuit, position_of_qubit0, position_of_qubit1):
    """
    Constructs a quantum XOR circuit using the provided input circuit and entanglement parameters.

    This function takes an existing quantum circuit that generates all possible inputs and applies
    a sequence of quantum gates to configure it as an XOR gate. The circuit is parameterized by the
    provided entanglement parameters which control the rotations and phase shifts on the specified qubits.

    Parameters:
    all_inputs_circuit (QuantumCircuit): The quantum circuit producing all possible inputs.
    parameters_of_entanglement_circuit (list): A list of parameters for the U and controlled-phase (cp) gates.
    position_of_qubit0 (int): The index of the first qubit to be used in the XOR operation.
    position_of_qubitb1 (int): The index of the second qubit to be used in the XOR operation.

    Returns:
    QuantumCircuit: The modified quantum circuit configured as an XOR gate.
    """

    qxor = all_inputs_circuit.copy() #copy the all inputs circuit
    qxor.barrier() #to visually separate circuit components
    qxor.cu(*parameters_of_entanglement_circuit, 0,  position_of_qubit1, position_of_qubit0)
  
    return qxor

def bin_neuron(qc,weight1,weight2,first_qbit_index):
    """
    Quantum circuit for a neuron.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    first_qbit_index (int): The first qubit of the three qubits to be used in the neuron.
    
    """

    qc.h(first_qbit_index)
    qc.h(first_qbit_index+1)

    qc.z(first_qbit_index+1)
    qc.cp(weight1,first_qbit_index,first_qbit_index+1)

    qc.z(first_qbit_index+1)
    qc.z(first_qbit_index)
    qc.cp(weight2,first_qbit_index,first_qbit_index+1)

    qc.h(first_qbit_index)
    qc.h(first_qbit_index+1)

    qc.x(first_qbit_index)
    qc.x(first_qbit_index+1)
    qc.ccx(first_qbit_index,first_qbit_index+1,first_qbit_index+2)

def qNN(all_inputs_circuit, parameters_of_entanglement_circuit, position_of_qubit0, position_of_qubit1):
    qxor = all_inputs_circuit.copy() #copy the all inputs circuit
    qxor.barrier() #to visually separate circuit components
    bin_neuron(qxor, parameters_of_entanglement_circuit[0], parameters_of_entanglement_circuit[1], 0)


def evaluate_quantum_circuit(quantum_circuit, number_of_shots, number_of_runs):
  """
  Evaluate a quantum circuit (XOR candidate) and return the counts (histogram of the outputs).

  Parameters:
  quantum_circuit (QuantumCircuit): The quantum circuit to be evaluated.
  number_of_shots (int): The number of shots to be used in the evaluation.
  number_of_runs (int): The number of times the quantum circuit is run.

  Returns:
  list: A list of dictionaries, where each dictionary represents the counts of the outputs of the quantum circuit.
  """

  #sample results with severals runs, each with several shots
  sampler = StatevectorSampler()
  #create jobs list
  jobs = []
  
  #run the circuit several times
  for arun in range(0,number_of_runs):
    #run the circuit
    job = sampler.run([(quantum_circuit)], shots = number_of_shots)
    #append the job to the jobs list
    jobs.append(job)

  #create the counts list
  counts = []

  #get and show raw results - counts
  for job in jobs:

    #get the data
    data_pub = job.result()[0].data
    #print("HI",job.result()[0].data.meas)
    job_counts = data_pub.meas.get_counts()

    #append the counts to the counts list
    counts.append(job_counts)

  #return the counts list
  return counts

#returns the squared error of the found density matrix compared with the expected density matrix
def sq_error(counts, expected_density_matrix): #counts - with the raw results after run the quantum circuit of a XOR candidate
  """
  Evaluate the squared error of the found density matrix compared with the expected density matrix.

  Parameters:
  counts (list): A list of dictionaries, where each dictionary represents the counts of the outputs of the quantum circuit.
  expected_density_matrix (numpy array): The expected density matrix.

  Returns:
  float: The squared error of the found density matrix compared with the expected density matrix.
  """
  statistics = dict() #defines the statistics dictionary
  for count in counts: #for each count
    for key,value in count.items(): #extract the key and value
      #update the statistics
      statistics = {key: statistics.get(key, 0) + value for key, value in count.items()}

  #get the total number of tests
  total_tests = sum([value for value in statistics.values()])

  #evaluate the frequencies of the outputs from the raw data collected
  frequencies = {state: count / total_tests for state, count in statistics.items()}

  #a density matrix of inputs from all the raw data collected (quantum circuit outputs)
  a_density_matrix = np.zeros((4, 2))

  #fill the density matrix
  for state, frequence in frequencies.items(): 

    #convert binary string to two integers
    inputs = int(state[0]), int(state[1])

    #update the density matrix
    a_density_matrix[inputs] = frequence

  #get the squared error
  #squared_error = np.sum(np.abs(a_density_matrix - expected_density_matrix))
  #squared_error = abs(a_density_matrix[0][1]-expected_density_matrix[0][1]) + abs(a_density_matrix[1][1]-expected_density_matrix[1][1]) + abs(a_density_matrix[2][1]-expected_density_matrix[2][1]) + abs(a_density_matrix[3][1]-expected_density_matrix[3][1])
  squared_error = 0
  for i in range(4):
    for j in range(2):
      squared_error += abs(a_density_matrix[i][j]-expected_density_matrix[i][j])
  
  #return the squared error
  return squared_error

#gets the expected density matrix
expected_density_matrix = np.zeros((4, 2))
expected_density_matrix[0][0] = 0.25
expected_density_matrix[1][1] = 0.25
expected_density_matrix[2][1] = 0.25
expected_density_matrix[3][0] = 0.25

def complete_function(objective_function, entanglement_circuit_paramameters):
  """
  Evaluate a quantum circuit (XOR candidate) and return the squared error compared with the expected density matrix.

  Parameters:
  objective_function (function): The function to generate the quantum circuit (XOR candidate).
  entanglement_circuit_paramameters (list): A list of parameters for the U and controlled-phase (cp) gates.

  Returns:
  float: The squared error of the found density matrix compared with the expected density matrix.
  """

  #generates a circuit that is a candidate to XOR
  qc_all_inputs = all_inputs_circuit() #generates a circuit that generates all the possible inputs
  candidate_to_xor = objective_function(qc_all_inputs, entanglement_circuit_paramameters, 0, 1) #generates a circuit that is a candidate to XOR
  candidate_to_xor.measure_all() #add measurements

  #evaluates the candidate
  counts = evaluate_quantum_circuit(candidate_to_xor, number_of_shots = 100, number_of_runs = 100)

  return sq_error(counts, expected_density_matrix)

def str_dictionary(dictionary):
  """
  Convert a dictionary to a string.

  Parameters:
  dictionary (dict): The dictionary to be converted.

  Returns:
  str: The string representation of the dictionary.
  """

  #gets the keys and values
  keys = list(dictionary.keys())
  values = list(dictionary.values())

  #initializes the string
  string = ""

  #converts the dictionary to a string
  for i in range(len(keys)):

    #gets the key and value
    key = str(keys[i])
    value = str(values[i])

    #appends the key and value
    if key != "error":
      string += key + ": " + value + ", "
    else:
      string += key + ": " + value

  #returns the string
  return string

def exaustive_grid_search(objective_function, grid_grain=5):
  """
  Perform an exhaustive grid search to find the parameters that minimize the squared error of the ``XOR`` candidate.

  Parameters:
  objective_function (function): The function to generate the quantum circuit (XOR candidate).
  grid_grain (int): The number of points in the grid search.

  Returns:
  list: The final coefficients.
  float: The final squared error.
  """
  
  #defines infos dictionary and errors list
  infos = dict()
  errors = []

  file_name = "analize-"+str(grid_grain)+".txt"

  if objective_function == xor_circuit_with_3_parameters:

    #computes the number of iterations
    iterations = 0

    #runs the grid search
    for phi in np.linspace(0, np.pi, grid_grain):
      for theta in np.linspace(0, np.pi, grid_grain):
        for _lambda in np.linspace(0, np.pi, grid_grain):

          #upgrade the iterations
          iterations += 1

          #defines the initial coefficients
          initial_coefficients = [theta, phi, _lambda]

          #evaluates the squared error
          squared_error = complete_function(objective_function, initial_coefficients)
          
          #updates infos dictionary and errors list
          infos[str(squared_error)] = {'theta': theta, 'phi': phi, 'lambda': _lambda, 'error': squared_error}
          errors.append(squared_error)

          #prints the result
          #print(initial_coefficients, str_dictionary(infos[str(squared_error)]), squared_error)
                  #save the weights and the error in the file
      if iterations == 1:
        with open(file_name, "w") as arquivo:
          content = 'theta: ' + str(theta) + ' phi: ' + str(phi) + ' lambda: ' + str(_lambda) + ' error: ' + str(squared_error) + '\n'
          arquivo.write(content)
      else:
        with open(file_name, "a") as arquivo:
          content = 'theta: ' + str(theta) + ' phi: ' + str(phi) + ' lambda: ' + str(_lambda) + ' error: ' + str(squared_error) + '\n'
          arquivo.write(content)

  elif objective_function == xor_circuit_with_1_parameter:

    #computes the number of iterations
    iterations = 0

    #runs the grid search
    for gamma in np.linspace(0, np.pi, grid_grain):

      #upgrade the iterations
      iterations += 1

      #defines the initial coefficients
      initial_coefficients = [gamma]

      #evaluates the squared error
      squared_error = complete_function(objective_function, initial_coefficients)
      
      #updates infos dictionary and errors list
      infos[str(squared_error)] = {'gamma': gamma, 'error': squared_error}
      errors.append(squared_error)

      #prints the result
      #print(initial_coefficients, str_dictionary(infos[str(squared_error)]), squared_error)
      if iterations == 1:
        with open(file_name, "w") as arquivo:
          content = 'gamma: ' + str(gamma) + ' error: ' + str(squared_error) + '\n'
          arquivo.write(content)
      else:
        with open(file_name, "a") as arquivo:
          content = 'gamma: ' + str(gamma) + ' error: ' + str(squared_error) + '\n'
          arquivo.write(content)

  elif objective_function == qNN:

    #computes the number of iterations
    iterations = 0

    #runs the grid search
    for gamma in np.linspace(0, np.pi, grid_grain):
      for phi in np.linspace(0, np.pi, grid_grain):

        #upgrade the iterations
        iterations += 1

        #defines the initial coefficients
        initial_coefficients = [gamma, phi]

        #evaluates the squared error
        squared_error = complete_function(objective_function, initial_coefficients)
        
        #updates infos dictionary and errors list
        infos[str(squared_error)] = {'gamma': gamma, 'phi': phi, 'error': squared_error}
        errors.append(squared_error)

        #prints the result
        #print(initial_coefficients, str_dictionary(infos[str(squared_error)]), squared_error)
        if iterations == 1:
          with open(file_name, "w") as arquivo:
            content = 'gamma: ' + str(gamma) + ' phi' + str(phi) + ' error: ' + str(squared_error) + '\n'
            #arquivo.write(content)
        else:
          with open(file_name, "a") as arquivo:
            content = 'gamma: ' + str(gamma) + ' phi' + str(phi) + ' error: ' + str(squared_error) + '\n'
            #arquivo.write(content)

  #gets the minimum squared error and the final coefficients
  minimum_error = min(errors)
  output = infos[str(minimum_error)]
  final_coefficients = list(output.values())[:-1]
  final_squared_error = minimum_error
    
  #returns the final coefficients and the final squared error
  return final_coefficients, final_squared_error

grid_grain = 10
file_name = "analize-"+str(grid_grain)+".txt"

print("MORE SIMPLE")
parameter, squared_error = exaustive_grid_search(xor_circuit_with_1_parameter, grid_grain)
with open(file_name, "a") as arquivo:
  content = 'gamma: ' + str(parameter[0]) + ' error: ' + str(squared_error)  + '\n'
  arquivo.write(content)
print("GAMMA: ", str(parameter[0]), "ERROR: ", squared_error,"\n")

print("MORE COMPLEX")
parameters, squared_error = exaustive_grid_search(xor_circuit_with_3_parameters, grid_grain)
print("THETA: ", str(parameters[0]), "PHI: ", str(parameters[1]), "LAMBDA: ", str(parameters[2]), "ERROR: ", squared_error,"\n")
with open(file_name, "a") as arquivo:
  content = 'theta: ' + str(parameters[0]) + ' phi: ' + str(parameters[1]) + ' lambda: ' + str(parameters[2]) + ' error: ' + str(squared_error) + '\n'
  arquivo.write(content)

"""print("qNN")
parameters, squared_error = exaustive_grid_search(qNN, grid_grain)
print("THETA: ", str(parameters[0]), "PHI: ", str(parameters[1]), "ERROR: ", squared_error,"\n")
with open(file_name, "a") as arquivo:
  content = 'theta: ' + str(parameters[0]) + ' phi: ' + str(parameters[1]) + ' error: ' + str(squared_error) + '\n'
  arquivo.write(content)"""
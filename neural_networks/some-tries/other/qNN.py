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
from current_circuit import *

#Do necesary other imports
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

    qNN = current_circuit(3,1) #create the qNN circuit
    auxiliary_circuit = all_inputs_circuit.copy() #copy the all inputs circuit
    auxiliary_circuit.measure_all() #add the measurement
    qNN.get_current_circuit().append(all_inputs_circuit, [0, 1]) #add the all inputs circuit
    qNN.add_bin_neuron3(0, 0, *parameters_of_entanglement_circuit, 0, 0) #add the neuron

    return qNN, auxiliary_circuit

def evaluate_quantum_circuit(quantum_circuit, number_of_shots = 1024, number_of_runs = 100):
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
  for _ in range(number_of_runs):

    #run the circuit
    job = sampler.run([(quantum_circuit)], shots = number_of_shots)
    #append the job to the jobs list
    jobs.append(job)

  #create the counts list
  counts = []

  #get and show raw results - counts
  for job in jobs:

    #get the data
    job_counts = job.result()[0].data.c.get_counts()

    #append the counts to the counts list
    counts.append(job_counts)

  #return the counts list
  return counts

def evaluate_auxiliary_quantum_circuit(quantum_circuit, number_of_shots=1024, number_of_runs=100):
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
    #print("HI",job.result()[0].data.values())
    job_counts = data_pub.meas.get_counts()
    #print(job_counts, type(job_counts))
    #append the counts to the counts list
    counts.append(job_counts)

  #return the counts list
  return counts

parameters = [pi, pi, pi/2, pi/2]
all_inputs_circuit = all_inputs_circuit()
qNN_circuit, auxiliary_circuit = qNN_circuit(all_inputs_circuit, parameters)
#qNN_circuit.print_circuit()
auxiliary_counts = evaluate_auxiliary_quantum_circuit(auxiliary_circuit)
counts = evaluate_quantum_circuit(qNN_circuit.get_current_circuit())
print(auxiliary_counts)#,counts
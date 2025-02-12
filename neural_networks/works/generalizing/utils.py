"""
  utils.py

Module that defines the util functions to create a quantum neuron.

Dependencies:
-

Since:
- 02/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

def single_qubit_neuron(qc,inputs,weights,number_of_bits=2,qbit_index=0):
    """
    Applies a quantum neuron operation to the given quantum circuit.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to which the quantum neuron operation is applied.
    inputs (list of floats): The values of the inputs to the neuron.
    weight1 (float): The weight of the first input of the neuron.
    weight2 (float): The weight of the second input of the neuron.
    weight3 (float): The bias of the neuron.
    qbit_index (int): The index of the qbit to which the quantum neuron operation is applied.
    """

    qc.h(qbit_index)

    last_gate = ""
    for index in range(number_of_bits):
        
        if last_gate == "rz":
            qc.rx(inputs[index]*weights[index]+weights[-1],qbit_index)
            last_gate = "rx"
        elif last_gate == "rx":
            qc.ry(inputs[index]*weights[index]+weights[-1],qbit_index)
            last_gate = "ry"
        else:
            qc.rz(inputs[index]*weights[index]+weights[-1],qbit_index)
            last_gate = "rz"

def multi_qubit_neuron(qc,parameters,number_of_bits=2,first_qbit_index=0):
    """
    Quantum circuit for a neuron.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    parameters (list of floats): The parameters of the neuron.
    number_of_bits (int): The number of qubits in the circuit.
    first_qbit_index (int): The first qubit of the three qubits to be used in the neuron.
    """
    
    for index in range(number_of_bits):
        qc.ry(parameters[index],first_qbit_index+index)

    for index in range(number_of_bits-1):
        qc.cx(first_qbit_index+index,first_qbit_index+index+1)

    for index in range(number_of_bits):
        qc.ry(parameters[index],first_qbit_index+index)

    for index in range(number_of_bits-1):
        qc.cx(first_qbit_index+index,first_qbit_index+index+1)

    qc.mcx(list(range(first_qbit_index,first_qbit_index+number_of_bits)),first_qbit_index+number_of_bits)

def generate_counts(quantum_circuit,sampler,number_of_runs=1,number_of_shots=1024):
    """
    Generate the counts of the quantum circuit.

    Parameters:
    quantum_circuit (QuantumCircuit): The quantum circuit to be executed.
    sampler (StatevectorSampler): The sampler to be used.
    number_of_runs (int): The number of times the quantum circuit is run.
    number_of_shots (int): The number of shots to be executed in each run.

    Returns:
    list: A list of dictionaries, where each dictionary represents the counts of the outputs of the quantum circuit.
    """

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
        data_pub = job.result()[0].data # 'pub' refers to Primitive Unified Bloc
        job_counts = data_pub.meas.get_counts()

        #append the counts to the counts list
        counts.append(job_counts)

    #return the counts list
    return counts
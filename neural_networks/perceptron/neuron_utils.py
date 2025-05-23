"""
  utils.py

Module that defines the util functions to create a quantum neuron.

Dependencies:
-

Since:
- 04/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

def phase_qubit_neuron(qc,inputs,weights,number_of_inputs=2,first_qubit_index=0,number_of_inputs_per_qubit=3):
    """
    Quantum circuit for a neuron with phase encoding.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to which the quantum neuron operation is applied.
    inputs (list of floats): The values of the inputs to the neuron.
    number_of_inputs (int): The number of qbits in the circuit.
    weights (list of floats): The weights of the inputs to the neuron.
    first_qubit_index (int): The index of the qbit to which the quantum neuron operation is applied.
    number_of_inputs_per_qubit (int): The number of inputs per qubit.

    """

    whole_part_of_division = number_of_inputs//number_of_inputs_per_qubit
    rest_of_division = number_of_inputs%number_of_inputs_per_qubit

    if number_of_inputs_per_qubit == 3:
        for index in range(whole_part_of_division):
            qc.h(first_qubit_index+index)
            qc.rz(weights[number_of_inputs_per_qubit*index]*inputs[number_of_inputs_per_qubit*index] + weights[-1], first_qubit_index+index)
            qc.ry(weights[number_of_inputs_per_qubit*index+1]*inputs[number_of_inputs_per_qubit*index+1] + weights[-1], first_qubit_index+index)
            qc.rx(weights[number_of_inputs_per_qubit*index+2]*inputs[number_of_inputs_per_qubit*index+2] + weights[-1], first_qubit_index+index)
    if number_of_inputs_per_qubit == 2:
        for index in range(whole_part_of_division):
            qc.h(first_qubit_index+index)
            qc.rz(weights[number_of_inputs_per_qubit*index]*inputs[number_of_inputs_per_qubit*index] + weights[-1], first_qubit_index+index)
            qc.ry(weights[number_of_inputs_per_qubit*index+1]*inputs[number_of_inputs_per_qubit*index+1] + weights[-1], first_qubit_index+index)

    if rest_of_division == 1:
        qc.h(first_qubit_index+whole_part_of_division)
        qc.rz(weights[-2]*inputs[-1] + weights[-1], first_qubit_index+whole_part_of_division)
    elif rest_of_division ==2:
        qc.h(first_qubit_index+whole_part_of_division)
        qc.rz(weights[-3]*inputs[-2] + weights[-1], first_qubit_index+whole_part_of_division)
        qc.ry(weights[-2]*inputs[-1] + weights[-1], first_qubit_index+whole_part_of_division)

    if number_of_inputs > number_of_inputs_per_qubit:

        if number_of_inputs%2:
            number_of_control_qubits=number_of_inputs//number_of_inputs_per_qubit+1
        else:
            number_of_control_qubits=number_of_inputs//number_of_inputs_per_qubit

        list_of_control_qubits = list(range(first_qubit_index,first_qubit_index+number_of_control_qubits))
        qc.mcx(list_of_control_qubits,first_qubit_index+number_of_control_qubits)

def amplitude_qubit_neuron(qc,parameters,number_of_inputs=2,first_qubit_index=0):
    """
    Quantum circuit for a neuron with amplitude encoding.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    parameters (list of floats): The parameters of the neuron.
    number_of_inputs (int): The number of qubits in the circuit.
    first_qubit_index (int): The first qubit of the three qubits to be used in the neuron.
    """
    
    for index in range(number_of_inputs):
        qc.ry(parameters[index],first_qubit_index+index)

    for index in range(number_of_inputs-1):
        qc.cx(first_qubit_index+index,first_qubit_index+index+1)

    for index in range(number_of_inputs):
        qc.ry(parameters[index],first_qubit_index+index)

    for index in range(number_of_inputs-1):
        qc.cx(first_qubit_index+index,first_qubit_index+index+1)

    qc.mcx(list(range(first_qubit_index,first_qubit_index+number_of_inputs)),first_qubit_index+number_of_inputs)

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
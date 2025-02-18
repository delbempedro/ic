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

def single_qubit_neuron(qc,inputs,weights,number_of_bits=2,first_qubit_index=0):
    """
    Applies a quantum neuron operation to the given quantum circuit.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to which the quantum neuron operation is applied.
    inputs (list of floats): The values of the inputs to the neuron.
    number_of_bits (int): The number of qbits in the circuit.
    weights (list of floats): The weights of the inputs to the neuron.
    first_qubit_index (int): The index of the qbit to which the quantum neuron operation is applied.
    """

    if number_of_bits <= 3:
        number_of_qubits_required = 1
    else:
        number_of_qubits_required = (number_of_bits//3)+1

    whole_part_of_division = number_of_bits//3
    rest_of_division = number_of_bits%3

    for index in range(whole_part_of_division):

        qc.h(first_qubit_index + index)
        last_gate = ""

        for sub_index in range(3):

            input_index = index * 3 + sub_index

            if last_gate == "rz":
                qc.rx(inputs[input_index] * weights[input_index] + weights[-1], first_qubit_index + index)
                last_gate = "rx"
            elif last_gate == "rx":
                qc.ry(inputs[input_index] * weights[input_index] + weights[-1], first_qubit_index + index)
                last_gate = "ry"
            else:
                qc.rz(inputs[input_index] * weights[input_index] + weights[-1], first_qubit_index + index)
                last_gate = "rz"

    last_gate = ""
    for index in range(rest_of_division):

        input_index = whole_part_of_division * 3 + index
        print(index, input_index, len(inputs), len(weights))
        print(inputs[input_index], weights[input_index], weights[-1])
        if last_gate == "rz":
            qc.rx(inputs[input_index] * weights[input_index] + weights[-1], first_qubit_index + whole_part_of_division + index)
            last_gate = "rx"
        elif last_gate == "rx":
            qc.ry(inputs[input_index] * weights[input_index] + weights[-1], first_qubit_index + whole_part_of_division + index)
            last_gate = "ry"
        else:
            qc.rz(inputs[input_index] * weights[input_index] + weights[-1], first_qubit_index + whole_part_of_division + index)
            last_gate = "rz"

    if number_of_qubits_required != 1:
        qc.mcx(list(range(first_qubit_index,first_qubit_index+whole_part_of_division)),first_qubit_index+number_of_bits)


def multi_qubit_neuron(qc,parameters,number_of_bits=2,first_qubit_index=0):
    """
    Quantum circuit for a neuron.
    
    Parameters:
    qc (QuantumCircuit): The quantum circuit to be modified.
    parameters (list of floats): The parameters of the neuron.
    number_of_bits (int): The number of qubits in the circuit.
    first_qubit_index (int): The first qubit of the three qubits to be used in the neuron.
    """
    
    for index in range(number_of_bits):
        qc.ry(parameters[index],first_qubit_index+index)

    for index in range(number_of_bits-1):
        qc.cx(first_qubit_index+index,first_qubit_index+index+1)

    for index in range(number_of_bits):
        qc.ry(parameters[index],first_qubit_index+index)

    for index in range(number_of_bits-1):
        qc.cx(first_qubit_index+index,first_qubit_index+index+1)

    qc.mcx(list(range(first_qubit_index,first_qubit_index+number_of_bits)),first_qubit_index+number_of_bits)

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
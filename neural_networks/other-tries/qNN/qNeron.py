"""
  qNeuron.py

Module that defines the quantum neuron class.

Dependencies:
- Uses typing to use Dict type.
- Uses qiskit to use QuantumCircuit, Aer and execute.
- Uses utils to use get_vector_from_int and create_hypergraph_state functions.

Since:
- 01/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#importing libraries
from typing import Dict
from qiskit import QuantumCircuit, Aer, execute #type: ignore
from utils import (create_hypergraph_state, get_vector_from_int)

#Neuron class
class Neuron:

    def __init__(self, number_of_qubits: int, weight: int = 1, input: int = 1):
        """
        Constructor method.

        Parameters:
        number_of_qubits (int): number of qubits of the circuit.
        weight (int): weight of the neuron. Default is 1.
        input (int): input of the neuron. Default is 1.
        """
        self.number_of_qubits = number_of_qubits
        self.weight = weight
        self.input = input
        self.control_flag = False
        self.create_circuit()

    def input_function(self):
        """
        Method to generate the input function circuit.

        The input function circuit is a quantum circuit composed by Hadamard gates
        and a hypergraph state corresponding to the input.

        The circuit is built as follows:

        1. Apply a Hadamard gate to each qubit of the circuit.
        2. Extract a vector from the given input.
        3. Apply the hypergraph state corresponding to the input vector.
        4. Convert the circuit to a gate.

        Parameters:
        None

        Returns:
        input_function (QuantumCircuit.to_gate): the input function circuit as a gate.
        """

        #checking if input_function() is called independently
        if not self.control_flag:
            raise RuntimeError("input_function() cannot be called independently.")

        input_function = QuantumCircuit(self.number_of_qubits)

        #applying hadamard to first number_of_qubits
        for q in range(self.number_of_qubits):
            input_function.h(q)

        #extracting vectors for input
        input_vector = get_vector_from_int(self.input, self.number_of_qubits)

        #applying hypergraph state corresponding to input.
        input_function = create_hypergraph_state(input_function,input_vector, self.number_of_qubits)

        #converting circuit to gate
        input_function = input_function.to_gate()

        return input_function

    def weight_function(self):
        """
        The weight function circuit is a quantum circuit composed by Hadamard gates, 
        a hypergraph state corresponding to the weight and X gates.

        The circuit is built as follows:

        1. Apply a Hadamard gate to each qubit of the circuit.
        2. Extract a vector from the given weight.
        3. Apply the hypergraph state corresponding to the weight vector.
        4. Apply an X gate to each qubit of the circuit.
        5. Convert the circuit to a gate.

        Parameters:
        None

        Returns:
        weight_function (QuantumCircuit.to_gate): the weight function circuit as a gate.
        """

        #checking if weight_function() is called independently
        if not self.control__flag:
            raise RuntimeError("weight_function() cannot be called independently.")

        weight_function = QuantumCircuit(self.number_of_qubits)

        #extracting vectors for weight
        input_vector = get_vector_from_int(self.weight, self.number_of_qubits)

        #applying hypergraph state corresponding to weight.
        weight_function = create_hypergraph_state(weight_function, input_vector, self.number_of_qubits)

        #applying hadamard to first number_of_qubits
        for q in range(self.number_of_qubits):
            weight_function.h(q)

        #applying X gate to first number_of_qubits
        for q in range(self.number_of_qubits):
            weight_function.x(q)

        #converting circuit to gate
        weight_function = weight_function.to_gate()

        #returning weight_function
        return weight_function

    def create_circuit(self):
        """
        Method to create the quantum circuit of the neuron.

        The circuit is composed by the input function, the weight function, 
        a Toffoli gate with the last qubit as target and a measurement of the last qubit.

        Parameters:
        None

        Returns:
        None
        """

        #creating circuit with number_of_qubits + 1(ancilla) qubit.
        self.circuit = QuantumCircuit(1 + self.number_of_qubits, 1)

        #defining switch_control_flag
        def switch_control_flag():
            self.control_flag = not self.control_flag

        #append input function for processing input
        switch_control_flag()
        self.circuit.append(self.input_function(), list(range(self.number_of_qubits)))
        switch_control_flag()

        #append weight function for processing input
        switch_control_flag()
        self.circuit.append(self.weight_function(), list(range(self.number_of_qubits)))
        switch_control_flag()

        #Toffoli gate at the end with target as ancilla qubit
        self.circuit.mcx(control_qubits=list(range(self.number_of_qubits)), target_qubit=self.number_of_qubits)

        #measure the last qubit.
        self.circuit.measure(self.number_of_qubits, 0)

    def measure_circuit(self, number_of_iters: int = 1000) -> Dict[str, int]:
        """
        Method to measure the quantum circuit of the neuron.

        Parameters:
        number_of_iters (int): number of iterations to measure the circuit.

        Returns:
        dict: a dictionary with keys as states and values as counts.
        """

        #checking if circuit is built
        if not hasattr(self, 'circuit'):
            raise RuntimeError("The circuit hasn't yet built.", "Please call build_circuit() first.")

        #setting backend
        backend = Aer.get_backend('qasm_simulator')

        #execute the circuit
        job = execute(self.circuit, backend, shots=number_of_iters)

        #get result and counts
        result = job.result()
        counts = result.get_counts(self.circuit)

        #returning counts
        return dict(counts)
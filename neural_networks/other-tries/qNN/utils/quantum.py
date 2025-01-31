"""
  quantum.py

Module that defines the utility functions.

Dependencies:
- Uses numpy to use array.
- Uses typing to use Dict and List types.
- Uses qiskit to use QuantumCircuit.
- Uses data to use get_possible_state_strings and get_amont_of_ones_to_states functions.

Since:
- 01/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

import numpy as np #type: ignore
from typing import List, Dict
from qiskit import QuantumCircuit #type: ignore
from data import (get_possible_state_strings, get_amont_of_ones_to_states)

def append_hypergraph_state(circuit: QuantumCircuit, data_vector: np.ndarray, states: np.ndarray, amont_of_ones: Dict[int, List[int]], number_of_qubits: int) -> QuantumCircuit:
    """
    Method to append a hypergraph state to a quantum circuit.

    Parameters:
    circuit (QuantumCircuit): the quantum circuit.
    data_vector (np.ndarray): the data vector.
    states (np.ndarray): all possible states.
    amont_of_ones (Dict[int, List[int]]): the amount of ones of the states.
    number_of_qubits (int): the number of qubits of the circuit.

    Returns:
    QuantumCircuit: the quantum circuit with the hypergraph state appended.
    """

    #create array to store signs
    is_sign_inverted = [1] * len(data_vector)

    #flipping all signs if all zero state has coef -1.
    if data_vector[0] == -1:
        for i in range(len(data_vector)):
            data_vector[i] *= -1

    #through all hypergraph states
    for ct in range(1, number_of_qubits + 1):

        #through all states
        for i in amont_of_ones.get(ct, []):

            #if hypergraph state is inverted
            if data_vector[i] == is_sign_inverted[i]:

                #apply hypergraph state
                state = states[i]
                ones_index = [j for j, x in enumerate(state) if x == '1']

                #if is the first hypergraph state
                if ct == 1:
                    circuit.z(ones_index[0])
                #if is the second hypergraph state
                elif ct == 2:
                    circuit.cz(ones_index[0], ones_index[1])
                #other cases
                else:
                    circuit.mcrz(-np.pi, [circuit.qubits[j] for j in ones_index[1:]], circuit.qubits[ones_index[0]])

                #invert signs
                for j, state in enumerate(states):
                    is_one = np.array([bit == '1' for bit in state])
                    if np.all(is_one[ones_index]):
                        is_sign_inverted[j] *= -1
    
    #return circuit
    return circuit


def create_hypergraph_state(circuit: QuantumCircuit, data_vector: np.ndarray, number_of_qubits: int) -> QuantumCircuit:
    """
    Method to create a hypergraph state quantum circuit.

    Parameters:
    circuit (QuantumCircuit): the quantum circuit to which the hypergraph state will be appended.
    data_vector (np.ndarray): the vector that represents the hypergraph state.
    number_of_qubits (int): the number of qubits of the circuit.

    Returns:
    QuantumCircuit: the resulting circuit with the hypergraph state appended.
    """
        
    #get all possible states strings and amont of ones
    states = get_possible_state_strings(number_of_qubits)
    amont_of_ones = get_amont_of_ones_to_states(states)

    #return resulting circuit
    return append_hypergraph_state(circuit, data_vector, states, amont_of_ones, number_of_qubits)
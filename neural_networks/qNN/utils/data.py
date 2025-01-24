"""
  data.py

Module that defines the utility functions.

Dependencies:
- Uses numpy to use array.
- Uses typing to use Dict and List types.
-

Since:
- 01/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

import numpy as np #type: ignore
from typing import Dict, List

def get_possible_state_strings(number_of_bits: int) -> np.ndarray:
    """
    Method to get all possible states as strings given a number of bits.

    Parameters:
    number_of_bits (int): number of bits to generate states.

    Returns:
    np.ndarray: array of strings containing all possible states.

    Raises:
    ValueError: if number of bits is negative, 0 or >= 10.
    """

    #validating number of bits
    if number_of_bits < 0:
        raise ValueError("Number of bits cannot be negative")
    if number_of_bits == 0:
        raise ValueError("Number of bits cannot be 0")
    if number_of_bits >= 10:
        raise ValueError("We are not currently supporting bits >= 10")

    #getting number of states by number of bits power of 2
    total_states = np.power(2, number_of_bits)

    #creating array to store states
    states = np.empty(total_states, dtype="<U10")

    #creating all possible states
    state_template = "{:0" + str(number_of_bits) + "b}"
    for i in range(total_states):
        states[i] = state_template.format(i)

    #returning states
    return states


def get_amont_of_ones_to_states(states: np.ndarray) -> Dict[int, List[int]]:
    """
    Method to get a dictionary of amount of ones to its respective states.

    Parameters:
    states (np.ndarray): array of strings containing all possible states.

    Returns:
    Dict[int, List[int]]: dictionary where keys are the amount of ones and values are lists of indices of states with that amount of ones.

    Raises:
    ValueError: if the states array is empty.
    """

    #validating states
    if len(states) == 0:
        raise ValueError("The states array is empty")

    #creating array to store amont of ones
    amont_of_ones: Dict[int, List[int]] = dict()

    #through all states
    for i in range(len(states)):
        ct = states[i].count('1')
        if ct not in amont_of_ones:
            amont_of_ones[ct] = []
        amont_of_ones[ct].append(i)

    #returning amont of ones
    return amont_of_ones

def calculate_succ_probability(counts: Dict[str, int]) -> float:
    """
    Method to calculate the probability of success of a quantum circuit.

    Parameters:
    counts (Dict[str, int]): dictionary with the counts of each state.

    Returns:
    float: the probability of success of the quantum circuit.

    Raises:
    ValueError: if the counts dict is empty.
    """
    
    #validating counts
    if len(counts) == 0:
        raise ValueError("The counts dict is empty")

    #calculating success probability
    total_count = sum(counts.values())
    succ_count = counts.get('1', 0)

    #returning success probability
    return succ_count / total_count
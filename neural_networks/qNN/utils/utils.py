"""
  utils.py

Module that defines the utility functions.

Dependencies:
- Uses numpy to use array.
- Uses typing to use Dict, List and Optional types.

Since:
- 01/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#importing libraries
import numpy as np #type: ignore
from typing import Dict, List, Optional

def get_bin_int(data: int, number_of_qubits: Optional[int] = None) -> str:
    """
    Method to get binary representation of an integer.

    Parameters:
    data (int): the integer to be converted to binary.
    number_of_qubits (Optional[int]): the number of qubits to fill with zeros if the binary representation is smaller.

    Returns:
    str: the binary representation of the given data.
    """

    #checking if number_of_qubits is None
    if number_of_qubits:
        return bin(data)[2:].zfill(np.power(2, number_of_qubits))
    
    #returning binary
    return bin(data)[2:]

def get_vector_from_int(data: int, num_qubits: int) -> np.ndarray:
    """
    Method to get a vector representation of an integer.

    Parameters:
    data (int): the integer to be converted to vector.
    num_qubits (int): the number of qubits to create the vector.

    Returns:
    np.ndarray: the vector representation of the given data.
    """

    #creating data vector
    bin_data = get_bin_int(data, num_qubits)
    data_vector = np.empty(np.power(2, num_qubits))

    #filling data vector
    for i, bit in enumerate(bin_data):
        data_vector[i] = np.power(-1, int(bit))

    #returning data vector
    return data_vector
"""
  current_circuit.py

Module that defines the quantum current quantum circuit.

Dependencies:
- Uses current_circuit.py module to define and run the current quantum circuit
- Uses qiskit_ibm_runtime module to defines service of the quantum circuit
- Uses math module to get the infinity value

Since:
- 11/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do qiskit necessary imports
from current_circuit import *
from qiskit_ibm_runtime import QiskitRuntimeService # type: ignore
import math
import random

def main(service):
    """
    Main function of the program.

    This function is the main entry point of the program.
    It will run the quantum circuit and compute the error
    of the neuron for a given set of inputs and weights.

    The function will also update the weights of the neuron
    to the best weights that resulted in the lowest error.

    Parameters:
    None

    Returns:
    None
    """
    #defines the weights and delta
    #weight1 = math.pi*random.random()
    #weight2 = math.pi*random.random()
    #weight3 = math.pi*random.random()
    #weight4 = math.pi*random.random()
    #delta = 0.1

    #defines the inputs
    inputs = [(0,0),(1,0),(0,1),(1,1)]

    #print("Weight1: ",weight1)
    #print("Weight2: ",weight2)
    #print("Weight3: ",weight3)
    #print("Weight4: ",weight4)

    #run for the first time
    #error = compute_error(inputs,weight1,weight2,weight3,weight4,service)

    grid_grain = 5
    frac = math.pi/grid_grain
    error = 5

    for i in range(grid_grain):
        weight1 = i*frac
        for j in range(grid_grain):
            weight2 = j*frac
            for k in range(grid_grain):
                weight3 = k*frac
                for l in range(grid_grain):
                    weight4 = l*frac

                    print(i,j,k,l)

                    current_error = compute_error(inputs,weight1,weight2,weight3,weight4,service)

                    if current_error < error:
                        error = current_error
                        print("Weight1: ",weight1)
                        print("Weight2: ",weight2)
                        print("Weight3: ",weight3)
                        print("Weight4: ",weight4)
                        print("Current Error: ",error)

def compute_error(inputs,weight1,weight2,weight3,weight4,service):
    """
    Compute the error of the quantum circuit for a given set of inputs and weights.

    Parameters:
    inputs (list): A list of tuples containing the input values.
    weight1 (float): The weight of the first input to the neuron.
    weight2 (float): The weight of the second input to the neuron.

    Returns:
    error (int): The total error of the quantum circuit for the given set of inputs and weights.
    """
    #initializes the error
    error = 0

    for input1,input2 in inputs:

        #defines the expected output
        expected_output = input1^input2

        #initializes the quantum circuit
        qc = current_circuit(3,1)

        #adds the neuron to the circuit
        qc.add_bin_neuron3(input1,input2,weight1,weight2,weight3,weight4,0,0)

        #runs (simulates) the circuit and save the result
        run = qc.run_circuit("3",service)
        output = int(max(run.keys(), key=run.get))

        #print("             ",abs(expected_output-output)," ",expected_output," ",output)
        #sum current error in the total error
        error += abs(expected_output - output)

    return error

if __name__ == "__main__":
    
    #connects to the service
    service = QiskitRuntimeService()

    main(service)
    print("I already learned!")
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
    weight1 = math.pi*random.random()
    weight2 = math.pi*random.random()
    weight3 = math.pi*random.random()
    weight4 = math.pi*random.random()
    delta = 0.1

    #defines the inputs
    inputs = [(0,0),(1,0),(0,1),(1,1)]

    print("Weight1: ",weight1)
    print("Weight2: ",weight2)
    print("Weight3: ",weight3)
    print("Weight4: ",weight4)

    #run for the first time
    error = compute_error(inputs,weight1,weight2,weight3,weight4,service)

    #controls with the weights are the same
    change_weights = True

    #controls with the delta is the same
    change_delta = False

    iterations = 0
    while error >= 1:
        iterations += 1

        if not change_weights:
            delta = delta*2
            change_delta = True

        change_weights = False

        for theta1 in (weight1-delta,weight1,weight1+delta):
            for theta2 in (weight2-delta,weight2,weight2+delta):
                for theta3 in (weight3-delta,weight3,weight3+delta):
                    for theta4 in (weight4-delta,weight4,weight4+delta):

                        #compare the new weights with the old ones
                        if not theta1 == weight1 or not theta2 == weight2 or not theta3 == weight3 or not theta4 == weight4:

                            #make sure the weights are between 0 and 2pi
                            theta1 = theta1%(2*math.pi)
                            theta2 = theta2%(2*math.pi)
                            theta3 = theta3%(2*math.pi)
                            theta4 = theta4%(2*math.pi)

                            print("     Theta1: ",theta1)
                            print("     Theta2: ",theta2)
                            print("     Theta3: ",theta3)
                            print("     Theta4: ",theta4)
                            #compute the error
                            current_error = compute_error(inputs,theta1,theta2,theta3,theta4,service)
                            print("     Current Error: ",current_error)

                            if current_error < 1:
                                weight1 = theta1
                                weight2 = theta2
                                weight3 = theta3
                                weight4 = theta4
                                exit()

                            #update the weights if the new error is lower
                            if current_error < error:
                                weight1 = theta1
                                weight2 = theta2
                                weight3 = theta3
                                weight4 = theta4
                                error = current_error
                                change_weights = True

        #make sure the weights are between 0 and 2pi    
        weight1 = weight1%(2*math.pi)
        weight2 = weight2%(2*math.pi)
        weight3 = weight3%(2*math.pi)
        weight4 = weight4%(2*math.pi)

        #save the weights and the error in the file
        if iterations == 1:
            file = open("analize.txt","w")
            file.write("Weight1: "+str(weight1)+"\n")
            file.write("Weight2: "+str(weight2)+"\n")
            file.write("Weight3: "+str(weight3)+"\n")
            file.write("Weight4: "+str(weight4)+"\n")
            file.write("Current Error: "+str(error)+"\n")
        else:
            file = open("analize.txt","a")
            file.write("Weight1: "+str(weight1)+"\n")
            file.write("Weight2: "+str(weight2)+"\n")
            file.write("Weight3: "+str(weight3)+"\n")
            file.write("Weight4: "+str(weight4)+"\n")
            file.write("Current Error: "+str(error)+"\n")
        file.close()

        if change_delta:
            change_delta = False
            delta = delta/2

    print("Weight1: ",weight1)
    print("Weight2: ",weight2)
    print("Weight3: ",weight3)
    print("Weight4: ",weight4)
    print("Error: ",error)

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
        qc = current_circuit(2,1)

        #adds the neuron to the circuit
        qc.add_bin_neuron3(input1,input2,weight1,weight2,weight3,weight4,0,0)

        #runs (simulates) the circuit and save the result
        run = qc.run_circuit("3",service)
        output = int(max(run.keys(), key=run.get))

        print("             ",abs(expected_output-output)," ",expected_output," ",output)
        #sum current error in the total error
        error += abs(expected_output - output)

    return error

if __name__ == "__main__":
    
    #connects to the service
    service = QiskitRuntimeService()

    main(service)
    print("I already learned!")
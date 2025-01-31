"""
  qNN.py

Module that defines the quantum neuron class.

Dependencies:
- Uses typing to use List type.
- Uses numpy to use array.
- Uses tqdm to use tqdm.
- Uses wandb to use wandb.
- Uses utils to use get_vector_from_int and calculate_succ_probability functions.
- Uses qNeuron to use Neuron class.

Since:
- 01/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

from typing import List
import numpy as np #type: ignore
from tqdm import tqdm #type: ignore
#import wandb #type: ignore
from utils import (get_vector_from_int, get_int_from_vector, calculate_succ_probability)
from qNeuron import Neuron #type: ignore


class NeuronTrainer:

    def __init__(self, number_of_qbits: int, fixed_weight: int, dataset_path: str, threshold: float = 0.5, number_of_runs: int = 8192, learning_rate_positive: float = 0.5, learning_rate_negative: float = 0.5):
        """
        Constructor method.

        Parameters:
        number_of_qbits (int): number of qubits of the circuit.
        fixed_weight (int): fixed weight of the neuron.
        dataset_path (str): path to the dataset.
        threshold (float): threshold to determine if the neuron is activated or not. Default is 0.5.
        number_of_runs (int): number of runs to measure the circuit. Default is 8192.
        learning_rate_positive (float): learning rate for positive samples. Default is 0.5.
        learning_rate_negative (float): learning rate for negative samples. Default is 0.5.
        """
        self.number_of_qbits = number_of_qbits
        self.fixed_weight = fixed_weight
        #self.data = self.read_dataset(dataset_path)
        self.threshold = threshold
        self.number_of_runs = number_of_runs
        self.learning_rate_positive = learning_rate_positive
        self.learning_rate_negative = learning_rate_negative
        self.accumulate_loss: List[float] = []
        self.number_of_steps = 0

        #initializing qNeuron
        self.Neuron = Neuron(number_of_qbits)

        #initializing random weight for the training
        self.weight_variable = np.random.randint(np.power(2, np.power(2, number_of_qbits)))

        #wandb.init(project="quantum-Neuron", config={"learning_rate_positive": learning_rate_positive, "learning_rate_negative": learning_rate_negative, "fixed_weight": fixed_weight, "number_of_qbits": number_of_qbits, "dataset": dataset_path, "number_of_runs": number_of_runs, "threshold": threshold})

    #def read_dataset(self, filepath: str) -> np.ndarray: return np.loadtxt(filepath, dtype=np.int64, delimiter=',')

    def invert_non_matching_bits(self, input: int):
        """
        Inverts a subset of non-matching bits between the input vector and the current weight vector.

        This method first generates binary vectors for the input and the weight variable. It then 
        identifies the indices where these vectors differ. A percentage of these non-matching indices, 
        determined by the positive learning rate, are randomly selected, and their corresponding bits 
        in the weight vector are inverted. Finally, the updated weight vector is converted back to 
        an integer and saved as the new weight variable.

        Parameters:
        input (int): The input integer from which a binary vector is derived for comparison.
        """

        #getting input and weight vectors
        input_vector = get_vector_from_int(input, self.number_of_qbits)
        weight_vector = get_vector_from_int(self.weight_variable, self.number_of_qbits)

        #selecting non matching bits
        non_match_ids = np.where(input_vector != weight_vector)[0]
        number_of_select = int(np.ceil(len(non_match_ids) * self.learning_rate_positive))
        selected_ids = np.random.choice(non_match_ids, number_of_select, replace=False)

        #inverting random selected bits
        for id in selected_ids:
            weight_vector[id] *= -1

        #saving new weight
        self.weight_variable = get_int_from_vector(weight_vector, self.number_of_qbits)

    def invert_matching_bits(self, input: int):
        """
        Inverts a subset of matching bits between the input vector and the current weight vector.

        This method first generates binary vectors for the input and the weight variable. It then 
        identifies the indices where these vectors match. A percentage of these matching indices, 
        determined by the negative learning rate, are randomly selected, and their corresponding bits 
        in the weight vector are inverted. Finally, the updated weight vector is converted back to 
        an integer and saved as the new weight variable.

        Parameters:
        input (int): The input integer from which a binary vector is derived for comparison.
        """

        #getting input and weight vectors
        input_vector = get_vector_from_int(input, self.number_of_qbits)
        weight_vector = get_vector_from_int(self.weight_variable, self.number_of_qbits)

        #selecting matching bits
        match_ids = np.where(input_vector == weight_vector)[0]
        number_of_select = int(np.ceil(len(match_ids) * self.learning_rate_negative))
        selected_ids = np.random.choice(match_ids, number_of_select, replace=False)

        #inverting random selected bits
        for id in selected_ids:
            weight_vector[id] *= -1

        #saving new weight
        self.weight_variable = get_int_from_vector(weight_vector, self.number_of_qbits)

    def calc_loss(self):
        """
        Calculates the loss of the quantum neuron using the fixed weight and the current weight variable as input.

        This method first sets the input and weight of the neuron to the current weight variable and the fixed weight, 
        respectively. It then builds the quantum circuit of the neuron and measures it to obtain the output distribution. 
        The loss is calculated as the probability of success in this distribution. The loss is then returned.

        Returns:
        float: The loss of the quantum neuron.
        """

        #setting input and weight
        self.Neuron.input = self.weight_variable
        self.Neuron.weight = self.fixed_weight

        #building circuit
        self.Neuron.build_circuit()

        #calculating loss
        loss = calculate_succ_probability(self.Neuron.measure_circuit(self.number_of_runs))

        #returning loss
        return loss

    def train_step(self, input: int, label: int):
        """
        Trains the quantum neuron one step with the given input and label.

        Parameters:
        input (int): the input to the neuron.
        label (int): the label of the input.

        Returns:
        bool: True if the training converged, False otherwise.
        """

        #setting input and weight
        self.Neuron.input = input
        self.Neuron.weight = self.weight_variable

        #building circuit
        self.Neuron.build_circuit()

        #calculating probability
        probability = calculate_succ_probability(self.Neuron.measure_circuit(self.number_of_runs))

        #getting currentloss
        loss = self.calc_loss()

        #accumulating loss
        self.accumulate_loss.append(loss)

        #updating number of steps
        self.number_of_steps += 1

        #checking for convergence
        if int(loss) == 1:
            #print("Training converged at step: {}".format(self.number_of_steps))
            return True
        if probability > self.threshold:
            pred = 1
        else:
            pred = 0
        if label == 1 and pred == 0:
            self.invert_non_matching_bits(input)
            #wandb.log({"probability": loss, "weight": self.weight_variable})
        elif label == 0 and pred == 1:
            self.invert_matching_bits(input)
            #wandb.log({"probability": loss, "weight": self.weight_variable})
        return False

    def train_epoch(self, epoch: int):
        """
        Trains the quantum neuron for one epoch using the dataset.

        This method iterates over the dataset, performing a training step for each data point 
        consisting of an input and its corresponding label. It checks for convergence after 
        each step, and if the training converges, it returns True immediately. Otherwise, it 
        returns False after processing the entire dataset.

        Parameters:
        epoch (int): The current epoch number.

        Returns:
        bool: True if the training converged during this epoch, False otherwise.
        """

        #iterating over dataset
        for i in tqdm(range(self.data.shape[0])):

            #getting input and label
            input = self.data[i, 0]
            label = self.data[i, 1]

            #training
            converged = self.train_step(input, label)

            #checking for convergence
            if converged:

                #returning True
                return True
            
        #returning False
        return False

    def train(self, number_of_epochs: int):
        """
        Trains the quantum neuron using the dataset for the given number of epochs.

        This method iterates over the given number of epochs, performing a training epoch 
        for each epoch. If the training converges at any point, it breaks out of the loop 
        and returns immediately. Otherwise, it continues training until all epochs have 
        been processed.

        Parameters:
        number_of_epochs (int): The number of epochs to train for.

        Returns:
        None
        """

        #iterating over epochs
        for i in range(number_of_epochs):

            #training
            converged = self.train_epoch(i)

            #checking for convergence
            if converged:
                break
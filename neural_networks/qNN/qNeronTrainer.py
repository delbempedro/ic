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
import wandb #type: ignore
from utils import (get_vector_from_int, get_int_from_vector, calculate_succ_probability)
from qNeuron import Neuron #type: ignore


class NeuronTrainer:
    def __init__(self, number_of_qbits: int, fixed_weight: int, dataset_path: str, threshold: float = 0.5, num_runs: int = 8192, learning_rate_positive: float = 0.5, learning_rate_negative: float = 0.5):

        self.number_of_qbits = number_of_qbits
        self.fixed_weight = fixed_weight
        self.data = self.read_dataset(dataset_path)
        self.threshold = threshold
        self.num_runs = num_runs
        self.learning_rate_positive = learning_rate_positive
        self.learning_rate_negative = learning_rate_negative
        self.Neuron = Neuron(number_of_qbits)
        self.accumulate_loss: List[float] = []
        self.num_steps = 0

        # Initializing random weight for the training
        self.weight_variable = np.random.randint(
            np.power(2, np.power(2, number_of_qbits)))

        wandb.init(
            project="quantum-Neuron",
            config={
                "learning_rate_positive": learning_rate_positive,
                "learning_rate_neg": learning_rate_negative,
                "fixed_weight": fixed_weight,
                "number_of_qbits": number_of_qbits,
                "dataset": dataset_path,
                "num_runs": num_runs,
                "threshold": threshold
            }
        )

    def read_dataset(self, filepath: str) -> np.ndarray:

        return np.loadtxt(filepath, dtype=np.int64, delimiter=',')

    def invert_non_matching_bits(self, input: int):

        input_vector = get_vector_from_int(input, self.number_of_qbits)
        weight_vector = get_vector_from_int(self.weight_variable,
                                            self.number_of_qbits)
        non_match_ids = np.where(input_vector != weight_vector)[0]
        num_select = int(np.ceil(len(non_match_ids) * self.learning_rate_positive))
        selected_ids = np.random.choice(non_match_ids,
                                        num_select,
                                        replace=False)
        for id in selected_ids:
            weight_vector[id] *= -1
        self.weight_variable = get_int_from_vector(weight_vector,
                                                   self.number_of_qbits)

    def invert_matching_bits(self, input: int):

        input_vector = get_vector_from_int(input, self.number_of_qbits)
        weight_vector = get_vector_from_int(self.weight_variable,
                                            self.number_of_qbits)
        match_ids = np.where(input_vector == weight_vector)[0]
        num_select = int(np.ceil(len(match_ids) * self.learning_rate_negative))
        selected_ids = np.random.choice(match_ids,
                                        num_select,
                                        replace=False)
        for id in selected_ids:
            weight_vector[id] *= -1
        self.weight_variable = get_int_from_vector(weight_vector,
                                                   self.number_of_qbits)

    def calc_loss(self):

        self.Neuron.input = self.weight_variable
        self.Neuron.weight = self.fixed_weight
        self.Neuron.build_circuit()
        loss = calculate_succ_probability(
            self.Neuron.measure_circuit(self.num_runs))
        return loss

    def train_step(self, input: int, label: int):

        self.Neuron.input = input
        self.Neuron.weight = self.weight_variable
        self.Neuron.build_circuit()
        prob = calculate_succ_probability(
            self.Neuron.measure_circuit(self.num_runs))
        loss = self.calc_loss()
        self.accumulate_loss.append(loss)
        self.num_steps += 1
        if int(loss) == 1:
            print("Training converged at step: {}".format(self.num_steps))
            return True
        if prob > self.threshold:
            pred = 1
        else:
            pred = 0
        if label == 1 and pred == 0:
            self.invert_non_matching_bits(input)
            wandb.log({"probability": loss, "weight": self.weight_variable})
        elif label == 0 and pred == 1:
            self.invert_matching_bits(input)
            wandb.log({"probability": loss, "weight": self.weight_variable})
        return False

    def train_epoch(self, epoch: int):

        for i in tqdm(range(self.data.shape[0])):
            input = self.data[i, 0]
            label = self.data[i, 1]
            converged = self.train_step(input, label)
            if converged:
                return True
        return False

    def train(self, num_epochs: int):

        for i in range(num_epochs):
            converged = self.train_epoch(i)
            if converged:
                break
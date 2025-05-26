"""
  trainer_circuit.py

Module that defines the trainer for the quantum neural network.

Dependencies:
- Uses itertools for generating combinations of parameters.
- Uses functools for partial function application.
- Uses numpy for numerical operations.

Since:
- 04/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do qiskit necessary imports

#do necessary imports
from itertools import product
from functools import partial
import numpy as np # type: ignore

#do my necessary imports
from trainer_utils import *

class trainer_qNN():
    
    def __init__(self,grid_grain=10,number_of_runs=1,number_of_shots=1024,number_of_inputs=2,type_of_run="simulation", save_history=False, tolerance=0.25, logic_gate="XOR", type_of_encoding=None, number_of_inputs_per_qubit=2, learning_rate=0.1, temperature=1.0, final_temperature=1e-3, alpha=0.95, population_size=20, mutation_rate=0.1, grid_start_index=0, grid_ranges=[]):
        """
        Create trainer for the quantum neural network.
        """
        #define random number generator
        #self._rng = rng or np.random.default_rng()

        #define simple attributes
        self._grid_grain = grid_grain
        self._number_of_runs = number_of_runs
        self._number_of_shots = number_of_shots
        self._number_of_inputs = number_of_inputs
        self._type_of_run = type_of_run
        self._save_history = save_history
        self._tolerance = tolerance
        self._logic_gate = logic_gate
        self._type_of_encoding = type_of_encoding

        #define type of encoding
        if type_of_encoding == "phase":
            self._number_of_inputs_per_qubit = number_of_inputs_per_qubit
        elif type_of_encoding == "amplitude":
            self._number_of_inputs_per_qubit = None
        else:
            raise ValueError("Invalid type of encoding.")

        if len(grid_ranges):
            self._grid_ranges = grid_ranges
        else:
            self._grid_ranges = None
        
        #define evaluate function
        partial_phase_qNN_evaluate = partial(phase_qNN_evaluate, number_of_inputs_per_qubit = self._number_of_inputs_per_qubit)
        self._evaluate_function = {"amplitude": amplitude_qNN_evaluate, "phase": partial_phase_qNN_evaluate}

        #define training methods
        if grid_start_index < grid_grain: #if grid start index is valid
            partial_cg_exhaustive_search = partial(self._cg_exhaustive_search, grid_start_index=grid_start_index)
        else:
            raise ValueError("Invalid grid start index.")
        partial_gradient_descent = partial(self._gradient_descent, learning_rate=learning_rate)
        partial_simulated_annealing = partial(self._simulated_annealing, temperature=temperature, final_temperature=final_temperature, alpha=alpha)
        partial_genetic_algorithm = partial(self._genetic_algorithm, population_size=population_size, mutation_rate=mutation_rate)
        self._training_methods = {"cg-exhaustive_search": partial_cg_exhaustive_search, "gradient_descent": partial_gradient_descent, "random_search": self._random_search, "simulated_annealing": partial_simulated_annealing, "genetic_algorithm": partial_genetic_algorithm}
        
        #define history list
        if save_history:
            self._history_dictonary = {"Error": [],"Best Error": [],"Parameters": [],"Best Parameters": [],"Counts": []}
        else:
            self._history_dictonary = None

        #define inputs and expected outputs
        self._inputs = [list(t) for t in product([0, 1], repeat=self._number_of_inputs)]
        self._expected_outputs = compute_expected_outputs(self._inputs, logic_gate=self._logic_gate)

    def get_results(self):
        """
        Get the results of the training.
        """
        if self._final_parameters is not None:
            #define dictonary
            dictonary_with_results = {"Final Parameters": [float(((parameter+np.pi)%2*np.pi)-np.pi) for parameter in self._final_parameters], "Final Error": self._final_error, "Number of Iterations": self._final_number_of_iterations, "History List": self._history_dictonary if self._history_dictonary is not None else []}

            return dictonary_with_results
        else:
            raise ValueError("The qNN has not been trained.")            
    
    def train(self, type_of_training=None):
        """
        Train the quantum neural network.
        """
        try:
            self._training_methods[type_of_training]()
        except KeyError:
            raise ValueError("Invalid type of training.")

    def _cg_exhaustive_search(self, grid_start_index=0):
        """
        Perform an exhaustive search to find optimal parameters for the quantum neural network.

        This function performs an exaustive search over a grid of parameters. The algorithm starts with a grid of parameters
        and iteratively evaluates them based on the quantum neural network's total error. The parameters that minimize the error
        are stored, and the search stops if the error falls below a specified tolerance.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        #initialize final error
        self._final_error = 1.1 #maximum possible error is 1.0

        #initialize final parameters
        self._final_parameters = [0]*(self._number_of_inputs+1)

        #initialize grid
        parameter_ranges = []

        number_of_parameters = {"amplitude":self._number_of_inputs*2,"phase":self._number_of_inputs+1}

        #define grid
        if self._grid_ranges is None:
            base_grid = np.linspace(-np.pi, np.pi, self._grid_grain)
            base_grid = np.roll(base_grid, -grid_start_index)
            parameter_ranges = [base_grid]*number_of_parameters[self._type_of_encoding]
        else:
            for start, end in self._grid_ranges:
                grid = np.linspace(start, end, self._grid_grain)
                grid = np.roll(grid, -grid_start_index)
                parameter_ranges.append(grid)

        #initialize iteration counter
        self._final_number_of_iterations = 0

        #exaustive search
        for parameters in product(*parameter_ranges):

            #update iteration counter
            self._final_number_of_iterations += 1

            if self._save_history:
                #compute total error and counts
                current_error, counts = self._evaluate_function[self._type_of_encoding](self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run, save_counts=True)
                self._history_dictonary["Counts"].append(counts)

            else:
                #compute total error
                current_error = self._evaluate_function[self._type_of_encoding](self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)
            
            #update final error and final parameters
            self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

            #save history
            if self._save_history:
                self._history_dictonary["Error"].append(current_error)
                self._history_dictonary["Best Error"].append(self._final_error)
                self._history_dictonary["Parameters"].append([float(((parameter+np.pi)%2*np.pi)-np.pi) for parameter in parameters])
                self._history_dictonary["Best Parameters"].append([float(((parameter+np.pi)%2*np.pi)-np.pi) for parameter in self._final_parameters])

            #check convergence
            if self._final_error < self._tolerance:

                return self.get_results()

        #return final parameters
        return self.get_results()

    def _gradient_descent(self, learning_rate=0.1):
        """
        Perform gradient descent to find optimal parameters for quantum neural network.

        This function uses gradient descent to search the parameter space of the quantum neural network.
        The algorithm starts with a random set of parameters and iteratively updates them in the direction
        of the gradient of the error function. The algorithm stops if the error falls below a specified
        tolerance or if the maximum number of iterations is reached.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        #initialize parameters randomly within [-pi, pi]
        parameters = random_parameters(type_of_encoding=self._type_of_encoding, number_of_inputs=self._number_of_inputs)

        #initialize final error and final parameters
        self._final_error = self._evaluate_function[self._type_of_encoding](self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)
        self._final_parameters = parameters.copy()

        #define the learning rate
        learning_rate = learning_rate

        #gradient descent
        for iteration in range(self._grid_grain**len(parameters)):

            #atualize iteration counter
            self._final_number_of_iterations = iteration+1

            #compute gradient
            gradient = compute_gradient(parameters, self._inputs, self._expected_outputs, self._number_of_inputs, self._number_of_runs, self._number_of_shots, self._type_of_run, evaluate_function=self._evaluate_function[self._type_of_encoding])

            #update parameters
            parameters -= learning_rate * gradient

            #compute current error
            current_error = self._evaluate_function[self._type_of_encoding](self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)

            #update final error and final parameters
            self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

            #save history
            if self._save_history:
                self._history_dictonary["Error"].append(current_error)
                self._history_dictonary["Best Error"].append(self._final_error)
                self._history_dictonary["Parameters"].append([float(((parameter+np.pi)%2*np.pi)-np.pi) for parameter in parameters])
                self._history_dictonary["Best Parameters"].append([float(((parameter+np.pi)%2*np.pi)-np.pi) for parameter in self._final_parameters])

            #check for convergence
            if current_error < self._tolerance:

                return self.get_results()   

        return self.get_results()

    def _random_search(self):        
        """
        Perform a random search to find optimal parameters for the quantum neural network.

        This function uses a random search to search the parameter space of the quantum neural network.
        The algorithm starts with a random set of parameters and iteratively updates them with a new
        random set of parameters. The algorithm stops if the error falls below a specified tolerance or
        if the maximum number of iterations is reached.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        #initialize final error and final parameters
        self._final_error = 1.1 #maximum possible error is 1.0
        self._final_parameters = None

        #initialize parameters
        parameters = random_parameters(type_of_encoding=self._type_of_encoding, number_of_inputs=self._number_of_inputs)

        #random search
        for iteration in range(self._grid_grain**len(parameters)):

            #atualize iteration counter
            self._final_number_of_iterations = iteration+1

            #update parameters and current error
            parameters = random_parameters(type_of_encoding=self._type_of_encoding, number_of_inputs=self._number_of_inputs)
            current_error = self._evaluate_function[self._type_of_encoding](self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)

            #update final error and final parameters
            self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

            #save history
            if self._save_history:
                self._history_dictonary["Error"].append(current_error)
                self._history_dictonary["Best Error"].append(self._final_error)
                self._history_dictonary["Parameters"].append([float(((parameter+np.pi)%2*np.pi)-np.pi) for parameter in parameters])
                self._history_dictonary["Best Parameters"].append([float(((parameter+np.pi)%2*np.pi)-np.pi) for parameter in self._final_parameters])

            #check for convergence
            if self._final_error < self._tolerance:

                return self.get_results()

        return self.get_results()

    def _simulated_annealing(self, temperature=1.0, final_temperature=1e-3, alpha=0.95):
        """
        Perform simulated annealing to find optimal parameters for the quantum neural network.

        This function uses simulated annealing to search the parameter space of the quantum neural network.
        The algorithm starts with a random set of parameters and iteratively updates them with a new
        random set of parameters. The algorithm stops if the error falls below a specified tolerance or
        if the maximum number of iterations is reached.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        #initialize parameters and current error
        parameters = random_parameters(type_of_encoding=self._type_of_encoding, number_of_inputs=self._number_of_inputs)
        current_error = self._evaluate_function[self._type_of_encoding](self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)
        
        #initialize final error and final parameters
        self._final_parameters = parameters.copy()
        self._final_error = current_error

        #define initial and final temperature
        temperature = temperature
        final_temperature = final_temperature

        #define alpha
        alpha = alpha

        #simulated annealing
        for iteration in range(self._grid_grain**len(parameters)):

            #atualize iteration counter
            self._final_number_of_iterations = iteration+1

            #update new parameters
            new_parameters = parameters + np.random.normal(0, 0.1, size=len(parameters))
            new_parameters = np.mod(new_parameters + np.pi, 2 * np.pi) - np.pi

            #update new error
            new_error = self._evaluate_function[self._type_of_encoding](self._inputs, self._expected_outputs, new_parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)

            #update delta
            delta = new_error - current_error

            #accept or reject
            if delta < 0 or np.exp(-delta / temperature) > np.random.rand():

                #atualize parameters and current error
                parameters = new_parameters
                current_error = new_error

                #update final error and final parameters
                self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

            #save history
            if self._save_history:
                self._history_dictonary["Error"].append(current_error)
                self._history_dictonary["Best Error"].append(self._final_error)
                self._history_dictonary["Parameters"].append([float(((parameter+np.pi)%2*np.pi)-np.pi) for parameter in parameters])
                self._history_dictonary["Best Parameters"].append([float(((parameter+np.pi)%2*np.pi)-np.pi) for parameter in self._final_parameters])

            #check convergence
            if temperature < final_temperature or self._final_error < self._tolerance:

                return self.get_results()
            
            #update temperature
            temperature *= alpha

        return self.get_results()
    
    def _genetic_algorithm(self, population_size=20, mutation_rate=0.1):
        """
        Perform a genetic algorithm to find the optimal parameters for the quantum neural network.

        This function uses a genetic algorithm to search the parameter space of the quantum neural network.
        The algorithm starts with a random population of parameters and iteratively updates them using
        the principle of survival of the fittest. The algorithm stops if the error falls below a specified
        tolerance or if the maximum number of iterations is reached.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        #define constants
        population_size = population_size
        mutation_rate = mutation_rate

        #initialize population
        population = [list(random_parameters(type_of_encoding=self._type_of_encoding, number_of_inputs=self._number_of_inputs)) for _ in range(population_size)]
        

        for generation in range(int( (self._grid_grain/population_size)**len(population[0]) )):

            #atualize iteration counter
            self._final_number_of_iterations = generation+1

            #compute errors
            errors = [self._evaluate_function[self._type_of_encoding](self._inputs, self._expected_outputs, individual, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run) for individual in population]

            #get best error
            best_idx = int(np.argmin(errors))

            #update final parameters and final error
            self._final_parameters = population[best_idx]
            self._final_error = errors[best_idx]

            #save history
            if self._save_history:
                self._history_dictonary["Best Error"].append(self._final_error)
                self._history_dictonary["Best Parameters"].append([float(((parameter+np.pi)%2*np.pi)-np.pi) for parameter in self._final_parameters])

            #check convergence
            if self._final_error < self._tolerance:
                    
                return self.get_results()

            #create new population list
            new_population = []

            #generate descendents
            while len(new_population) < population_size:
                p1, p2 = select_parents(population, errors)
                c1, c2 = crossover(p1, p2)
                new_population.extend([mutate(c1, mutation_rate=mutation_rate), mutate(c2, mutation_rate=mutation_rate)])

            #update population
            population = new_population[:population_size]

        return self.get_results()
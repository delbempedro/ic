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
    
    def __init__(self,grid_grain=10,number_of_runs=1,number_of_shots=1024,number_of_inputs=2,type_of_run="simulation", save_history=False, tolerance=0.25, logic_gate="XOR", type_of_enconding=None, number_of_inputs_per_qubit=2):
        """
        Create trainer for the quantum neural network.
        """

        #define simple attributes
        self._grid_grain = grid_grain
        self._number_of_runs = number_of_runs
        self._number_of_shots = number_of_shots
        self._number_of_inputs = number_of_inputs
        self._type_of_run = type_of_run
        self._save_history = save_history
        self._tolerance = tolerance
        self._logic_gate = logic_gate
        self._max_iterations = number_of_inputs**grid_grain
        self._type_of_enconding = type_of_enconding
        self._control_flag = False

        #define type of enconding
        if type_of_enconding == "phase":
            self._number_of_inputs_per_qubit = number_of_inputs_per_qubit
        elif type_of_enconding == "amplitude":
            self._number_of_inputs_per_qubit = None
        else:
            raise ValueError("Invalid type of enconding.")
        
        #define evaluate function
        partial_phase_qNN_evaluate = partial(phase_qNN_evaluate, number_of_inputs_per_qubit = self._number_of_inputs_per_qubit)
        self._evaluate_function = {"amplitude": amplitude_qNN_evaluate, "phase": partial_phase_qNN_evaluate}
        
        #define history list
        if save_history:
            self.history_list = []
        else:
            self.history_list = None

        #define inputs and expected outputs
        self._inputs = [list(t) for t in product([0, 1], repeat=self._number_of_inputs)]
        self._expected_outputs = compute_expected_outputs(self._inputs, logic_gate=self._logic_gate)

    def get_results(self):
        """
        Get the results of the training.
        """
        if type(self._final_parameters) == None:
            raise ValueError("The qNN has not been trained.")
        else:
            #define dictonary
            dictonary_with_results = {"Final Parameters": [float(parameter%(2*np.pi)) for parameter in self._final_parameters], "Final Error": self._final_error, "Number of Iterations": self._final_number_of_iterations, "History List": self._history_list}

            return dictonary_with_results
    
    def train(self, type_of_training=None):
        """
        Train the quantum neural network.
        """
        #switch control flag
        self._control_flag = True

        if type_of_training == "exaustive_search":

            return self.exaustive_search()
            
        elif type_of_training == "gradient_descent":

            return self.gradient_descent()
            
        elif type_of_training == "random_search":

            return self.random_search()
            
        elif type_of_training == "simulated_annealing":

            return self.simulated_annealing()
            
        elif type_of_training == "genetic_algorithm":

            return self.genetic_algorithm()

        else:

            raise ValueError("Invalid type of training.")

    def exaustive_search(self):
        """
        Perform an exaustive search to find optimal parameters for the quantum neural network.

        This function performs an exaustive search over a grid of parameters. The algorithm starts with a grid of parameters
        and iteratively evaluates them based on the quantum neural network's total error. The parameters that minimize the error
        are stored, and the search stops if the error falls below a specified tolerance.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        if self._control_flag:

            #initialize final error
            self._final_error = 1.1 #maximum possible error is 1.0

            #initialize final parameters
            self._final_parameters = [0]*(self._number_of_inputs+1)

            #initialize grid
            grid = np.linspace(-np.pi, np.pi, self._grid_grain)

            #initialize history list
            self._history_list = []

            #initialize iteration counter
            self._final_number_of_iterations = 0

            #exaustive search
            for parameters in product(grid, repeat=(self._number_of_inputs+1)):

                #update iteration counter
                self._final_number_of_iterations += 1

                #compute total error
                current_error = self._evaluate_function[self._type_of_enconding](self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)

                #update final error and final parameters
                self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

                #save history
                if self._save_history:
                    self._history_list.append(self._final_error)

                #check convergence
                if self._final_error < self._tolerance:

                    #switch control flag
                    self._control_flag = False

                    return self.get_results()

            #switch control flag
            self._control_flag = False

            #return final parameters
            return self.get_results()
        
        else:
            raise ValueError("The method cannot be called directly.")

    def gradient_descent(self):
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
        if self._control_flag:

            #initialize parameters randomly within [-pi, pi]
            parameters = random_parameters(tipe_of_enconding=self._type_of_enconding, number_of_inputs=self._number_of_inputs)

            #initialize final error and final parameters
            self._final_error = self._evaluate_function[self._type_of_enconding](self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)
            self._final_parameters = parameters.copy()

            #define the learning rate
            learning_rate = 0.1

            #initialize history list
            self._history_list = []

            #gradient descent
            for iteration in range(self._max_iterations):

                #atualize iteration counter
                self._final_number_of_iterations = iteration+1

                #compute gradient
                gradient = compute_gradient(parameters, self._inputs, self._expected_outputs, self._number_of_inputs, self._number_of_runs, self._number_of_shots, self._type_of_run, evaluate_function=self._evaluate_function[self._type_of_enconding])

                #update parameters
                parameters -= learning_rate * gradient

                #compute current error
                current_error = self._evaluate_function[self._type_of_enconding](self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)

                #update final error and final parameters
                self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

                #save history
                if self._save_history:
                    self._history_list.append(self._final_error)

                #check for convergence
                if current_error < self._tolerance:

                    #switch control flag
                    self._control_flag = False

                    return self.get_results()

            #switch control flag
            self._control_flag = False      

            return self.get_results()
        
        else:

            raise ValueError("The method cannot be called directly.")

    def random_search(self):        
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
        if self._control_flag:

            #initialize final error and final parameters
            self._final_error = 1.1 #maximum possible error is 1.0
            self._final_parameters = None

            #initialize history list
            self._history_list = []

            #random search
            for iteration in range(self._max_iterations):

                #atualize iteration counter
                self._final_number_of_iterations = iteration+1

                #update parameters and current error
                parameters = random_parameters(tipe_of_enconding=self._type_of_enconding, number_of_inputs=self._number_of_inputs)
                current_error = self._evaluate_function[self._type_of_enconding](self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)

                #update final error and final parameters
                self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

                #save history
                if self._save_history:
                    self._history_list.append(self._final_error)

                #check for convergence
                if self._final_error < self._tolerance:

                    #switch control flag
                    self._control_flag = False

                    return self.get_results()

            #switch control flag
            self._control_flag = False

            return self.get_results()
        
        else:

            raise ValueError("The method cannot be called directly.")
    
    def simulated_annealing(self):
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
        if self._control_flag:
            #initialize parameters and current error
            parameters = random_parameters(tipe_of_enconding=self._type_of_enconding, number_of_inputs=self._number_of_inputs)
            current_error = self._evaluate_function[self._type_of_enconding](self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)
            
            #initialize final error and final parameters
            self._final_parameters = parameters.copy()
            self._final_error = current_error

            #define initial and final temperature
            temperature = 1.0
            final_temperature = 1e-3

            #define alpha
            alpha = 0.95

            #initialize history list
            self._history_list = []

            #simulated annealing
            for iteration in range(self._max_iterations):

                #atualize iteration counter
                self._final_number_of_iterations = iteration+1

                #update new parameters
                new_parameters = parameters + np.random.normal(0, 0.1, size=len(parameters))
                new_parameters = np.mod(new_parameters + np.pi, 2 * np.pi) - np.pi

                #update new error
                new_error = self._evaluate_function[self._type_of_enconding](self._inputs, self._expected_outputs, new_parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)

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
                    self._history_list.append(self._final_error)

                #check convergence
                if temperature < final_temperature or self._final_error < self._tolerance:

                    #switch control flag
                    self._control_flag = False

                    return self.get_results()
                
                #update temperature
                temperature *= alpha

            #switch control flag
            self._control_flag = False

            return self.get_results()
        
        else:

            raise ValueError("The method cannot be called directly.")
        
    def genetic_algorithm(self):
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
        if self._control_flag:
            #define constants
            population_size = 20
            mutation_rate = 0.1

            #initialize population
            population = [list(random_parameters(tipe_of_enconding=self._type_of_enconding, number_of_inputs=self._number_of_inputs)) for _ in range(population_size)]
            
            #initialize history list
            self._history_list = []

            for generation in range(self._max_iterations):

                #atualize iteration counter
                self._final_number_of_iterations = generation+1

                #compute errors
                errors = [self._evaluate_function[self._type_of_enconding](self._inputs, self._expected_outputs, individual, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run) for individual in population]

                #get best error
                best_idx = int(np.argmin(errors))

                #update final parameters and final error
                self._final_parameters = population[best_idx]
                self._final_error = errors[best_idx]

                #save history
                if self._save_history:
                    self._history_list.append(self._final_error)

                #check convergence
                if self._final_error < self._tolerance:
                    
                    #switch control flag
                    self._control_flag = False

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

            #switch control flag
            self._control_flag = False

            return self.get_results()
        
        else:

            raise ValueError("The method cannot be called directly.")
"""
  trainer_circuit.py

Module that defines the trainer for the quantum neural network.

Dependencies:
- Uses itertools for generating combinations of parameters.
- Uses numpy for numerical operations.

Since:
- 04/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do qiskit necessary imports

#do necessary imports
from itertools import product
import numpy as np # type: ignore

#do my necessary imports
from trainer_utils import *

class trainer_qNN():
    
    def __init__(self,grid_grain=10,number_of_runs=1,number_of_shots=1024,number_of_inputs=2,type_of_run="simulation", save_history=False, tolerance=0.25, logic_gate="XOR", type_of_enconding=None, number_of_inputs_per_qubit=2):
        """
        Create trainer for the quantum neural network.
        """

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
        if type_of_enconding == "phase":
            self._number_of_inputs_per_qubit = number_of_inputs_per_qubit
        elif type_of_enconding != "amplitude":
            raise ValueError("Invalid type of enconding.")
        if save_history:
            self.history_list = []
        else:
            self.history_list = None
        self._inputs = [list(t) for t in product([0, 1], repeat=self._number_of_inputs)]
        self._expected_outputs = compute_expected_outputs(self._inputs, logic_gate=self._logic_gate)

        self._control_flag = False

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

            if self._type_of_enconding == "amplitude":

                return self.amplitude_qNN_exaustive_search()
            else:

                return self.phase_qNN_exaustive_search()
            
        elif type_of_training == "gradient_descent":

            if self._type_of_enconding == "amplitude":

                return self.amplitude_qNN_gradient_descent()
            else:

                return self.phase_qNN_gradient_descent()
            
        elif type_of_training == "random_search":

            if self._type_of_enconding == "amplitude":

                return self.amplitude_qNN_random_search()
            else:

                return self.phase_qNN_random_search()
            
        elif type_of_training == "simulated_annealing":

            if self._type_of_enconding == "amplitude":

                return self.amplitude_qNN_simulated_annealing()
            else:

                return self.phase_qNN_simulated_annealing()
            
        elif type_of_training == "genetic_algorithm":

            if self._type_of_enconding == "amplitude":

                return self.amplitude_qNN_genetic_algorithm()
            else:

                return self.phase_qNN_genetic_algorithm()

        else:

            raise ValueError("Invalid type of training.")

        #switch control flag
        self._control_flag = False

    def phase_qNN_exaustive_search(self):
        """
        Perform an exhaustive search of the parameter space to find the optimal parameters for phase qNN.

        This function iterates over all possible parameter combinations within a specified grid and evaluates
        each set of parameters based on the quantum neural network's total error. The self._final parameters that
        minimize the error are stored, and the search stops if the error falls below a specified tolerance.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        if self._control_flag:
            #initialize final error
            self._final_error = 1

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
                current_error = phase_qNN_evaluate(self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run, number_of_inputs_per_qubit=self._number_of_inputs_per_qubit)

                #update final error and final parameters
                self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

                #save history
                if self._save_history:
                    self._history_list.append(self._final_error)

                #check convergence
                if self._final_error < self._tolerance:
                    return self.get_results()

            #return final parameters
            return self.get_results()

        else:

            raise ValueError("The method cannot be called directly.")
    
    def amplitude_qNN_exaustive_search(self):
        """
        Perform an exhaustive search of the parameter space to find the optimal parameters for amplitude qNN.

        This function iterates over all possible parameter combinations within a specified grid and evaluates
        each set of parameters based on the quantum neural network's total error. The self._final parameters that
        minimize the error are stored, and the search stops if the error falls below a specified tolerance.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        if self._control_flag:
            #initialize final error
            self._final_error = 1

            #initialize final parameters
            self._final_parameters = [0]*self._number_of_inputs

            #initialize grid
            grid = np.linspace(-np.pi, np.pi, self._grid_grain)

            #initialize history list
            self._history_list = []

            #initialize iteration counter
            self._final_number_of_iterations = 0

            #exaustive search
            for parameters in product(grid, repeat=self._number_of_inputs*2):

                #update iteration counter
                self._final_number_of_iterations += 1

                #update current error and parameters
                current_error = amplitude_qNN_evaluate(self._inputs,self._expected_outputs,parameters,number_of_inputs=self._number_of_inputs)

                #update final error and final parameters
                self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

                #save history
                if self._save_history:
                    self._history_list.append(self._final_error)

                #check convergence
                if self._final_error < self._tolerance:
                    return self.get_results()

            return self.get_results()
        
        else:

            raise ValueError("The method cannot be called directly.")

    def phase_qNN_gradient_descent(self):
        """
        Optimize the quantum neural network parameters using independent gradient descent updates.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        if self._control_flag:
            #initialize parameters randomly within [-pi, pi]
            parameters = random_parameters(tipe_of_enconding=self._type_of_enconding, number_of_inputs=self._number_of_inputs)

            #initialize final error and final parameters
            self._final_error = phase_qNN_evaluate(self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run, number_of_inputs_per_qubit=self._number_of_inputs_per_qubit)
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
                gradient = phase_qNN_compute_gradient(parameters, self._inputs, self._expected_outputs, self._number_of_inputs, self._number_of_runs, self._number_of_shots, self._type_of_run)

                #update parameters
                parameters -= learning_rate * gradient

                #compute current error
                current_error = phase_qNN_evaluate(self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run, number_of_inputs_per_qubit=self._number_of_inputs_per_qubit)

                #update final error and final parameters
                self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

                #save history
                if self._save_history:
                    self._history_list.append(self._final_error)

                #check for convergence
                if current_error < self._tolerance:
                    return self.get_results()

            return self.get_results()
        
        else:

            raise ValueError("The method cannot be called directly.")

    def amplitude_qNN_gradient_descent(self):
        """
        Optimize the quantum neural network parameters using gradient descent.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        if self._control_flag:
            #initialize parameters randomly within [-pi, pi]
            parameters = random_parameters(tipe_of_enconding=self._type_of_enconding, number_of_inputs=self._number_of_inputs)

            #initialize final error and final parameters
            self._final_error = amplitude_qNN_evaluate(self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)
            self._final_parameters = parameters.copy()

            #define learning rate
            learning_rate = 0.1

            #initialize history list
            self._history_list = []

            #gradient descent
            for iteration in range(self._max_iterations):

                #atualize iteration counter
                self._final_number_of_iterations = iteration+1

                #compute gradient
                gradient = amplitude_qNN_compute_gradient(parameters, self._inputs, self._expected_outputs, self._number_of_inputs, self._number_of_runs, self._number_of_shots, self._type_of_run)

                #update parameters
                parameters -= learning_rate * gradient

                #compute current error
                current_error = amplitude_qNN_evaluate(self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run)

                #update final error and final parameters
                self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

                #save history
                if self._save_history:
                    self._history_list.append(self._final_error)

                #check for convergence
                if current_error < self._tolerance:
                    return self.get_results()

            return self.get_results()
        
        else:

            raise ValueError("The method cannot be called directly.")

    def phase_qNN_random_search(self):
        """
        Perform a random search to find optimal parameters for the phase quantum neural network.

        This function randomly samples parameter combinations and evaluates them based on the quantum neural
        network's total error. The self._final parameters that minimize the error are stored, and the search stops
        if the error falls below a specified tolerance.

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
                current_error = phase_qNN_evaluate(self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run, number_of_inputs_per_qubit=self._number_of_inputs_per_qubit)

                #update final error and final parameters
                self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

                #save history
                if self._save_history:
                    self._history_list.append(self._final_error)

                #check for convergence
                if self._final_error < self._tolerance:
                    return self.get_results()

            return self.get_results()
        
        else:

            raise ValueError("The method cannot be called directly.")
    
    def amplitude_qNN_random_search(self):
        """
        Perform a random search to find optimal parameters for the amplitude quantum neural network.

        This function randomly samples parameter combinations and evaluates them based on the quantum neural
        network's total error. The self._final parameters that minimize the error are stored, and the search stops
        if the error falls below a specified tolerance.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        if self._control_flag:
            #initialize final parameters and final error
            self._final_error = 1.1 #maximum possible error is 1.0
            self._final_parameters = None

            #initialize history list
            self._history_list = []

            #randam search
            for iteration in range(self._max_iterations):

                #atualize iteration counter
                self._final_number_of_iterations = iteration+1

                #update parameters and current error
                parameters = random_parameters(tipe_of_enconding=self._type_of_enconding, number_of_inputs=self._number_of_inputs)
                current_error = amplitude_qNN_evaluate(self._inputs, self._expected_outputs, parameters, number_of_inputs=self._number_of_inputs)

                #update final error and final parameters
                self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

                #save history
                if self._save_history:
                    self._history_list.append(self._final_error)

                #check convergence
                if self._final_error < self._tolerance:
                    return self.get_results()

            return self.get_results()
        
        else:

            raise ValueError("The method cannot be called directly.")

    def phase_qNN_simulated_annealing(self):
        """
        Perform simulated annealing to find optimal parameters for the phase quantum neural network.

        This function uses simulated annealing to search the parameter space of the quantum neural network.
        The algorithm starts with a random set of parameters and iteratively searches for better parameters
        by randomly perturbing the current parameters and accepting or rejecting the new parameters based on
        the Metropolis criterion. The algorithm stops if the error falls below a specified tolerance or if
        the maximum number of iterations is reached.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters, number of iterations (int) and history list of errors (list of floats).
        """
        if self._control_flag:
            #initialize parameters and current error
            parameters = random_parameters(tipe_of_enconding=self._type_of_enconding, number_of_inputs=self._number_of_inputs)
            current_error = phase_qNN_evaluate(self._inputs, self._expected_outputs, parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run, number_of_inputs_per_qubit=self._number_of_inputs_per_qubit)
            
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
                new_error = phase_qNN_evaluate(self._inputs, self._expected_outputs, new_parameters, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run, number_of_inputs_per_qubit=self._number_of_inputs_per_qubit)

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
                    return self.get_results()
                
                #update temperature
                temperature *= alpha

            return self.get_results()
        
        else:

            raise ValueError("The method cannot be called directly.")

    def amplitude_qNN_simulated_annealing(self):
        """
        Optimize the quantum neural network parameters using simulated annealing.

        This function initializes parameters randomly and iteratively refines them
        through a process of simulated annealing. It evaluates the parameter sets
        by computing errors and updates them based on acceptance criteria determined
        by the simulated annealing algorithm. The process continues until the temperature
        falls below a certain threshold or the error falls below a specified tolerance.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters,
        number of iterations (int), and history list of errors (list of floats).
        """
        if self._control_flag:
            #initialize parameters and current error
            parameters = random_parameters(tipe_of_enconding=self._type_of_enconding, number_of_inputs=self._number_of_inputs)
            current_error = amplitude_qNN_evaluate(self._inputs, self._expected_outputs, parameters, number_of_inputs=self._number_of_inputs)
                    
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
                new_error = amplitude_qNN_evaluate(self._inputs, self._expected_outputs, new_parameters, number_of_inputs=self._number_of_inputs)

                #update delta
                delta = new_error - current_error

                #accept or reject
                if delta < 0 or np.exp(-delta / temperature) > np.random.rand():

                    #atualize parameters and current error
                    parameters = new_parameters
                    current_error = new_error
                    
                    self._final_parameters, self._final_error = update_if_better(parameters, current_error, self._final_parameters, self._final_error)

                #save history
                if self._save_history:
                    self._history_list.append(self._final_error)

                #check convergence
                if temperature < final_temperature or self._final_error < self._tolerance:
                    return self.get_results()

                #update temperature
                temperature *= alpha

            return self.get_results()
        
        else:

            raise ValueError("The method cannot be called directly.")
        
    def phase_qNN_genetic_algorithm(self):
        """
        Perform a genetic algorithm to find optimal parameters for the phase quantum neural network.

        This function uses a genetic algorithm to search the parameter space of the quantum neural network.
        The algorithm starts with a random population of parameters and iteratively generates new parameters
        by randomly selecting parents and using crossover and mutation to generate new parameters.
        The algorithm stops if the error falls below a specified tolerance or if the maximum number of
        iterations is reached.

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
                errors = [phase_qNN_evaluate(self._inputs, self._expected_outputs, individual, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run, number_of_inputs_per_qubit=self._number_of_inputs_per_qubit) for individual in population]

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
        
        else:

            raise ValueError("The method cannot be called directly.")
    
    def amplitude_qNN_genetic_algorithm(self):
        """
        Optimize the quantum neural network parameters using genetic algorithm.

        Parameters:
        None

        Returns:
        The optimal parameters (list of floats), the total error (float) of the optimal parameters,
        number of iterations (int), and history list of errors (list of floats).
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
                errors = [amplitude_qNN_evaluate(self._inputs, self._expected_outputs, individual, number_of_runs=self._number_of_runs, number_of_shots=self._number_of_shots, number_of_inputs=self._number_of_inputs, type_of_run=self._type_of_run) for individual in population]

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
        
        else:

            raise ValueError("The method cannot be called directly.")
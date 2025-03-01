"""
  current_circuit.py

Module that defines the quantum current quantum circuit.

Dependencies:
- Uses qiskit module to define a quantum circuit
- Uses qiskit_ibm_runtime module to run the quantum circuit
- Uses qiskit_aer module to simulate the quantum circuit
- Uses qiskit.transpiler.preset_passmanagers module to transplie the quantum circuit

Since:
- 02/2025

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do qiskit necessary imports
from qiskit import QuantumCircuit #type: ignore
from qiskit.primitives import StatevectorSampler #type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler # type: ignore
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager # type: ignore

#do my necessary imports
from utils import *

class current_circuit():
    
    def __init__(self,num_of_qbits,num_of_classical_bits):
        """
        Create a new quantum circuit.
        """

        self._num_of_qbits = num_of_qbits
        self._num_of_classical_bits = num_of_classical_bits
        self._qc = QuantumCircuit(self._num_of_qbits,self._num_of_classical_bits)

    
    def get_current_circuit(self):
        """
        Get the current quantum circuit.
        
        Returns:
        The current quantum circuit (QuantumCircuit).
        """

        return self._qc

    def get_num_of_qbits(self):
        """
        Get the number of qbits in the quantum circuit.
        
        Returns:
        The number of qbits in the quantum circuit (int).
        """

        return self._num_of_qbits

    def get_num_of_classical_bits(self):
        """
        Get the number of classical bits in the quantum circuit.
        
        Returns:
        The number of classical bits in the quantum circuit (int).
        """

        return self._num_of_classical_bits
    
    def print_current_circuit(self,output='text',style='ascii'):
        """
        Print the current quantum circuit.
        """

        print(self._qc.draw(output=output,style=style))

    def add_single_qubit_neuron(self,inputs,weights,number_of_bits=2,first_qubit_index=0,number_of_inputs_per_qubit=3):
        """
        Add a quantum neuron operation to the current quantum circuit.

        Parameters:
        inputs (list of floats): The inputs to the neuron.
        weights (list of floats): The weights of the inputs to the neuron.
        number_of_bits (int): The number of qbits in the circuit.
        first_qubit_index (int): The index of the qbit to which the quantum neuron operation is applied.
        number_of_inputs_per_qubit (int): The number of inputs per qubit.
        """

        single_qubit_neuron(self._qc,inputs,weights,number_of_bits=number_of_bits,first_qubit_index=first_qubit_index,number_of_inputs_per_qubit=number_of_inputs_per_qubit)

    def add_multi_qubit_neuron(self,parameters,number_of_bits=2,first_qubit_index=0): 
        """
        Add a neuron to the current quantum circuit.
        
        Parameters:
        weight1 (float): The weight of the first input to the neuron.
        weight2 (float): The weight of the second input to the neuron.
        weight3 (float): The weight of the third input to the neuron.
        weight4 (float): The weight of the fourth input to the neuron.
        number_of_bits (int): The number of qbits in the circuit.
        first_qubit_index (int): The index of the first qbit that the neuron will use.
        """

        multi_qubit_neuron(self._qc,parameters,number_of_bits=number_of_bits,first_qubit_index=first_qubit_index)

    def evaluate(self, number_of_shots = 1024, number_of_runs = 100, type_of_run = "simulation"):
        """
        Evaluate a quantum circuit and return the counts (histogram of the outputs).

        Parameters:
        quantum_circuit (QuantumCircuit): The quantum circuit to be evaluated.
        number_of_shots (int): The number of shots to be used in the evaluation.
        number_of_runs (int): The number of times the quantum circuit is run.
        type_of_run (str): The type of run to be used in the evaluation.

        Returns:
        list: A list of dictionaries, where each dictionary represents the counts of the outputs of the quantum circuit.
        """

        #define the sampler
        if type_of_run == "real_run":
            service = QiskitRuntimeService()
            backend = service.least_busy(operational=True, simulator=False)
            pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
            self._qc = pass_manager.run(self._qc)
            sampler = Sampler(backend)
        else:
            sampler = StatevectorSampler()
        

        #get the counts
        counts = generate_counts(self._qc,sampler,number_of_runs=number_of_runs,number_of_shots=number_of_shots)

        #return the counts
        return counts
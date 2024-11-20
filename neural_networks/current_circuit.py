"""
  current_circuit.py

Module that defines the quantum current quantum circuit.

Dependencies:
- Uses neuron.py module to define a single quantum neuron
- Uses qiskit module to define a quantum circuit
- Uses qiskit_ibm_runtime module to run the quantum circuit
- Uses qiskit_aer module to simulate the quantum circuit
- Uses qiskit.transpiler.preset_passmanagers module to transplie the quantum circuit

Since:
- 11/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do qiskit necessary imports
from qiskit import QuantumCircuit # type: ignore
from qiskit_aer import AerSimulator # type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler # type: ignore
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager # type: ignore
from neuron import *

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
    
    def add_neuron(self,input1_value,input2_value,weight1,weight2,first_qbit_index,first_classical_bit_index):
        """
        Add a neuron to the current quantum circuit.
        
        Parameters:
        input1_value (int): The value of the first input to the neuron.
        input2_value (int): The value of the second input to the neuron.
        weight1 (float): The weight of the first input to the neuron.
        weight2 (float): The weight of the second input to the neuron.
        first_qbit_index (int): The index of the first qbit that the neuron will use.
        first_classical_bit_index (int): The index of the first classical bit that the neuron will use.
        """
        neuron(self._qc,input1_value,input2_value,weight1,weight2,first_qbit_index,first_classical_bit_index)

    def add_bin_neuron(self,input1_value,input2_value,weight1,weight2,first_qbit_index,first_classical_bit_index):
        """
        Add a neuron to the current quantum circuit.
        
        Parameters:
        input1_value (int): The value of the first input to the neuron.
        input2_value (int): The value of the second input to the neuron.
        weight1 (float): The weight of the first input to the neuron.
        weight2 (float): The weight of the second input to the neuron.
        first_qbit_index (int): The index of the first qbit that the neuron will use.
        first_classical_bit_index (int): The index of the first classical bit that the neuron will use.
        """
        bin_neuron(self._qc,input1_value,input2_value,weight1,weight2,first_qbit_index,first_classical_bit_index)
    
    def run_circuit(self,type_of_run,service):
        
        def results():
            if type_of_run == "1":#real run

                #transpile your circuit
                backend = service.least_busy(operational=True, simulator=False)
                pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
                qc_transpiled = pass_manager.run(self._qc)
                sampler = Sampler(backend)

                #run your circuit with samples
                job = sampler.run([qc_transpiled])
                result = job.result()

            elif type_of_run == "2":#simulation with noise
                
                #transpile your circuit
                real_backend = service.backend("ibm_sherbrooke")
                aer = AerSimulator.from_backend(real_backend) 
                pass_manager = generate_preset_pass_manager(backend=aer, optimization_level=1)
                qc_transpiled = pass_manager.run(self._qc)

                #simulete your circuit
                sampler = Sampler(mode=aer)
                result = sampler.run([qc_transpiled]).result()

            elif type_of_run == "3":#simulation without noise

                #transpile your circuit
                aer_sim = AerSimulator()
                pass_manager = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
                qc_transpiled = pass_manager.run(self._qc)

                #simulete your circuit
                with Session(backend=aer_sim) as session:
                    sampler = Sampler()
                    result = sampler.run([qc_transpiled]).result()

            return result[0].data.c.get_counts()
            
        #defines data as results counts
        data = results()

        return data
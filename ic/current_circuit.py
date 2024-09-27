"""
  current_circuit.py

Module that defines the quantun current quantum circuit.

Dependencies:
- Uses the full_adder.py module to create the class current_circuit which contains the struct full adder in a quantum circuit.

Since:
- 09/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""

#do qiskit necessary imports
from qiskit import QuantumCircuit # type: ignore
from qiskit_aer import AerSimulator # type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler # type: ignore
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager # type: ignore

#do other necessary imports
from ic.full_adder import *
from ic.simple_adder import *

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
    
    def add_simple_adder(self,qbit1_value,qbit2_value,first_qbit_index,first_classical_bit_index):
        """
        Add simple adder in quantum circuit.
        
        Parameters:
        qbit1_value (int): The value of the first qbit to be used in the simple adder.
        qbit2_value (int): The value of the second qbit to be used in the simple adder.
        first_qbit_index (int): The first qubit of the four qubits to be used in the carry out.
        first_classical_bit_index (int): The first classical bit of the four classical bits to be used in the carry out.
        
        """
        simple_adder(self._qc,qbit1_value,qbit2_value,first_qbit_index,first_classical_bit_index)

    def add_full_adder(self,qbit1_value,qbit2_value,first_qbit_index,first_classical_bit_index,carry_in=False):
        """
        Add full adder in quantum circuit.
        
        Parameters:
        qbit1_value (int): The value of the first qbit to be used in the full adder.
        qbit2_value (int): The value of the second qbit to be used in the full adder.
        carry_in (int): If the carry in is used in the full adder.
        first_qbit_index (int): The first qubit of the eight qubits to be used in the full adder.
        first_classical_bit_index (int): The first classical bit of the five classical bits to be used in the full adder.
        
        """
        full_adder(self._qc,qbit1_value,qbit2_value,first_qbit_index,first_classical_bit_index,carry_in)

    def run_circuit(self,type_of_run,service):
        
        
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
        
        #defines data as results counts
        data = result[0].data.c.get_counts()

        return data
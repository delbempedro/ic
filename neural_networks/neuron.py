"""
  neuron.py

Module that defines a single quantum neuron

Dependencies:
- Uses the qiskit_ibm_runtime module to connect to the service.
- Uses analizes_results.py to get results.

Since:
- 11/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""
import importlib.util
import os


module_path = os.path.join(os.path.dirname(__file__), '../ic/current_circuit.py')
module_path = os.path.abspath(module_path)
spec = importlib.util.spec_from_file_location("current_circuit", module_path)
current_circuit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(current_circuit)
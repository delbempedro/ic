"""
  manage_results.py

Module that defines manage results in a quantum circuit.

Dependencies:
- Uses the qiskit_ibm_runtime module to connect to the service.
- Uses analizes_results.py to get results.

Since:
- 11/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""
#do the necessary imports
from qiskit_ibm_runtime import QiskitRuntimeService
from analizes_results import *

#connects to the service
service = qiskit_ibm_runtime.QiskitRuntimeService()

with open("job_ids_4_statistics.txt", "w") as file:
    contain = file.read()

print(contain)

#gets results

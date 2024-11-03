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
import qiskit_ibm_runtime
from analizes_results import *

#connects to the service
service = qiskit_ibm_runtime.QiskitRuntimeService()

#reads "job_ids_4_statistics.txt" and creates a dictionary
data_dict = {}
with open("job_ids_4_statistics.txt", "r") as file:
    for line in file:
            line = line.rstrip()  #remove unnecessary \n
            if not line:
                continue  #ignore empty lines
            
            #count the indentation
            indentation = len(line) - len(line.lstrip())
            
            #remove left spaces
            content = line.lstrip()

            #if it's a new level
            if indentation == 0:
                current_level = data_dict
                current_level[content] = {}
                stack = [current_level[content]]
            else:
                #calculate the level of indentation
                level = indentation // 4
                
                #ajust the stack
                while len(stack) > level:
                    stack.pop()
                
                current_level = stack[-1]
                if content not in current_level:
                    current_level[content] = {}
                
                #add the new level to the stack
                stack.append(current_level[content])

#1 full adder statistics
for ibm_hardware in data_dict["1 full adder"]:
    print(ibm_hardware)
    for job_id in data_dict["1 full adder"][ibm_hardware]:
        print("    "+job_id)
        results = get_results(job_id=job_id, service=service, save_txt_with_results=True)
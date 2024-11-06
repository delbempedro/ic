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

#computes the results
#goes trough all the full adders
for num_of_full_adders in data_dict:

    #goes trough all the hardware
    for ibm_hardware in data_dict[num_of_full_adders]:

        number_of_jobs = 0
        number_of_shots = 0
        general_results = {}
        directory = num_of_full_adders+"/"+ibm_hardware+"/"

        #goes trough all the jobs
        for job_id in data_dict[num_of_full_adders][ibm_hardware]:

            #get the results of the job
            job_directory = directory+job_id+".txt"
            get_results(job_id=job_id, service=service, directory=job_directory)
            
            #analyzes the results
            with open(job_directory, "r") as current_file:

                #goes trough all the lines
                for line_index, line in enumerate(current_file):

                    if line_index > 1: #ignore the first line
                        key, value = line.split()

                        #add values to the general results
                        if key in general_results: #if the key is already in the dictionary
                            general_results[key] += int(value)
                        else: #if the key is not in the dictionary
                            general_results[key] = int(value)

                    elif line_index == 0: #count the number of shots
                        number_of_shots += int(line.split()[-1])

            #count the number of jobs
            number_of_jobs += 1

        if len(general_results): #only if there are results

            corret_value = "1"*len(list(general_results.keys())[0])
            per_cent_of_hits = (general_results[corret_value]/number_of_shots)*100.0
            current_directory = directory+"general_results"+".txt"
            more_frequency_result = max(general_results, key=general_results.get)

            #save the general results
            with open(current_directory, "w") as current_file:
                #print("I'm here")

                current_file.write("number of jobs: "+str(number_of_jobs)+"\n")
                current_file.write("percentage of hits: "+f"{per_cent_of_hits:.2f}"+"%"+"\n")
                current_file.write("more frequency result: "+str(more_frequency_result)+"\n")
                # goes trough all the keys
                for key, value in general_results.items():
                    current_file.write(key+" "+str(value)+"\n")
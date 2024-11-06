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


#reads the file and creates a dictionary
def read_list_of_jobs(jobs_file):
    """
    Reads the file and creates a dictionary.

    The file is expected to have the following structure:
    - The first level of indentation is for the number of full adders.
    - The second level of indentation is for the hardware.
    - The third level of indentation is for the job id.

    For example:
    1 full adder
        ibm_kyiv
            cwf8k440r6b0008p9a50
            cwf8k7w9r49g0085syag
        ibm_brisbrane
            cwkyf649r49g0089cep0
    2 full adder
        ibm_kyiv
            cwf538d543p00085zqk0
    
    Args:
        jobs_file: The file to be read.

    Returns:
        A dictionary with the following structure:
        - key: number of full adders
        - value: dictionary with key: hardware, value: list of job ids
    """
    data_dict = {}
    with open(jobs_file, "r") as file:
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

    return data_dict

#computes the results
def computes_results(data_dict):
    """
    Goes trough all the full adders and all the hardware and for each combination of them, 
    it goes trough all the jobs and get the results of each job. After that, it analyzes 
    the results and save them in a file.

    Parameters:
    data_dict (dict): A dictionary where the keys are the numbers of full adders and the 
    values are dictionaries where the keys are the hardware and the values are lists of 
    job ids.
    """
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

                    bit_a_bit_value = compute_bit_a_bit_value_most_frequent_result(general_results, number_of_shots)
                    current_file.write("bit a bit value most frequent result: "+bit_a_bit_value+"\n")
                    # goes trough all the keys
                    for key, value in general_results.items():
                                
                        current_file.write(key+" "+str(value)+"\n")

def compute_bit_a_bit_value_most_frequent_result(general_results, number_of_shots):
    """
    Computes the bit a bit value most frequent result.

    It takes a dictionary (general_results) and the total number of shots.
    It returns a string with the bit a bit value most frequent result.

    The algorithm works by initializing a list of weights with the same length of the keys in the dictionary.
    Then it goes through all the keys and for each key it goes through all the bits. For each bit it adds the value of the bit multiplied by the value of the key to the corresponding weight.
    Then it goes through all the weights and if the weight is greater than half of the total number of shots it adds "1" to the bit a bit value, otherwise it adds "0".
    Finally it returns the bit a bit value most frequent result.

    Parameters
    ----------
    general_results : dict
        A dictionary with the results.
    number_of_shots : int
        The total number of shots.

    Returns
    -------
    bit_a_bit_value : str
        A string with the bit a bit value most frequent result.
    """
    #initializes the weights
    weights = [0]*len(list(general_results.keys())[0])

    # goes trough all the keys computing the weights
    for key, value in general_results.items():

        index = 0
        for bit in key:

            weights[index] += int(bit)*value
            index += 1

    bit_a_bit_value = ""
    # goes trough all the weights computing the bit a bit value
    for weight in weights:
        if weight >= number_of_shots/2:
            bit_a_bit_value += "1"
        else:
            bit_a_bit_value += "0"

    return bit_a_bit_value

if __name__ == "__main__":
    
    #connects to the service
    service = qiskit_ibm_runtime.QiskitRuntimeService()
    
    #reads "job_ids_4_statistics.txt" and computes the results
    computes_results(read_list_of_jobs("job_ids_4_statistics.txt"))
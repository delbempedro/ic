"""
  analizes_results.py

Module that defines analizes results in a quantum circuit.

Dependencies:
- Uses pathlib module verify the existence of a directory.

Since:
- 11/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""
#do the necessary imports
from pathlib import Path

def get_results(job_id="", service="", directory=""):
    """
    Gets the results of a job if the file doesn't exist yet.

    Parameters:
    job_id (str): the id of the job to get the results from
    service (QiskitRuntimeService): the service that contains the job
    save_txt_with_results (bool): if True, the results will be saved in a txt file
    directory (str): the directory where the file will be saved

    Returns:
    The results of the job
    """
    if not Path(directory).exists() == False:

        results = service.job(job_id).result()

        #performs the necessary processing for measurements coming from the
        #IBM plataform instead of direct measurements made by qiskit code
        statistics = {}
        if type(results) == dict:
            contains = results["results"][0]['data']
            number_of_shots = results["metadata"]["execution"]["execution_spans"][0][-1]["0"][0][0]

            #gruops the bits of each measurements
            for measure in range(number_of_shots):
                current_measure = ""
                for qbit in contains:
                    bit = contains[qbit]["samples"][measure][-1]
                    current_measure = current_measure + bit

                if current_measure in statistics:
                    statistics[current_measure] += 1
                else:
                    statistics[current_measure] = 1

        else:
            statistics = results[0].data.c.get_counts()
            number_of_shots = results[0].data.c.num_shots

        more_frequency_result = max(statistics, key=statistics.get)
            
        #saves the results in a txt file if save_txt_with_results is True
        save_name = directory
        with open(save_name, "w") as file:
            file.write("number of shots: " + str(number_of_shots) + "\n")
            file.write("more frequency result: " + str(more_frequency_result) + "\n")
            for key in statistics:
                file.write(key + " " + str(statistics[key]) + "\n")
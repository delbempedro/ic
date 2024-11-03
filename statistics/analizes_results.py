"""
  analizes_results.py

Module that defines analizes results in a quantum circuit.

Dependencies:
-

Since:
- 11/2024

Authors:
- Pedro C. Delbem. <pedrodelbem@usp.br>
"""
def get_results(job_id="", service=service, save_txt_with_results=False, directory=""):
    """
    Gets the results of a job from the service.

    Parameters
    ----------
    job_id : str
        The id of the job to get the results from.
    service : QiskitRuntimeService
        The service to get the job from.
    save_txt_with_results : bool
        Whether to save the results to a text file.

    Returns
    -------
    dict
        The results of the job.
    """
    results = service.job(job_id).result()[0].data.c.get_counts()
    if save_txt_with_results:
        save_name = directory+"/"+str(job_id)+".txt"
        with open(save_name, "w") as file:
            file.write(str(results))
 
    return results
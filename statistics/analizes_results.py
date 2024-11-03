#installs necessary packages
%pip install matplotlib
%pip install qiskit_ibm_runtime

#imports necessary packages
import matplotlib.pyplot as plt
import qiskit_ibm_runtime

#connects to the service
service = qiskit_ibm_runtime.QiskitRuntimeService()

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
        with open(save_name, "w") as f:
            f.write(str(results))
 
    return results
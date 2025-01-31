#!/usr/bin/env python
# coding: utf-8

# # Some packages and methods

# In[218]:


#import pandas as pd 
import numpy as np 


# In[219]:


"""get_ipython().run_line_magic('pip', 'install pylatexenc')
get_ipython().run_line_magic('pip', 'install qiskit')
get_ipython().run_line_magic('pip', 'install qiskit_aer')
get_ipython().run_line_magic('pip', 'install qiskit_ibm_runtime')"""


# In[220]:


from qiskit_ibm_runtime import QiskitRuntimeService
my_token="d3751e47e134ce622b9fd7675f7e2d0364f516c548ad8e086798b2bfebd1dc9de61db3a22b1bcf836795247ec17ec1693a1196cf0d33087fd9b22862a86b1629"
service = QiskitRuntimeService.save_account(channel="ibm_quantum", token=my_token, overwrite=True) # AccountAlreadyExistsError: 'Named account (default-ibm-quantum) already exists. Set overwrite=True to overwrite.'


# In[221]:


service = QiskitRuntimeService()
# service.backends()


# In[ ]:


# Number of qubits required from the IBM quantum device
nqubits = 4


# In[ ]:


backend = service.least_busy(
    operational=True, min_num_qubits=nqubits, simulator=False
)
# backend


# In[ ]:


from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorSampler
from qiskit.result import QuasiDistribution


# # Neural Network training
#  - use the ``Sampler`` method from ``qiskit``

# # A circuit to produce all the inputs at once
# - generate a quasi-uniform distribution
#   -  from a circuit constructed by using the ``Two-Local`` method 
#   - all inputs generated at once

# In[237]:


# The TwoLocal generates a circuit that mixture two qubits - an entanglement circuit
# Angles pi and pi/2 aims at generating all the combinations of two qubits 
def qc_all_inputs():
  unset_2b_circ = TwoLocal(2, "rx", "cz", entanglement="linear", reps=1) #, parameter_prefix="theta") - it changes the name of the parameter variables of the TwoLocal
  parameter_values = [np.pi,np.pi,np.pi/2,np.pi/2] # previously chosen/learned parameters to generate all the possible inputs
  parameter_dict = dict(zip(unset_2b_circ.parameters, parameter_values))
  initial_states_2b_circ = unset_2b_circ.assign_parameters(parameter_dict) # a circuit that can generate all the possible inputs

  return initial_states_2b_circ


# # A circuit to copy the NN inputs
#  - save inputs for a input-output mapping

# In[259]:


def qc_qubitcopy(initial_states_circ, nqubits=2, qb0=0, qb1=1, qbcp0=2, qbcp1=3): # qb0, qb1, qbcp0,cpcp1 are the indices of qubits to "copy"/reproduce and those copied 
  print(initial_states_circ) 
  qcp = QuantumCircuit(nqubits+2) # 2*nqubits -- double of the number of qubits of the neuron network inputs
  qcp = qcp.compose(initial_states_circ, qubits=list(range(0,nqubits))) # first half with the initial_states_2b_circ
  qcp.barrier() # to visually separate circuit components
  # A layer to copy/reproduce the generated inputs in qubits 0 and 1
  qcp.x(qbcp0)    # change value of qubit 2 from 0 to 1
  qcp.x(qbcp1)    # change value of qubit 3 from 0 to 1
  qcp.cx(qb0,qbcp0) # qb0 ''AND'' 1 (or NOT qb0) to copy qubit 0 to qubit 2
  qcp.cx(qb1,qbcp1) # qb1 ''AND'' 1 (or NOT qb1) to copy qubit 1 to qubit 3
  qcp.x(qbcp0)    # NOT of qubit 2 => qubit 2 equal to equal qubit 0
  qcp.x(qbcp1)    # NOT of qubit 3 => qubit 3 equal to equal qubit 1

  return qcp


# # ``XOR`` candidate circuits

# In[264]:


def qc4xor(all_inputs_circ, entanglement_circ_param): 
  qxor = all_inputs_circ.copy()
  qxor.barrier() # to visually separate circuit components
  # Angles used to learn - the search space exploration, aiming to configure a XOR neural network
  # Set the parameters of U gate (theta, phi, and lambda), for rotations on qubit 0
  qxor.u(*entanglement_circ_param, 0) # it has the degrees of freedom

  return qxor


# In[265]:


# Another circuit to work as XOR - from Figure 3 of the paper of Tacchino et al., 2019:
# "An artificial neuron implemented on an actual quantum processor"
# Under construction ...
def qc4xor_another(all_inputs_circ, qb0, qb1): #qb0, qb1 are the qubits for XORing 
  qxor = all_inputs_circ.copy()
  qxor.barrier() # to visually separate circuit components
  qxor.z(qb0)
  qxor.z(qb1)
  qxor.cz(qb0,qb1)
  qxor.cx(qb0,qb1)
  qxor.cx(qb1,qb0)
  # Hadamard gate not include since it is used to revert the Hardmard used in the entanglment layer Ui

  return qxor


# In[266]:


def qc4xor_another_2(all_inputs_circ, entanglement_circ_param, qb0, qb1): #qb0, qb1 are the qubits for XORing 
  qxor = all_inputs_circ.copy()
  qxor.barrier() # to visually separate circuit components
  gamma = 0
  qxor.cu(*entanglement_circ_param, gamma, qb1, qb0) #phi, lambda, gamma <- entanglement_circ_param
  
  return qxor


# In[267]:


def qc4xor_another_3(all_inputs_circ, gamma, qb0, qb1): #qb0, qb1 are the qubits for XORing 
  qxor = all_inputs_circ.copy()
  qxor.barrier() # to visually separate circuit components
  qxor.cp(gamma, qb1, qb0) #phi, lambda, gamma <- entanglement_circ_param
  
  return qxor


# # Mounting a circuit to train a quantum NN as ``XOR``

# In[243]:


# visualize
qc_all_inputs = qc_all_inputs()
"""qc_all_inputs.decompose().draw('mpl')"""


# In[244]:


nqubits = len(qc_all_inputs.qubits)


# In[260]:


reversable_circ = qc_qubitcopy(qc_all_inputs, nqubits = len(qc_all_inputs.qubits), qb0=0, qb1=1, qbcp0=2, qbcp1=3) # qb0, qb1, qbcp0,cpcp1 are the indices of qubits to "copy"/reproduce and those copied


# In[261]:


"""reversable_circ.decompose().draw('mpl')"""


# In[268]:


# include a qxor candidate circuit 
# for setting up parameter values - a searching space exploration - a machine learning

#gamma = 0 # parameter value
#a_xor_candidate = qc4xor_another_3(reversable_circ, gamma, 0, 1)

entanglement_circ_param = 1 * np.pi * np.random.rand(3) #the theta, phi, and lambda parameters of U Gate
a_xor_candidate = qc4xor_another_2(reversable_circ, entanglement_circ_param, 0, 1)


# In[269]:


"""a_xor_candidate.decompose().draw('mpl')"""


# # First - check the initial state distribution
# - from the reversable_circuit
#  - initial states with a copy of the generate qubits to check

# In[270]:


def evaluate_qc(qc, shots = 100, nruns=100):

  #sample results with severals runs, each with several shots
  sampler = StatevectorSampler()
  jobs = []
  for arun in range(0,nruns):
    job = sampler.run([(qc)], shots = shots)
    jobs.append(job)

  #get and show raw results - counts
  counts = []
  for job in jobs:
    data_pub = job.result()[0].data # 'pub' refers to Primitive Unified Bloc
    job_counts = data_pub.meas.get_counts()
    counts.append(job_counts)
  #print(counts)

  return counts


# In[271]:


reversable_circ_meas = reversable_circ.copy()


# In[272]:


reversable_circ_meas.measure_all()


# In[273]:


"""reversable_circ_meas.decompose().draw('mpl')"""


# In[274]:


counts_rev_circ_meas = evaluate_qc(reversable_circ_meas, shots = 100, nruns=100)#, shots = 10, nruns=10)


# In[275]:


counts = counts_rev_circ_meas


# In[276]:


statistics = dict()
for count in counts:
  for k,v in count.items():
    statistics = {k: statistics.get(k, 0) + v for k, v in count.items()}
total_tests = sum([ v for v in statistics.values()])
frequencies = {state: count / total_tests for state, count in statistics.items()}


# In[277]:


frequencies


# # Second - check distributions after
#  - inputs 
#  - after entangelment
# 
# using the reversable_circuit with 2 copying layer

# In[278]:


a_xor_candidate_meas = a_xor_candidate.copy() 


# In[279]:


a_xor_candidate_meas.measure_all() 


# In[280]:


a_xor_candidate_meas.decompose().draw('mpl') 


# In[281]:


counts_a_xor_candidate_meas = evaluate_qc(a_xor_candidate_meas, shots = 100, nruns=100)#, shots = 10, nruns=10)


# In[282]:


counts = counts_a_xor_candidate_meas


# In[283]:


#counts


# In[284]:


statistics = dict()
for count in counts:
  for k,v in count.items():
    statistics = {k: statistics.get(k, 0) + v for k, v in count.items()}
total_tests = sum([ v for v in statistics.values()])
frequencies = {state: count / total_tests for state, count in statistics.items()}


# In[285]:


frequencies


# # Run and measure from a quantum circuit 
# - ``XOR`` in the current case

# ## Statistics from the quantum circuit results

# In[205]:


# details and examples of such a code in notebook aqNN-v0.ipynb
def sq_error(counts, expected_density_matrix): # counts - with the raw results after run the quantum circuit of a XOR candidate
  #frequencies of the outputs from the raw data collected
  statistics = dict()
  for count in counts:
    for k,v in count.items():
      statistics = {k: statistics.get(k, 0) + v for k, v in count.items()}
  total_tests = sum([ v for v in statistics.values()])
  frequencies = {state: count / total_tests for state, count in statistics.items()}

  # a density matrix of inputs from all the raw data collected (quantum circuit outputs)
  a_ds_matrix = np.zeros((4, 2)) 
  for state, freq in frequencies.items(): 
    print(state)
    inputs = int(state[2:4], 2)  # Convert binary string to integer index 
    output = int(state[0:1], 2) 
    a_ds_matrix[inputs, output] = freq 

  #for m, d in zip(a_ds_matrix, expected_density_matrix): 
  #    print(m,d,m - d)
  ##for m, d in zip(a_ds_matrix[1], expected_density_matrix[1]): 
  ##    print(m,d,m - d)
  squared_error = abs(a_ds_matrix[0][1]-expected_density_matrix[0][1]) +       abs(a_ds_matrix[1][1]-expected_density_matrix[1][1]) +       abs(a_ds_matrix[2][1]-expected_density_matrix[2][1]) +       abs(a_ds_matrix[3][1]-expected_density_matrix[3][1])
  # squared error from 
  # squared_error = np.sum(np.abs(a_ds_matrix - expected_density_matrix)) / 8
  #squared_error = np.sum(np.abs(a_ds_matrix[1] - expected_density_matrix[1])) / 4

  return squared_error 


# In[90]:


# Density matrix of the expected outputs
ds_expected_outputs = np.zeros((4, 2))
ds_expected_outputs[0][0] = 0.25
ds_expected_outputs[1][1] = 0.25
ds_expected_outputs[2][1] = 0.25
ds_expected_outputs[3][0] = 0.25


# In[286]:


#df = pd.DataFrame(ds_expected_outputs, ['00', '01', '10', '11'],  ['0', '1'])
#df


# ## Calculate the accuracy of the ``XOR`` candidate

# In[206]:


sq_error(counts, ds_expected_outputs)


# # Optimize circuit parameters to improve accuracy

# In[171]:


# 2 * np.pi --or-- 1 * np.pi 
entanglement_circ_param = 1 * np.pi * np.random.rand(3) #the theta, phi, and lambda parameters of U Gate
entanglement_circ_param


# ## A naïve optimization algorithm 

# In[172]:


class result:
    def __init__(self, x, fun):
        self.x= x
        self.fun = fun

def exaustive_grid_search(objective_function, initial_coeffs, grid_grain=10):
  # initial_coeffs are fake for the current algorithm
  logs = dict()
  errors = []
  for theta in np.linspace(0, 1 * np.pi, grid_grain):
    for phi in np.linspace(0, 1 * np.pi, grid_grain):
      for lam in np.linspace(0, 1 * np.pi, grid_grain):
        initial_coeffs = [theta, phi, lam]
        sqe = objective_function(initial_coeffs)
        logs[str(sqe)] = {'theta': theta, 'phi': phi, 'lambda': lam, 'error': sqe}
        errors.append(sqe)
        print(initial_coeffs, logs[str(sqe)], sqe)
  min_error = min(errors)
  output = logs[str(min_error)]
  result.x = list(output.values())[:-1]
  result.fun = min_error
    
  return result


# In[173]:


class result:
    def __init__(self, x, fun):
        self.x= x
        self.fun = fun

def exaustive_grid_search_1d(objective_function, initial_coeffs, grid_grain=10): 
  # initial_coeffs are fake for the current algorithm
  logs = dict() 
  errors = [] 
  for gamma in np.linspace(0, 1 * np.pi, grid_grain): 
    initial_coeffs = [gamma] 
    sqe = objective_function(initial_coeffs) 
    logs[str(sqe)] = {'gamma': gamma, 'error': sqe} 
    errors.append(sqe) 
    print(initial_coeffs, logs[str(sqe)], sqe) 
  min_error = min(errors) 
  output = logs[str(min_error)] 
  result.x = list(output.values())[:-1] 
  result.fun = min_error 

  return result 


# In[189]:


#initial_coeffs <=> entanglement_circ_param 
#def evaluate_qc4xor_another_3(initial_coeffs, qc_all_inputs=all_inputs_reversable_circ_2, expected_density_matrix=ds_expected_outputs):
def evaluate_qc4xor_another_2(initial_coeffs, qc_all_inputs=reversable_circ, expected_density_matrix=ds_expected_outputs):
  #prepare qc with parameters of the XOR candidate set to initial_coeffs: entanglement_circ_param values 
  a_xor_candidate = qc4xor_another_2(qc_all_inputs, initial_coeffs, 0, 1) 
  #a_xor_candidate = qc4xor_another_3(qc_all_inputs, initial_coeffs[0], 0, 1)
  a_xor_candidate.measure_all() 
  #run and sample circuit final states 
  counts = evaluate_qc(a_xor_candidate, shots = 10, nruns=10) 
  #return the squared error of the found density matrix compared with the expected density matrix 
  return sq_error(counts, expected_density_matrix) 


# In[112]:


#qc_reversable.decompose().draw()


# In[113]:


from scipy.optimize import minimize


# In[207]:


##### Optimize the coefficients 
initial_coeffs = 1 * np.pi * np.random.rand(3) # entanglement_circ_param = the theta, phi, and lambda parameters of U Gate
#objective_function = evaluate_a_model#(qc_model=qc4xor, qc_all_inputs=qc_reversable, qc_model_parameters=initial_coeffs, expected_density_matrix=ds_expected_outputs)
#objective_function = evaluate_qc4xor
#objective_function = evaluate_qc4xor_another_2
#initial_coeffs = 1 * np.pi * np.random.rand(1)
objective_function = evaluate_qc4xor_another_2
grid_grain=10
result = exaustive_grid_search(objective_function, initial_coeffs, grid_grain=grid_grain) #40 #30 #10)
#result = exaustive_grid_search_1d(objective_function, initial_coeffs, grid_grain=grid_grain) #40 #30 #10)


# Results
optimized_coeffs = result.x
optimized_value = result.fun

# Print results
print("Optimized Coefficients:", optimized_coeffs)
print("Optimized Expectation Value:", optimized_value)
print("Used grid_grain:", grid_grain)


# In[ ]:


# checking the best found circuit


# In[208]:


print(optimized_coeffs)


# In[209]:


#handy_trials = [0] #[np.pi/3] #[np.pi, 0, np.pi] #[0, 0, 0] #[np.pi, np.pi/1, 0]


# In[210]:


#a_xor_candidate = qc4xor_another_3(reversable_circ, handy_trials[0], 0, 1) 
#a_xor_candidate = qc4xor_another_3(reversable_circ, optimized_coeffs[0], 0, 1) 
a_xor_candidate = qc4xor_another_2(reversable_circ, optimized_coeffs, 0, 1) 
a_xor_candidate.measure_all() 
a_xor_candidate.decompose().draw('mpl') 


# In[217]:


counts = evaluate_qc(a_xor_candidate, shots = 100, nruns=100) #, shots = 10, nruns=10)
sq_error(counts, ds_expected_outputs)


# In[213]:


#frequencies of the outputs from the raw data collected
statistics = dict()
for count in counts:
  for k,v in count.items():
    statistics = {k: statistics.get(k, 0) + v for k, v in count.items()}
total_tests = sum([ v for v in statistics.values()])
frequencies = {state: count / total_tests for state, count in statistics.items()}

# a density matrix of inputs by outputs from the raw data collected
a_ds_matrix = np.zeros((4, 2))
for state, freq in frequencies.items():
  print(state, freq)
  inputs = int(state[2:4], 2)  # Convert binary string to integer index
  output = int(state[0:1], 2)
  #print(state, inputs, output)
  a_ds_matrix[inputs, output] = freq

a_ds_matrix


# In[196]:


df = pd.DataFrame(a_ds_matrix, ['00', '01', '10', '11'],  ['0', '1'])
df


# In[ ]:





# In[ ]:





# In[ ]:





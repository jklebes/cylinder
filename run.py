import cmath
import pickle
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math
import collections
import timeit
import seaborn as sb
import argparse
import metropolisengine
me_version = metropolisengine.__version__ #save version number
import system2D as ce

def loop_num_field_coeffs(num_field_coeff_range, fieldstep_range, n_steps, method = "sequential", outdir = None):
  """
  A set of runs over different number field coeffs
  choose and interesting position in pararmeter space
  time it
  :param wavenumber_range:
  :param kappa_range:
  :return:
  """
  global num_field_coeffs # n as in c_-n to c_n - number of c variables is 2n+1
  global fieldsteps_per_ampstep
  results = collections.defaultdict(list)
  for num in num_field_coeff_range:
    num_field_coeffs = num
    results_line = collections.defaultdict(list)
    for fieldsteps in fieldstep_range:
      fieldsteps_per_ampstep = fieldsteps
      start_time = timeit.default_timer()
      names, means, cov_matrix = single_run(n_steps=n_steps, method=method, outdir = outdir, title = "ncoeffs"+str(num_field_coeffs)+"_fsteps"+str(fieldsteps_per_ampstep))
      time = timeit.default_timer() - start_time
      means_dict = dict([(name, mean) for (name,mean) in zip (names, means)])
      print(cov_matrix)
      coeffs_names = names[1:(1+2*num_field_coeffs[0])*(1+2*num_field_coeffs[1])]
      var_dict = dict([(name, cov_matrix[i][i]) for (i, name) in enumerate(coeffs_names)])
      covar_dict = dict([(name1+"_"+name2, cov_matrix[i][j]) for (i, name1) in enumerate(coeffs_names) for (j, name2) in enumerate(coeffs_names) if i!= j])
      print(means_dict)
      for name in names:
        results_line[name+"_mean"].append(means_dict[name])
      results_line["time_per_experiment"].append(time)
      for name in var_dict:
        results_line[name+"_variance"].append(var_dict[name])
      for name in covar_dict:
        results_line[name+"_covariance"].append(covar_dict[name])
    for name in results_line:
      results[name].append(results_line[name])
  return results 

def loop_amplitude_C(amplitude_range, C_range, n_steps, method = "fixed-amplitude", outdir =None):
  """
  A set of runs looping over a grid of wavenumber, bending rigdity values
  :param wavenumber_range:
  :param kappa_range:
  :return:
  """
  global initial_amplitude
  global C
  results = collections.defaultdict(list)
  for a in amplitude_range:
    initial_amplitude = a
    results_line = collections.defaultdict(list)
    for c in C_range:
      C=c
      names, means, cov_matrix = single_run(n_steps=n_steps, method=method, outdir = outdir,title = "amplitude"+str(a)+"_C"+str(C))
      means_dict = dict([(name, mean) for (name,mean) in zip (names, means)])
      var_dict = dict([(name, cov_matrix[i][i]) for (i, name) in enumerate(names[:2+2*num_field_coeffs])])
      covar_dict = dict([(name1+"_"+name2, cov_matrix[i][j]) for (i, name1) in enumerate(names[:2+2*num_field_coeffs]) for (j, name2) in enumerate(names[:2+2*num_field_coeffs]) if i!= j])
      for name in names:
        results_line[name+"_mean"].append(means_dict[name])
      for name in var_dict:
        results_line[name+"_variance"].append(var_dict[name])
      for name in covar_dict:
        results_line[name+"_covariance"].append(covar_dict[name])
    for name in results_line:
      results[name].append(results_line[name])
  return results 

def loop_wavenumber_kappa(wavenumber_range, kappa_range, n_steps, method = "simultaneous", outdir=None):
  """
  A set of runs looping over a grid of wavenumber, bending rigdity values
  :param wavenumber_range:
  :param kappa_range:
  :return:
  """
  global wavenumber
  global kappa
  results = collections.defaultdict(list)
  for wvn in wavenumber_range:
    wavenumber = wvn
    results_line = collections.defaultdict(list)
    for kb in kappa_range:
      # run
      kappa=kb
      means_dict, cov_matrix = single_run(n_steps=n_steps, method=method, outdir = outdir, title = "wvn"+str(round(wvn,3))+"_kappa"+str(round(kappa,3)))
      coeffs_names = ["param_"+str(i) for i in range(1,(1+2*num_field_coeffs[0])*(1+2*num_field_coeffs[1]))]
      var_dict = dict([(name, cov_matrix[i][i]) for (i, name) in enumerate(coeffs_names)])
      covar_dict = dict([(name1+"_"+name2, cov_matrix[i][j]) for (i, name1) in enumerate(coeffs_names) for (j, name2) in enumerate(coeffs_names) if i!= j])
      for name in means_dict:
        results_line[name].append(means_dict[name])
      for name in var_dict:
        results_line[name+"_variance"].append(var_dict[name])
      for name in covar_dict:
        results_line[name+"_covariance"].append(covar_dict[name])
    for name in results_line:
      results[name].append(results_line[name])
  return results 


def loop_wavenumber_alpha(wavenumber_range, alpha_range, n_steps, method = "simultaneous", outdir=None):
  """
  A set of runs looping over a grid of wavenumber, alpha values
  :param wavenumber_range:
  :param alpha_range:
  :return:
  """
  global wavenumber
  global alpha
  results = collections.defaultdict(list)
  for wvn in wavenumber_range:
    wavenumber = wvn
    results_line = collections.defaultdict(list)
    for alpha_ in alpha_range:
      # run
      alpha = alpha_

      assert(alpha <=0)
      means_dict, cov_matrix = single_run(n_steps=n_steps, method=method, outdir = outdir, title = "wvn"+str(wvn)+"_alpha"+str(alpha))
      print("means_dict", means_dict)
      var_dict = dict([(name, cov_matrix[i][i]) for (i, name) in enumerate(names[:2+2*num_field_coeffs])])
      print(kappa, wavenumber, means_dict["abs_amplitude"])
      covar_dict = dict([(name1+"_"+name2, cov_matrix[i][j]) for (i, name1) in enumerate(names[:2+2*num_field_coeffs]) for (j, name2) in enumerate(names[:2+2*num_field_coeffs]) if i!= j])
      for name in names:
        results_line[name].append(means_dict[name])
      for name in var_dict:
        results_line[name+"_variance"].append(var_dict[name])
      for name in covar_dict:
        results_line[name+"_covariance"].append(covar_dict[name])
    for name in results_line:
      results[name].append(results_line[name])
  return results

def plot_save(range1, range2, results, title, exp_dir = '.'):
  """
  Save data in a CSV file and generate plots
  :param wavenumber_range:
  :param kappa_range:
  :param results:
  :param title:
  :return:
  """
  print(results)
  #cut ranges to shape of data
  range1 = range1[-len(results):]
  range2 = range2[-len(results[0]):]
  df = pd.DataFrame(index=range1, columns=range2, data=results)
  df.to_csv(os.path.join(exp_dir, title + ".csv"))
  if not isinstance(df.loc[range1[0],range2[0]], complex):
    # plot only mean parameters that are real values, such as mean abs(somethign)
    print("plotting png")
    sb.heatmap(df, cmap = "viridis")
    plt.savefig(os.path.join(exp_dir, title + ".png"))
    plt.close()


def run_experiment(exp_type,  range1, range2, n_steps, method):
  """
  Launches a set of runs exploring the stability on a grid of 2 parameters.
  :param type:
  :param experiment_title:
  :param range1:
  :param range2:
  :return
  """
  #make directory for the experiment
  now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  exp_dir= os.path.join("out", "exp-"+now)
  os.mkdir(exp_dir)
  #save variables about how the experiment was run
  exp_notes = {"experiment type": " ".join(exp_type), "n_steps": n_steps, "temp":temp, "method": method, "C": C,"kappa": kappa,  "alpha": alpha, "n":n, "u":u, "num_field_coeffs":num_field_coeffs, "range1":range1, "range2": range2, "radius":radius, "amplitude":initial_amplitude, "wavenumber": wavenumber, "measure every n ampsteps":measure_every, "total number ampsteps": n_steps*measure_every, "notes":notes, "start_time": now, "me-version": me_version}
  if method == "sequential":
    exp_notes["fieldsteps per ampstep"] = fieldsteps_per_ampstep
  exp_notes = pd.DataFrame.from_dict(exp_notes, orient="index", columns=["value"])
  exp_notes.to_csv(os.path.join(exp_dir, "notes.csv"))
  # run experiment of the requested type
  # TODO: could be switch or dict of functions
  if exp_type == ("wavenumber", "kappa"):
    results = loop_wavenumber_kappa(wavenumber_range=range1, kappa_range=range2, n_steps=n_steps, method=method, outdir = exp_dir)
    print("results", results)
  elif exp_type == ("wavenumber", "alpha"):
    results = loop_wavenumber_alpha(wavenumber_range=range1, alpha_range=range2, n_steps=n_steps, method=method, outdir = exp_dir)
    print(results["abs_amplitude"])
  elif exp_type == ("amplitude", "C"):
    results = loop_amplitude_C(amplitude_range=range1, C_range=range2, n_steps=n_steps, method=method, outdir = exp_dir)
  elif exp_type == ("num_field_coeffs", "fieldsteps_per_ampstep"):
    results = loop_num_field_coeffs(num_field_coeff_range=range1, fieldstep_range=range2, n_steps=n_steps, method=method, outdir = exp_dir)
    #save results spreadsheets and plots - mainly mean abs(amplitude) and its variance
    print(results)
  else:
    print("experiment type not found: ", exp_type)
  for name in results:
    plot_save(range1=range1, range2=range2, results=results[name], title=exp_type[0]+ "_"+name+"_", exp_dir=exp_dir)


def single_run(n_steps, method = "simultaneous", field_coeffs=None, amplitude=None, outdir = None, title = None):
  """
  for examining a single run over time.
  with all the data recording
  :param kappa:
  :param wavenumber:
  :param amp_steps:
  :param fieldsteps_per_ampstep:
  :return:
  """
  ########### initial values ##############
  # read from files or generate
  if field_coeffs is None:
    field_coeffs = np.array(list(map(lambda x: metropolisengine.MetropolisEngine.gaussian_complex(),range(0,(2*num_field_coeffs[0]+1)*(2*num_field_coeffs[1]+1)))))
    print("initialized random complex coeffs", field_coeffs)
  if amplitude is None:
    global initial_amplitude
    amplitude = initial_amplitude #take from global
  if os.path.isfile("./last_sigma_2.pickle") and os.path.getsize("./last_sigma_2.pickle"):
    f = open('last_sigma_2.pickle', 'rb')
    sampling_width = pickle.load(f)
  else:
    sampling_width = .05
  if os.path.isfile("./last_cov.pickle") and os.path.getsize("./last_cov.pickle"):
    f = open('last_cov.pickle', 'rb')
    cov = pickle.load(f)
    if len(cov) != (2*num_field_coeffs[0]+1)*(2*num_field_coeffs[1]+1):
      print("rejected because dimensions are wrong")
      cov=None
  else:
    cov=None

  ########### setup system, metropolis engine, link energy functions #############
  se = ce.System2D(wavenumber=wavenumber, radius=radius, alpha=alpha, C=C, u=u, n=n, kappa=kappa, gamma=gamma, num_field_coeffs= num_field_coeffs)
  #function [real values], [complex values] -> energy 

  #np.reshape unflattens coeffs c_{j beta} in format [c_{-n -m} ... c{n -m } ... c{-n 0} ... c_{n 0} ... c{-n m } ... c_{n m} ]
  # to matrix format  [[c_{-n -m} ... c{n -m }] 
  #                      ... 
  #                    [c{-n 0} ... c_{n 0} ]
  #                      ...
  #                    [c{-n m } ... c_{n m}]]
  z_array_len= num_field_coeffs[0]*2+1
  theta_array_len= num_field_coeffs[1]*2+1
  energy_fct_field_term = lambda real_params, complex_params: se.calc_field_energy(np.reshape(complex_params, (theta_array_len, z_array_len)))
  energy_fct_surface_term = lambda real_params, complex_params : se.calc_surface_energy(*real_params) #also need se.calc_field_energy_ampltiude_change to be saved to energy_dict "surface" slot 
  energy_fct_field_term_alt = lambda real_params, complex_params: se.calc_field_energy_amplitude_change(*real_params,np.reshape(complex_params, (theta_array_len, z_array_len)))
  energy_fct_by_params_group = {"complex": {"field": energy_fct_field_term}, "real": {"field": energy_fct_field_term_alt, "surface": energy_fct_surface_term}, "all":{"field": energy_fct_field_term_alt, "surface":energy_fct_surface_term}}
  me = metropolisengine.MetropolisEngine(energy_functions = energy_fct_by_params_group,  initial_complex_params=field_coeffs, initial_real_params = [float(amplitude)], covariance_matrix_complex=cov, sampling_width=sampling_width, temp=temp)
  #also input system constraint : steps with |amplitude| > 1 to be rejected
  me.set_reject_condition(lambda real_params, complex_params : abs(real_params[0])>=1 )  
  se.save_temporary_matrices()

 
  ########### start of data collection ############
  states=[]
  other_states = []
  if method == "sequential":
    for i in range(n_steps):
      for j in range(measure_every):
        accepted = me.step_real_group() #use outputted flag to trigger numerical integration in System on amplitude change
        #print("step real group", accepted, me.real_params)        
        if accepted: se.save_temporary_matrices()
        for ii in range(fieldsteps_per_ampstep):
          me.step_complex_group() # no need to save and look at "accept" flag when only field coeffs change
          #print("field_coeffs", me.complex_params)
      print("measure", i , "sampling widths", me.real_group_sampling_width, me.complex_group_sampling_width, me.real_params,me.complex_params)# "cov", me.covariance_matrix[0,0], me.covariance_matrix[1,1])
      me.measure() #update mean, covariance matrix, other parameters' mean by sampling this step
  elif method == "simultaneous":
    for i in range(n_steps):
      for j in range(measure_every):
        accepted = me.step_all()
        if accepted: se.save_temporary_matrices()
      #print("measure", i, "sampling widths", me.sampling_width, "cov")#, me.covariance_matrix[0,0], me.covariance_matrix[1,1])
      #print("step counters", me.measure_step_counter, me.field_step_counter, me.amplitude_step_counter)
      me.measure() #update mean, covariance matrix, other parameters' mean by sampling this step
      # TODO save time series data in metropolis enigne on calling measure()
  elif method == "fixed-amplitude":
    me.m -= 1
    for i in range(n_steps):
      for j in range(measure_every):
        for ii in range(fieldsteps_per_ampstep):
          me.step_complex_group()
      me.measure() #update mean, covariance matrix, other parameters' mean by sampling this step
  elif method == "no-field":
    me.m = 1 #number of dimensions of parameter space for adaptive purposes such as target acceptance rate.  actually just 1 degree of freedom for amplitude-only run.
    for i in range(n_steps):
      for j in range(measure_every):
        accepted = me.step_real_group()
        if accepted: se.save_temporary_matrices
      me.measure() #update mean, covariance matrix, other parameters' mean by sampling this step

  if outdir is not None and os.path.isdir(outdir):
    me.save_time_series()
    df = me.df #pd.DataFrame()
    #print(outdir, title, me.params_names, states)
    df.to_csv(os.path.join(outdir, title + ".csv"))
  #dump in files for order of magnitude estimate to start next simulation from
  f = open('last_cov.pickle', 'wb')
  pickle.dump(me.covariance_matrix_complex, f)
  print("cov", np.round(me.covariance_matrix_complex,3))
  f = open('last_sigma_2.pickle', 'wb')
  if method == "sequential":
    pickle.dump([me.real_group_sampling_width, me.complex_group_sampling_width], f)
  else:
    pickle.dump(me.real_group_sampling_width, f)
  result_names = me.params_names
  result_names.extend(me.observables_names)
  # the old version: raw means of whole simulation
  #result_means = [i for i in me.real_mean]
  #result_means.extend([i for i in me.complex_mean])
  #result_means.extend([i for i in me.observables_mean])
  # the new version: means of equilibrated part of simulation
  # prompt metropolisengine to collect stats
  me.save_equilibrium_stats()
  result_means = me.equilibrated_means #should list real param means, complex param means, observables means
                                       # maybe also: save g, neff => quality of sampling, error bars
  print("result_means,", result_means)
  return result_means , me.covariance_matrix_complex

# coefficients
alpha = -1
C = 1
u = 1
n = 1
kappa = .1
gamma = 1
temp = 1

# system dimensions
initial_amplitude= 0  #also fixed system amplitude for when amplitude is static
radius = 1
wavenumber = .4

# simulation details
num_field_coeffs = (0,0) # z-direction modes indices go from -.. to +..; theta direction indices go from -.. to +..
initial_sampling_width = .025
measure_every =10
fieldsteps_per_ampstep = 1  #nly relevant for sequential

#notes = "with n = 6 - expect more of field conforming to shape.  On fixed shape a=.8." #describe motivation for a simulation here!

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='get description')
  parser.add_argument('-n', '--notes', dest = 'notes',  type=str, nargs=1,
                    help='add experiment description with --notes', required=True)
  notes=parser.parse_args().notes

  # specify type, range of plot; title of experiment
  loop_type = ("wavenumber", "kappa")
  range1 = np.arange(0.05, 1.4, .7)
  range2 = np.arange(0, .5, .3)
  n_steps = 2000#n measuring steps- so there are n_steps * measure_every amplitude steps and n_steps*measure_every*fieldsteps_per_ampsteps fieldsteps
  method = "sequential"

  #single_run(kappa=kappa, wavenumber=wavenumber, n_steps=n_steps, method="no-field")

  run_experiment(loop_type,  range1, range2, n_steps, method)

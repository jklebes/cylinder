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
me_version = "0.2.19"#metropolisengine.__version__ #save version number
import system_cylinder2D as cylinder2d
import system_cylinder as cylinder


def loop(single_run_lambda, range1, range2):
  """
  generic loop: runs a 2D grid of single simulations; 
  information about which two simulation parameters to vary is built into argument single_run_lambda()
  
  :param single_run_lambda: the function to launch a single simulation, taking two arguments
  :type single_run_lambda: function var1 var2 -> simulation output
  :param range1: iterable of values that the function can take as var1
  :param range2: iterable of values that the function can take as var2
  :returns results: dict of 2D arrays
  :rtype: dict 
  """
  results = collections.defaultdict(list)
  for var1 in range1:
    results_line = collections.defaultdict(list)
    for var2 in range2:
      start_time = timeit.default_timer()
      param_names, observables_names, means_dict, cov_matrix = single_run_lambda(var1, var2) 
      time = timeit.default_timer() - start_time
      #print(cov_matrix)
      #coeffs_names = ["param_"+str(i) for i in range(1,(1+2*num_field_coeffs[0])*(1+2*num_field_coeffs[1]))]
      coeffs_names= param_names[1:]
      var_dict = dict([(name, cov_matrix[i][i]) for (i, name) in enumerate(coeffs_names)])
      covar_dict = dict([(name1+"_"+name2, cov_matrix[i][j]) for (i, name1) in enumerate(coeffs_names) for (j, name2) in enumerate(coeffs_names) if i!= j])
      #print(means_dict)
      for name in means_dict:
        results_line[name+"_mean"].append(means_dict[name])
      results_line["time_per_experiment"].append(time)
      for name in var_dict:
        results_line[name+"_variance"].append(var_dict[name])
      for name in covar_dict:
        results_line[name+"_covariance"].append(covar_dict[name])
    for name in results_line:
      results[name].append(results_line[name])
  return results 

functions_dict = {("num_field_coeffs", "fieldsteps_per_ampstep"): lambda num, fsteps: single_run(n_steps=n_steps, method=method, outdir = outdir, title = ("ncoeffs"+str(num)+"_fsteps"+str(fsteps)),
                                          num_field_coeffs=num, fieldsteps_per_ampstep = fsteps, 
                                          alpha=alpha, C=C, n=n, u=u, gamma=gamma, kappa=kappa, radius=radius,
                                          wavenumber=wavenumber, intrinsic_curvature = intrinsic_curvature),
                  ("amplitude", "C"): lambda var_a, var_C: single_run(n_steps=n_steps, method=method, outdir = outdir, title = ("a"+str(round(var_a,2))+"_C"+str(round(var_C,2))),
                                          num_field_coeffs=num_field_coeffs, fieldsteps_per_ampstep = fieldsteps_per_ampstep, 
                                          alpha=alpha, C=var_C, n=n, u=u, gamma=gamma, kappa=kappa, radius=radius,
                                          wavenumber=wavenumber, intrinsic_curvature = intrinsic_curvature, amplitude=var_a),
                  ("wavenumber", "kappa"): lambda var_wvn, var_kappa: single_run(n_steps=n_steps, method=method, outdir = outdir, title = ("wvn"+str(round(var_wvn,2))+"_kappa"+str(round(var_kappa,2))),
                                          num_field_coeffs=num_field_coeffs, fieldsteps_per_ampstep = fieldsteps_per_ampstep, 
                                          alpha=alpha, C=C, n=n, u=u, gamma=gamma, kappa=var_kappa, radius=radius,
                                          wavenumber=var_wvn, intrinsic_curvature = intrinsic_curvature),
                  ("wavenumber", "alpha"): lambda var_wvn, var_alpha: single_run(n_steps=n_steps, method=method, outdir = outdir, title = ("wvn"+str(round(var_wvn,2))+"_alpha"+str(round(var_alpha,2))),
                                          num_field_coeffs=num_field_coeffs, fieldsteps_per_ampstep = fieldsteps_per_ampstep, 
                                          alpha=var_alpha, C=C, n=n, u=u, gamma=gamma, kappa=kappa, radius=radius,
                                          wavenumber=var_wvn, intrinsic_curvature = intrinsic_curvature),                   
                  ("wavenumber", "intrinsic_curvature"): lambda var_wvn, var_H0: single_run(n_steps=n_steps, method=method, outdir = outdir, title = ("wvn"+str(round(var_wvn,2))+"_H0"+str(round(var_H0,2))),
                                          num_field_coeffs=num_field_coeffs, fieldsteps_per_ampstep = fieldsteps_per_ampstep, 
                                          alpha=alpha, C=C, n=n, u=u, gamma=gamma, kappa=kappa, radius=radius,
                                          wavenumber=var_wvn, intrinsic_curvature = var_H0)                   
                  }



def plot_save(range1, range2, results, title, exp_dir = '.'):
  """
  Save data in a CSV file and generate plots (2D heatmap)

   
  results array can be smaller than length of ranges, values will be assumed to correspond to end of range.
  if results array contains complex data, data will be saved to csv but no png will be generated.
  :param range1: list of values as row indices
  :param range2: list of values for column headers
  :param results: 2D array of simulation outcomes for the variable in question.  
  :param title: string, base of filename for csv and png.  should include variable name.
  :param expdir: directory path to save to, default here
  :rtype: None
  """
  #cut ranges to shape of data
  range1 = range1[-len(results):]
  range2 = range2[-len(results[0]):]
  df = pd.DataFrame(index=range1, columns=range2, data=results)
  df.to_csv(os.path.join(exp_dir, title + ".csv"))
  if not isinstance(df.loc[range1[0],range2[0]], complex):
    # plot only mean parameters that are real values, such as mean abs(something)
    print("plotting png")
    sb.heatmap(df, cmap = "viridis")
    plt.savefig(os.path.join(exp_dir, title + ".png"))
    plt.close()


def run_experiment(exp_type,  range1, range2):
  """
  Runs a 2D grid of single simulations 
  :param exp_type: tuple of strings, indicating which two variables will be varied
  :param experiment_title:
  :param range1: list of values for first variable to take
  :param range2: list of values for second variable to take
  :rtype: None
  """
  #make directory for the experiment
  global outdir
  now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  exp_dir= os.path.join("out", "exp-"+now)
  outdir = exp_dir
  os.mkdir(exp_dir)
  #save descritpion about how the experiment was run
  exp_notes = {"experiment type": " ".join(exp_type), "n_steps": n_steps, "temp":temp, "method": method, "C": C,"kappa": kappa,  "alpha": alpha, "n":n, "u":u, "num_field_coeffs":num_field_coeffs, "range1":range1, "range2": range2, "radius":radius, "amplitude":initial_amplitude, "wavenumber": wavenumber, "measure every n ampsteps":measure_every, "total number ampsteps": n_steps*measure_every, "notes":notes, "start_time": now, "me-version": me_version}
  if method == "sequential":
    exp_notes["fieldsteps per ampstep"] = fieldsteps_per_ampstep
  exp_notes = pd.DataFrame.from_dict(exp_notes, orient="index", columns=["value"])
  exp_notes.to_csv(os.path.join(exp_dir, "notes.csv"))
  # run experiment of the requested type
  try: 
    results = loop(functions_dict[exp_type], range1, range2)
  except KeyError:
    print("experiment type not found: ", exp_type)
    print("please set exp_type to one of", list(functions_dict.keys()))
    results = {}
  # save and plot results
  for name in results:
    plot_save(range1=range1, range2=range2, results=results[name], title=name, exp_dir=exp_dir)


def single_run(n_steps, 
              num_field_coeffs, fieldsteps_per_ampstep, alpha, C, n, u, gamma, kappa, radius, intrinsic_curvature,
              wavenumber,method = "sequential", amplitude=None, field_coeffs=None, outdir = None, title = None):
  """
  run a single simulation of lengths nsteps and given method 

  takes all variables as arguments rather than from global namespace so that some can be set to loop variables as needed
  :param nsteps: 
  :param method: choose from "simultaneous" for stepping all parameters at the same time
                 "sequential" for stepping real and complex paramters alternatingly
                 "no-field" for cylinder-only
                 "fixed-amplitude" for moving field on frozen cylinder shapes

  """
  ########### initial values ##############
  # read from files or generate
  if field_coeffs is None:
    field_coeffs = np.array(list(map(lambda x: metropolisengine.MetropolisEngine.gaussian_complex(sigma=.1),range(0,(2*num_field_coeffs[0]+1)*(2*num_field_coeffs[1]+1)))))
    print("initialized random complex coeffs", field_coeffs)
  if amplitude is None:
    global initial_amplitude
    amplitude = initial_amplitude #take from global
  if os.path.isfile("./last_sigma_2.pickle") and os.path.getsize("./last_sigma_2.pickle"):
    f = open('last_sigma_2.pickle', 'rb')
    sampling_width = pickle.load(f)
  else:
    sampling_width = .05
  if method!="no-field" and os.path.isfile("./last_cov.pickle") and os.path.getsize("./last_cov.pickle"):
    f = open('last_cov.pickle', 'rb')
    cov = pickle.load(f)
    if cov is not None and len(cov) != (2*num_field_coeffs[0]+1)*(2*num_field_coeffs[1]+1):
      print("rejected because dimensions are wrong")
      cov=None
  else:
    cov=None

  ########### setup system, metropolis engine, link energy functions #############
  if method == "no-field":
    se = cylinder.Cylinder(wavenumber=wavenumber, radius=radius, 
                           kappa=kappa, gamma=gamma, intrinsic_curvature =intrinsic_curvature)
    energy_fct_surface_term = lambda real_params, complex_params : se.calc_surface_energy(*real_params) #also need se.calc_field_energy_ampltiude_change to be saved to energy_dict "surface" slot  
    energy_fct_by_params_group = { "real": {"surface": energy_fct_surface_term}, "all":{ "surface":energy_fct_surface_term}}
    me = metropolisengine.MetropolisEngine(energy_functions = energy_fct_by_params_group,  initial_complex_params=None, initial_real_params = [float(amplitude)], 
                                 covariance_matrix_complex=None, sampling_width=sampling_width, temp=temp, complex_sample_method=None)
  else:  
    se = cylinder2d.Cylinder2D(wavenumber=wavenumber, radius=radius, alpha=alpha, C=C, u=u, n=n, 
                           kappa=kappa, gamma=gamma, intrinsic_curvature =intrinsic_curvature, num_field_coeffs= num_field_coeffs)
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
    energy_fct_field_term_alt = lambda real_params, complex_params: se.calc_field_energy_amplitude_change(*real_params,np.reshape(complex_params, (theta_array_len, z_array_len)))
    
    energy_fct_surface_term = lambda real_params, complex_params : se.calc_surface_energy(*real_params) #also need se.calc_field_energy_ampltiude_change to be saved to energy_dict "surface" slot  
    energy_fct_by_params_group = {"complex": {"field": energy_fct_field_term}, "real": {"field": energy_fct_field_term_alt, "surface": energy_fct_surface_term}, "all":{"field": energy_fct_field_term_alt, "surface":energy_fct_surface_term}}
    me = metropolisengine.MetropolisEngine(energy_functions = energy_fct_by_params_group,  initial_complex_params=field_coeffs, initial_real_params = [float(amplitude)], 
                                 covariance_matrix_complex=cov, sampling_width=sampling_width, temp=temp, complex_sample_method="magnitude-phase")
  #set metropolisengine parameter names
  params_names=["amplitude"]
  if method != "no-field":
    coeff_indices = [(n,m) for n in range(-num_field_coeffs[0], num_field_coeffs[0]+1) for m in range(-num_field_coeffs[1], num_field_coeffs[1]+1)]
    params_names.extend(["coeff_"+str(n)+"_"+str(m) for n,m in coeff_indices])
    me.params_names = params_names
  observables_names =  ["abs_"+name for name in params_names]
  observables_names.extend(["amplitude_squared"])
  me.observables_names = observables_names
  #also input system constraint : steps with |amplitude| > 1 to be rejected
  me.set_reject_condition(lambda real_params, complex_params : abs(real_params[0])>=.99 )  
  #save temp matrices after initiaing metropolis engine, which called inital energy calculation and filled tmp matrices
  se.save_temporary_matrices()
 
  ########### start of data collection ############
  if method == "sequential":
    for i in range(n_steps):
      for j in range(measure_every):
        accepted = me.step_real_group() #use outputted flag to trigger numerical integration in System on amplitude change  
        if accepted: se.save_temporary_matrices()
        for ii in range(fieldsteps_per_ampstep):
          me.step_complex_group() # no need to save and look at "accept" flag when only field coeffs change
          #print("field_coeffs", me.complex_params)
      print("measure", i , "sampling widths", me.real_group_sampling_width, me.complex_group_sampling_width, me.real_params,me.complex_params, me.energy)# "cov", me.covariance_matrix[0,0], me.covariance_matrix[1,1])
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
      print(i, me.complex_group_sampling_width, me.complex_params)
  elif method == "no-field":
    me.m = 1 #number of dimensions of parameter space for adaptive purposes such as target acceptance rate.  actually just 1 degree of freedom for amplitude-only run.
    for i in range(n_steps):
      for j in range(measure_every):
        accepted = me.step_real_group()
      me.measure() #update mean, covariance matrix, other parameters' mean by sampling this step

  ########## save, resturn results ################
  if outdir is not None and os.path.isdir(outdir):
    me.save_time_series()
    df = me.df #pd.DataFrame()
    #print(outdir, title, me.params_names, states)
    df.to_csv(os.path.join(outdir, title + ".csv"))
  #dump in files for order of magnitude estimate to start next simulation from
  f = open('last_cov.pickle', 'wb')
  pickle.dump(me.covariance_matrix_complex, f)
  #print("cov", np.round(me.covariance_matrix_complex,3))
  f = open('last_sigma_2.pickle', 'wb')
  if method == "sequential":
    pickle.dump([me.real_group_sampling_width, me.complex_group_sampling_width], f)
  else:
    pickle.dump(me.real_group_sampling_width, f)
  me.save_equilibrium_stats()
  result_means = me.equilibrated_means 
  return me.params_names, me.observables_names, result_means , me.covariance_matrix_complex

# global params - will use values set here if not loop variable
# coefficients
alpha = -1
C = 2
u = 1
n = 1
kappa = .3
gamma = 1
temp = .1
intrinsic_curvature = 0

# system dimensions
initial_amplitude= 0  #also fixed system amplitude for when amplitude is static
radius = 1
wavenumber = .8

# simulation details
num_field_coeffs = (6,1) # z-direction modes indices go from -.. to +..; theta direction indices go from -.. to +..
initial_sampling_width = .025
measure_every =5
fieldsteps_per_ampstep = 4  #only relevant for sequential

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='get description')
  parser.add_argument('-n', '--notes', dest = 'notes',  type=str, nargs=1,
                    help='add experiment description with --notes', required=True)
  notes=parser.parse_args().notes

  # specify type, range of plot; title of experiment
  loop_type = ("amplitude", "C")
  range1 = (0,)
  range2 = (5,2,1,.5,.1)
  n_steps = 800 
  method= "fixed-amplitude"

  run_experiment(loop_type,  range1, range2)

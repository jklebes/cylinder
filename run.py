import cmath
import pickle
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import system as ce #TODO: refactor name
import metropolis_engine
import scipy.integrate as integrate
import math
import collections
import timeit
import seaborn as sb
import argparse

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
      var_dict = dict([(name, cov_matrix[i][i]) for (i, name) in enumerate(names[:2+2*num_field_coeffs])])
      covar_dict = dict([(name1+"_"+name2, cov_matrix[i][j]) for (i, name1) in enumerate(names[:2+2*num_field_coeffs]) for (j, name2) in enumerate(names[:2+2*num_field_coeffs]) if i!= j])
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
      names, means, cov_matrix = single_run(n_steps=n_steps, method=method, outdir = outdir)
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
      names, means, cov_matrix = single_run(n_steps=n_steps, method=method, outdir = outdir, title = "wvn"+str(wvn)+"_kappa"+str(kappa))
      means_dict = dict([(name, mean) for (name,mean) in zip (names, means)])
      print(names)
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
      names, means, cov_matrix = single_run(n_steps=n_steps, method=method, outdir = outdir, title = "wvn"+str(wvn)+"_alpha"+str(alpha))
      means_dict = dict([(name, mean) for (name,mean) in zip (names, means)])
      print(names)
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
  #print(results)
  #cut ranges to shape of data
  range1 = range1[-len(results):]
  range2 = range2[-len(results[0]):]
  df = pd.DataFrame(index=range1, columns=range2, data=results)
  df.to_csv(os.path.join(exp_dir, title + ".csv"))
  print(df.loc[range1[0], range2[0]],  not isinstance(df.loc[range1[0],range2[0]], complex))
  if not isinstance(df.loc[range1[0],range2[0]], complex):
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
  exp_notes = {"experiment type": " ".join(exp_type), "n_steps": n_steps, "temp":temp, "method": method, "C": C,"kappa": kappa,  "alpha": alpha, "n":n, "u":u, "num_field_coeffs":num_field_coeffs, "range1":range1, "range2": range2, "radius":radius, "amplitude":initial_amplitude, "wavenumber": wavenumber, "measure every n ampsteps":measure_every, "total number ampsteps": n_steps*measure_every, "notes":notes}
  if method == "sequential":
    exp_notes["fieldsteps per ampstep"] = fieldsteps_per_ampstep
  exp_notes = pd.DataFrame.from_dict(exp_notes, orient="index", columns=["value"])
  exp_notes.to_csv(os.path.join(exp_dir, "notes.csv"))
  # run experiment of the requested type
  # TODO: could be switch or dict of functions
  if exp_type == ("wavenumber", "kappa"):
    results = loop_wavenumber_kappa(wavenumber_range=range1, kappa_range=range2, n_steps=n_steps, method=method, outdir = exp_dir)
    print(results["abs_amplitude"])
  elif exp_type == ("wavenumber", "alpha"):
    results = loop_wavenumber_alpha(wavenumber_range=range1, alpha_range=range2, n_steps=n_steps, method=method, outdir = exp_dir)
    print(results["abs_amplitude"])
  elif exp_type == ("amplitude", "C"):
    results = loop_amplitude_C(amplitude_range=range1, C_range=range2, n_steps=n_steps, method=method, outdir = exp_dir)
  elif exp_type == ("num_field_coeffs", "fieldsteps_per_ampstep"):
    results = loop_num_field_coeffs(num_field_coeff_range=range1, fieldstep_range=range2, n_steps=n_steps, method=method, outdir = exp_dir)
  #save results spreadsheets and plots - mainly mean abs(amplitude) and its variance
  print(results)
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
  if field_coeffs is None:
    #field_coeffs = dict([(i, metropolis_engine.MetropolisEngine.gaussian_complex()) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    #switch to nparray version
    field_coeffs = np.array(list(map(lambda x: metropolis_engine.MetropolisEngine.gaussian_complex(),range(0,2*num_field_coeffs+1))))
  if amplitude is None:
    global initial_amplitude
    amplitude = initial_amplitude #take from global
  ########### setup #############
  se = ce.System(wavenumber=wavenumber, radius=radius, alpha=alpha, C=C, u=u, n=n, kappa=kappa, gamma=gamma, num_field_coeffs= num_field_coeffs)
  #print("alpha", se.alpha)  
  #try getting last results from files
  if os.path.isfile("./last_sigma_2.pickle") and os.path.getsize("./last_sigma_2.pickle"):
    f = open('last_sigma_2.pickle', 'rb')
    sampling_width = pickle.load(f)
  else:
    sampling_width = .05
  if os.path.isfile("./last_cov.pickle") and os.path.getsize("./last_cov.pickle"):
    try:
      f = open('last_cov.pickle', 'rb')
      cov = pickle.load(f)
      x = np.zeros(2* num_field_coeffs+2)* cov # to check if dimensions mathc
    except ValueError:
      cov=None
  else:
    cov=None

  #use covariance from earlier files, optimize static covariance for speed
  me = metropolis_engine.ComplexAdaptiveMetropolisEngine(initial_field_coeffs=field_coeffs, covariance_matrix=cov,sampling_width=sampling_width,  initial_amplitude=amplitude, temp=temp)
  state = me.construct_state(amplitude=amplitude, field_coeffs=field_coeffs)
  se.evaluate_integrals(amplitude=amplitude)
  field_energy = se.calc_field_energy(state=state, amplitude_change=False)
  surface_energy = se.calc_surface_energy(amplitude, amplitude_change=False)
  ########### start of data collection ############
  states=[]
  other_states = []
  if method == "sequential":
    for i in range(n_steps):
      for j in range(measure_every):
        state, surface_energy, field_energy = me.step_amplitude(state=state,
                                                              field_energy=field_energy, surface_energy=surface_energy,
                                                              system=se)
        #print("amplitude change", amplitude, surface_energy, field_energy)
        for ii in range(fieldsteps_per_ampstep):
          # all simultaneously since main saving is from not re-integrating
          # energy difference method for changing single field coeff is only an advantage at larger number of fieldcoeffs
          state, field_energy = me.step_fieldcoeffs(state=state, field_energy=field_energy, 
              system=se)
          #print("field_energy",field_energy, me.field_sampling_width)
      #print("measure", i, "sampling widths", me.field_sampling_width, me.amplitude_sampling_width, "state", state)# "cov", me.covariance_matrix[0,0], me.covariance_matrix[1,1])
      me.measure(state=state) #update mean, covariance matrix, other parameters' mean by sampling this step
      states.append(state)
      other_states.append(me.construct_observables_state2(state))
  elif method == "simultaneous":
    for i in range(n_steps):
      for j in range(measure_every):
        amplitude, field_coeffs, surface_energy, field_energy = me.step_all(amplitude=amplitude,
                                                              field_coeffs=field_coeffs,
                                                              field_energy=field_energy, surface_energy=surface_energy,
                                                              system=se) #this doesnt measure mean, cov; update cov for sampling
      print("measure", i, "sampling widths", me.sampling_width, "cov", me.covariance_matrix[0,0], me.covariance_matrix[1,1])
      print("step counters", me.measure_step_counter, me.field_step_counter, me.amplitude_step_counter)
      me.measure(amplitude, field_coeffs) #update mean, covariance matrix, other parameters' mean by sampling this step
      # TODO collect to states, other_states    
  elif method == "fixed-amplitude":
    me.m -= 1
    for i in range(n_steps):
      for j in range(measure_every):
        for ii in range(fieldsteps_per_ampstep):
          state, field_energy = me.step_fieldcoeffs(state=state, field_energy=field_energy, system=se)
      print("measure", i, "sampling widths", me.field_sampling_width)#, me.amplitude_sampling_width, "cov", me.covariance_matrix[0,0], me.covariance_matrix[1,1])
      me.measure(state=state) #update mean, covariance matrix, other parameters' mean by sampling this step
      states.append(state)
      other_states.append(me.construct_observables_state2(state))
  elif method == "no-field":
    # TODO - we could not calculate field?
    field_coeffs = dict([(i, 0+0j) for i in range(-num_field_coeffs, num_field_coeffs+1)])
    surface_energy = se.calc_surface_energy(amplitude, amplitude_change=True)
    field_energy = se.calc_field_energy(field_coeffs, amplitude, amplitude_change=True)
    me.m = 1 #number of dimensions of parameter space for adaptive purposes such as target acceptance rate.  actually just 1 degree of freedom for amplitude-only run.
    for i in range(n_steps):
      for j in range(measure_every):
        state, surface_energy, field_energy = me.step_amplitude(state=state,
                                                              field_energy=field_energy, surface_energy=surface_energy,
                                                              system=se)
      me.measure(state=state) #update mean, covariance matrix, other parameters' mean by sampling this step
      states.append(state)
      other_states.append(me.construct_observables_state2(state))
  if outdir is not None and os.path.isdir(outdir):
    df = pd.DataFrame(states)
    print(outdir, title, me.params_names)
    df.to_csv(os.path.join(outdir, title + ".csv"), header=me.params_names)
    df_other = pd.DataFrame(other_states)
    df_other.to_csv(os.path.join(outdir, title + "_other.csv"), header = me.observables_names)
  #dump in files for order of magnitude estimate to start next simulation from
  f = open('last_cov.pickle', 'wb')
  pickle.dump(me.covariance_matrix, f)
  print("cov", np.round(me.covariance_matrix,3))
  f = open('last_sigma_2.pickle', 'wb')
  if method == "sequential":
    pickle.dump([me.field_sampling_width, me.amplitude_sampling_width], f)
  else:
    pickle.dump(me.sampling_width, f)
  result_means = list(me.mean)+list(me.observables) # change away from np array at this point because it is mixed complex and float type
  result_names = me.params_names
  result_names.extend(me.observables_names)
  return result_names, result_means , me.covariance_matrix

# coefficients
alpha = -1
C = 1
u = 1
n = 1
kappa = .1
gamma = 1
temp = 1

# system dimensions
initial_amplitude= 0.8  #also fixed system amplitude for when amplitude is static
radius = 1
wavenumber = .4

# simulation details
num_field_coeffs = 0
initial_sampling_width = .025
measure_every = 20
fieldsteps_per_ampstep = 1  #nly relevant for sequential

#notes = "with n = 6 - expect more of field conforming to shape.  On fixed shape a=.8." #describe motivation for a simulation here!

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='get description')
  parser.add_argument('-n', '--notes', dest = 'notes',  type=str, nargs=1,
                    help='add experiment description with --notes', required=True)
  notes=parser.parse_args().notes

  # specify type, range of plot; title of experiment
  loop_type = ("num_field_coeffs", "fieldsteps_per_ampstep")
  range1 = range(0,6, 1)
  range2 = range(1, 16, 5)
  n_steps = 500#n measuring steps- so there are n_steps * measure_every amplitude steps and n_steps*measure_every*fieldsteps_per_ampsteps fieldsteps
  method = "sequential"

  #single_run(kappa=kappa, wavenumber=wavenumber, n_steps=n_steps, method="no-field")

  run_experiment(loop_type,  range1, range2, n_steps, method)

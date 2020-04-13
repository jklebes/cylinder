import pickle
import os 
import cmath
import copy
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import system as ce #TODO: refactor name
import metropolis_engine

def loop_wavenumber_kappa(wavenumber_range, kappa_range, n_steps):
  """
  A set of runs looping over a grid of wavenumber, bending rigdity values
  :param wavenumber_range:
  :param kappa_range:
  :return:
  """
  # TODO: maybe convergence check
  results = []
  for wavenumber in wavenumber_range:
    results_line = []
    times_line = []
    for kappa in kappa_range:
      # reset to starting values 0 for each run
      amplitude = 0
      initial_field_coeffs = dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
      # run
      field_coeffs, amplitude = single_run(kappa=kappa, wavenumber=wavenumber, field_coeffs=initial_field_coeffs, amplitude=amplitude,
                                    n_steps=n_steps)
      results_line.append(amplitude)
      print(kappa, wavenumber, amplitude)
    results.append(results_line)
  return  np.array(results)


def plot_save(wavenumber_range, kappa_range, results, title):
  """
  Save data in a CSV file and generate plots
  :param wavenumber_range:
  :param kappa_range:
  :param results:
  :param title:
  :return:
  """
  df = pd.DataFrame(index=wavenumber_range, columns=kappa_range, data=results)
  df.to_csv(title + ".csv")
  plt.imshow(results, extent=[min(kappa_range), max(kappa_range), max(wavenumber_range), min(wavenumber_range)])
  plt.colorbar()
  plt.savefig(title + ".png")
  plt.close()


def run_experiment(type, experiment_title, range1, range2, n_steps, plot=True):
  """
  Launches a set of runs exploring the stability on a grid of 2 parameters.
  :param type:
  :param experiment_title:
  :param range1:
  :param range2:
  :param amp_steps:
  :param converge_stop:
  :param fieldsteps_per_ampstep:
  :param plot:
  :return:
  """
  # TODO: lookup from type, decide on loop function
  converge_time, results = loop_wavenumber_kappa(wavenumber_range=range1, kappa_range=range2, n_steps=n_steps)
  if plot:
    plot_save(wavenumber_range=range1, kappa_range=range2, results=results, title=experiment_title)
    plot_save(wavenumber_range=range1, kappa_range=range2, results=converge_time,
              title=experiment_title + "_convergence_time_")
  # TODO: save data
  print(converge_time, results)


def single_run(kappa,wavenumber, n_steps, method = "simultaneous", field_coeffs=None, amplitude=None):
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
    field_coeffs = dict([(i, metropolis_engine.MetropolisEngine.gaussian_complex()) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
  if amplitude is None:
    amplitude = 0
  ########### setup #############
  se = ce.System(wavenumber=wavenumber, radius=radius, alpha=alpha, C=C, u=u, n=n, kappa=kappa, gamma=gamma)
  
  #try getting last results from files
  if os.path.isfile("./last_sigma.pickle") and os.path.getsize("./last_sigma.pickle"):
    f = open('last_sigma.pickle', 'rb')
    sampling_width = pickle.load(f)
  else:
    sampling_width = .05
  if os.path.isfile("./last_cov.pickle") and os.path.getsize("./last_cov.pickle"):
    f = open('last_cov.pickle', 'rb')
    cov = pickle.load(f)
  else:
    cov=None
  
  me = metropolis_engine.RobbinsMonroAdaptiveMetropolisEngine(initial_field_coeffs=field_coeffs, covariance_matrix=cov,sampling_width=sampling_width,  initial_amplitude=amplitude, temp=temp)
  surface_energy = se.calc_surface_energy(amplitude, amplitude_change=True)
  field_energy = se.calc_field_energy(field_coeffs, amplitude, amplitude_change=True)
  ########### start of data collection ############
  amplitudes = [amplitude]
  c_0s=[abs(field_coeffs[0])]
  sigmas=[]
  means=[]
  amplitude_cov=[]
  amplitude_c0_cov=[]
  c0_cov=[]
  if method == "sequential":
    for i in range(n_steps):
      amplitude, field_energy, surface_energy = me.step_amplitude(amplitude=amplitude, field_coeffs=field_coeffs,
                                                             field_energy=field_energy, surface_energy=surface_energy,
                                                             system=se)
      field_coeffs, field_energy = me.step_fieldcoeffs_sequential(amplitude=amplitude,
                                                               field_coeffs=field_coeffs, field_energy=field_energy,
                                                               surface_energy=surface_energy,
                                                               system=se)
      amplitudes.append(amplitude)
  elif method == "simultaneous":
    for i in range(n_steps):
      amplitude, field_coeffs, field_energy, surface_energy = me.step_all(amplitude=amplitude,
                                                             field_coeffs=field_coeffs,
                                                             field_energy=field_energy, surface_energy=surface_energy,
                                                             system=se)
      amplitudes.append(amplitude)
      sigmas.append(me.sampling_width)
      c_0s.append(abs(field_coeffs[0]))
      amplitude_cov.append(me.covariance_matrix[0,0])
      amplitude_c0_cov.append(me.covariance_matrix[0,4])
      c0_cov.append(me.covariance_matrix[4,4])
      means.append(me.mean[0])
  plt.scatter(range(len(amplitudes)), amplitudes, label='amplitude')
  plt.scatter(range(len(amplitudes)), c_0s, label='fieldcoeff 0')
  plt.legend()
  plt.savefig("amplitudes_vs_time.png")
  plt.close()
  plt.scatter(range(len(sigmas)), sigmas, marker='.', label="sigma")
  plt.scatter(range(len(sigmas)), amplitude_cov,marker='.', label = "covariance matrix[0,0]")
  plt.scatter(range(len(sigmas)), amplitude_c0_cov,marker='.', label = "covariance matrix[0,4]")
  plt.scatter(range(len(sigmas)), c0_cov,marker='.', label = "covariance matrix[4,4]")
  plt.legend()
  plt.savefig("amplitude_proposal_dist.png")
  plt.close()
  plt.scatter(range(len(sigmas)), [a*s**2 for (a,s) in zip(amplitude_cov, sigmas)] ,marker='.', label = "sigma**2 * cov[0,0]")
  plt.scatter(range(len(sigmas)), [a*s**2 for (a,s) in zip(amplitude_c0_cov, sigmas)] ,marker='.', label = "sigma**2 * cov[0,4]")
  plt.scatter(range(len(sigmas)), [a*s**2 for (a,s) in zip(c0_cov, sigmas)] ,marker='.', label = "sigma**2 * cov[4,4]")
  plt.legend()
  plt.savefig("amplitude_proposal_dist_2.png")
  plt.close()
  plt.scatter(range(len(sigmas)), means ,marker='.', label = "amplitude_mean")
  plt.legend()
  plt.savefig("mean_amplitude.png")
  plt.close()
  #dump in files
  f = open('last_cov.pickle', 'wb')
  pickle.dump(me.covariance_matrix, f)
  f = open('last_sigma.pickle', 'wb')
  pickle.dump(me.sampling_width, f)
  return amplitudes

# coefficients
alpha = -1
C = 1
u = 1
n = 1
kappa = 1
gamma = 1
temp = 0.1

# system dimensions
radius = 1
wavenumber = 1.5

# simulation details
num_field_coeffs = 3
#user-define jump distribution
#sampling_dist = lambda width :  random.uniform(-1*x, x)
sampling_dist = lambda width : random.gauss(0, width)
initial_sampling_width = .025

if __name__ == "__main__":
  # specify type, range of plot; title of experiment
  loop_type = ("wavenumber", "kappa")
  experiment_title = loop_type[0] + "_" + loop_type[1]
  range1 = np.arange(0.005, 1.2, .05)
  range2 = np.arange(0, 1, .05)
  n_steps = 800

  assert (alpha <= 0)

  single_run(kappa, wavenumber, n_steps, method="simultaneous")

  # run_experiment(loop_type, experiment_title,
  # range1, range2, amp_steps, converge_stop, fieldsteps_per_ampstep)

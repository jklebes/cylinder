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
def loop_wavenumber_kappa(wavenumber_range, kappa_range, n_steps, method = "simultaneous"):
  """
  A set of runs looping over a grid of wavenumber, bending rigdity values
  :param wavenumber_range:
  :param kappa_range:
  :return:
  """
  abs_amplitude = []
  cov_amplitude = []
  abs_c0 = []
  cov_c0 = []
  cross_cov = []
  c0_index = 1+num_field_coeffs
  for wavenumber in wavenumber_range:
    abs_amplitude_line = []
    cov_amplitude_line = []
    abs_c0_line=[]
    cov_c0_line=[]
    cross_cov_line= []
    for kappa in kappa_range:
      # run
      field_coeffs, amplitude, cov_matrix = single_run(kappa=kappa, wavenumber=wavenumber, n_steps=n_steps, method=method)
      abs_amplitude_line.append(amplitude)
      cov_amplitude_line.append(cov_matrix[0,0])
      abs_c0_line.append(abs(field_coeffs[0]))
      cov_c0_line.append(cov_matrix[c0_index,c0_index])
      cross_cov_line.append(cov_matrix[0,c0_index])
      print(kappa, wavenumber, amplitude)
    abs_amplitude.append(abs_amplitude_line)
    cov_amplitude.append(cov_amplitude_line)
    abs_c0.append(abs_c0_line)
    cov_c0.append(cov_c0_line)
    cross_cov.append(cross_cov_line)
  return  {"abs_amplitude": np.array(abs_amplitude),
           "cov_amplitude":np.array(cov_amplitude),
           "abs_c0": np.array(abs_c0),
           "cov_c0": np.array(cov_c0),
           "cross_cov": np.array(cross_cov)}


def plot_save(wavenumber_range, kappa_range, results, title, exp_dir = '.'):
  """
  Save data in a CSV file and generate plots
  :param wavenumber_range:
  :param kappa_range:
  :param results:
  :param title:
  :return:
  """
  print(results)
  df = pd.DataFrame(index=wavenumber_range, columns=kappa_range, data=results)
  df.to_csv(os.path.join(exp_dir, title + ".csv"))
  plt.imshow(results, extent=[min(kappa_range), max(kappa_range), max(wavenumber_range), min(wavenumber_range)])
  plt.colorbar()
  plt.savefig(os.path.join(exp_dir, title + ".png"))
  plt.close()


def run_experiment(exp_type, experiment_title, range1, range2, n_steps, method = "simultaneous"):
  """
  Launches a set of runs exploring the stability on a grid of 2 parameters.
  :param type:
  :param experiment_title:
  :param range1:
  :param range2:
  :return:
  """
  # TODO: lookup from type, decide on loop function
  results = loop_wavenumber_kappa(wavenumber_range=range1, kappa_range=range2, n_steps=n_steps, method=method)

  #make directory for the experiment
  now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  exp_dir= os.path.join("out", "exp-"+now)
  os.mkdir(exp_dir)
  #save eveythin about how the experiment was run
  exp_notes = {"n_steps": n_steps, "temp":temp, "method": method, "C": C, "alpha": alpha, "n":n, "u":u, "num_field_coeffs":num_field_coeffs,
      "range1":range1, "range2": range2, "radius":radius, "notes":""}
  notes = pd.DataFrame.from_dict(exp_notes, orient="index", columns=["value"])
  notes.to_csv(os.path.join(exp_dir, "notes.csv"))
  #save results spreadsheets and plots - mainly mean abs(amplitude) and its variance
  for name in results:
    plot_save(wavenumber_range=range1, kappa_range=range2, results=results[name], title=experiment_title+ "_"+name+"_", exp_dir=exp_dir)
  print(results["abs_amplitude"])


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
  print("alpha", se.alpha)  
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

  #use covariance from earlier files, optimize static covariance for speed
  me = metropolis_engine.RobbinsMonroAdaptiveMetropolisEngine(initial_field_coeffs=field_coeffs, covariance_matrix=cov,sampling_width=sampling_width,  initial_amplitude=amplitude, temp=temp)
  field_energy = se.calc_field_energy(field_coeffs, amplitude, amplitude_change=True)
  surface_energy = se.calc_surface_energy(amplitude, amplitude_change=False)
  ########### start of data collection ############
  amplitudes = []
  c_0s=[]
  sigmas=[]
  means=[]
  amplitude_cov=[]
  amplitude_c0_cov=[]
  c0_cov=[]
  step_sizes=[]
  if method == "sequential":
    for i in range(n_steps):
      #TODO correct interpretation of output tuple?
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
      amplitude, field_coeffs, surface_energy, field_energy = me.step_all(amplitude=amplitude,
                                                             field_coeffs=field_coeffs,
                                                             field_energy=field_energy, surface_energy=surface_energy,
                                                             system=se)
      amplitudes.append(amplitude)
      sigmas.append(me.sampling_width)
      c_0s.append(abs(field_coeffs[0]))
      amplitude_cov.append(me.covariance_matrix[0,0])
      amplitude_c0_cov.append(me.covariance_matrix[0,1])
      c0_cov.append(me.covariance_matrix[1,1])
      means.append(me.mean[0])
      step_sizes.append(me.steplength_c)
  elif method == "no-field":
    field_coeffs = dict([(i, 0+0j) for i in range(-num_field_coeffs, num_field_coeffs+1)])
    #me = metropolis_engine.StaticCovarianceAdaptiveMetropolisEngine(initial_field_coeffs=field_coeffs, covariance_matrix=cov,sampling_width=sampling_width,  initial_amplitude=amplitude, temp=temp) # no need ot calculate covariance matrix for amplitude-only run
    surface_energy = se.calc_surface_energy(amplitude, amplitude_change=True)
    field_energy = se.calc_field_energy(field_coeffs, amplitude, amplitude_change=True)
    me.m = 1 #number of dimensions of parameter space for adaptive purposes such as target acceptance rate.  actually just 1 degree of freedom for amplitude-only run.
    for i in range(n_steps):
      #print("field energy in", field_energy)
      amplitude, surface_energy, field_energy = me.step_amplitude(amplitude=amplitude, field_coeffs=field_coeffs, field_energy=field_energy, surface_energy=surface_energy, system=se)
      #print("field energy recieved", field_energy)
      me.step_counter+=1
      amplitudes.append(amplitude)
      # print("amplitude", amplitude)
      sigmas.append(me.sampling_width)
      step_sizes.append(me.steplength_c)
      amplitude_cov.append(me.covariance_matrix[0,0])
      c_0s.append(abs(field_coeffs[0]))
      means.append(me.mean[1])
  print("latest amplitude", amplitude, "surface area", se.A_integrals[0].real, "bending energy", integrate.quad(lambda z: se.Kthth_integrand(amplitude, z),  0, 2 * math.pi / se.wavenumber))
  se.calc_surface_energy(amplitude=.05)
  print("curved a=.05", "surface area,", se.A_integrals[0].real, "bending energy", integrate.quad(lambda z: se.Kthth_integrand(.05,  z),  0, 2 * math.pi / se.wavenumber))
  se.calc_surface_energy(amplitude=0)
  print("flat", "surface area", se.A_integrals[0].real, "bending energy", integrate.quad(lambda z: se.Kthth_integrand(0, z),  0, 2 * math.pi / se.wavenumber))
  plt.scatter(range(len(sigmas)), sigmas, marker='.', label="sigma")
  plt.scatter(range(len(sigmas)), amplitude_cov, marker='.', label="covariance matrix[0,0]")
  #plt.scatter(range(len(sigmas)), amplitude_c0_cov, marker='.', label="covariance matrix[0,1]")
  #plt.scatter(range(len(sigmas)), c0_cov, marker='.', label="covariance matrix[1,1]")
  plt.legend()
  plt.savefig("amplitude_proposal_dist.png")
  plt.close()
  plt.scatter(range(len(amplitudes)), amplitudes, label='amplitude')
  plt.scatter(range(len(amplitudes)), c_0s, label='fieldcoeff 0')
  plt.legend()
  plt.savefig("amplitudes_vs_time.png")
  plt.close()
  plt.scatter(range(len(sigmas)), [a*s**2 for (a,s) in zip(amplitude_cov, sigmas)] ,marker='.', label = "sigma**2 * cov[0,0]")
  #plt.scatter(range(len(sigmas)), [a*s**2 for (a,s) in zip(amplitude_c0_cov, sigmas)] ,marker='.', label = "sigma**2 * cov[0,2]")
  #plt.scatter(range(len(sigmas)), [a*s**2 for (a,s) in zip(c0_cov, sigmas)] ,marker='.', label = "sigma**2 * cov[2,2]")
  plt.legend()
  plt.savefig("amplitude_proposal_dist_2.png")
  plt.close()
  plt.scatter(range(len(sigmas)), step_sizes, label = "adaptation step size",marker='.') 
  plt.legend()
  plt.savefig("steplength_constant.png")
  plt.close()
  plt.scatter(range(len(sigmas)), means ,marker='.', label = "c0_mean")
  plt.legend()
  plt.savefig("mean_c0.png")
  plt.close()
  #dump in files
  f = open('last_cov.pickle', 'wb')
  pickle.dump(me.covariance_matrix, f)
  print("cov", np.round(me.covariance_matrix,3))
  f = open('last_sigma.pickle', 'wb')
  pickle.dump(me.sampling_width, f)
  print("sampling width", me.sampling_width)
  print("last surface energy", se.calc_surface_energy(amplitude))
  print("from bending", se.calc_surface_energy(amplitude, amplitude_change=False)-gamma*se.A_integrals[0].real)
  print("flat", se.calc_surface_energy(.00001))
  print("flat", se.calc_surface_energy(0))
  print("from bending", se.calc_surface_energy(.00001, amplitude_change=False)-gamma*se.A_integrals[0].real)
  return me.mean[1:], me.mean[0], me.covariance_matrix

# coefficients
alpha = -1
C = 1
u = 1
n = 1
kappa = 0
gamma = 1
temp = 0.001

# system dimensions
radius = 1
wavenumber = 1.50

# simulation details
num_field_coeffs = 0
initial_sampling_width = .025

if __name__ == "__main__":
  # specify type, range of plot; title of experiment
  loop_type = ("wavenumber", "kappa")
  experiment_title = loop_type[0] + "_" + loop_type[1]
  range1 = np.arange(0.005, .5, .03)
  range2 = np.arange(0, .3, .03)
  n_steps = 10000
  method = "simultaneous"

  assert (alpha <= 0)

  #single_run(kappa=kappa, wavenumber=wavenumber, n_steps=n_steps, method="no-field")

  run_experiment(loop_type, experiment_title, range1, range2, n_steps, method)

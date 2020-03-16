import cmath
import copy
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calc_energy as ce


def step_fieldcoeffs_sequential(wavenumber, field_coeffs, field_energy, surface_energy, amplitude, field_max_stepsize,
                                system_energy):
  """
  Try stepping each field coefficient c_i once, in a random order
  -> this is Gibbs sampling
  I chose sequential for greater acceptance rates
  Disadvantage: 2n+1 recalculations, of the products and sums in system_energy.calc_field_energy
   (not of numerical integration)
  :param wavenumber:
  :param field_coeffs:
  :param field_energy:
  :param surface_energy:
  :param amplitude:
  :param field_max_stepsize:
  :param system_energy:
  :return:
  """
  indices_randomized = [*field_coeffs]  # unpack iterable into list - like list() but faster
  random.shuffle(indices_randomized)
  for index in indices_randomized:
    field_coeffs, field_energy = step_fieldcoeff(index, wavenumber,
                                                 field_coeffs, field_energy, surface_energy, amplitude,
                                                 field_max_stepsize, system_energy)
  print(field_energy, field_coeffs)
  return field_coeffs, field_energy


def step_fieldcoeff(field_coeff_index, wavenumber,
                    field_coeffs, field_energy, surface_energy, amplitude, field_max_stepsize, system_energy):
  """
  Stepping a single field coefficient c_i: generate a random complex value.  Accept or reject.
  :param field_coeff_index:
  :param wavenumber:
  :param field_coeffs:
  :param field_energy:
  :param surface_energy:
  :param amplitude:
  :param field_max_stepsize:
  :param system_energy:
  :return:
  """
  proposed_field_coeff = field_coeffs[field_coeff_index] + rand_complex(field_sampling_width)
  new_field_energy = system_energy.calc_field_energy_diff(field_coeff_index, proposed_field_coeff, field_coeffs,
                                                          amplitude, radius=radius, n=n, alpha=alpha,
                                                     C=C, u=u,
                                                     wavenumber=wavenumber, amplitude_change=False)
  if metropolis_decision(temp, field_energy + surface_energy, new_field_energy + surface_energy):
    field_energy = new_field_energy
    field_coeffs[field_coeff_index] = proposed_field_coeff
  return field_coeffs, field_energy


def step_amplitude(wavenumber, kappa, amplitude, field_coeffs, surface_energy, field_energy, system_energy):
  """
  Stepping amplitude by metropolis algorithm.
  :param wavenumber:
  :param kappa:
  :param amplitude:
  :param field_coeffs:
  :param surface_energy:
  :param field_energy:
  :param system_energy:
  :return:
  """
  proposed_amplitude = amplitude + sampling_dist(amplitude_sampling_width )
  if abs(proposed_amplitude) >= 1:
    # don't accept.
    # like an infinite energy barrier to self-intersection.
    # does not violate symmetric jump distribution, because this is like
    # an energy-landscape-based decision after generation
    return amplitude, surface_energy, field_energy
  new_field_energy = system_energy.calc_field_energy(field_coeffs, proposed_amplitude, radius=radius, n=n, alpha=alpha,
                                                     C=C, u=u,
                                                     wavenumber=wavenumber)
  new_surface_energy = system_energy.calc_surface_energy(proposed_amplitude, wavenumber=wavenumber, radius=radius,
                                                         gamma=gamma,
                                                         kappa=kappa, amplitude_change=False)
  if metropolis_decision(temp, (field_energy + surface_energy), (new_field_energy + new_surface_energy)):
    field_energy = new_field_energy
    surface_energy = new_surface_energy
    amplitude = proposed_amplitude
  return amplitude, surface_energy, field_energy

def step_all(wavenumber, kappa, amplitude, field_coeffs, surface_energy, field_energy, system_energy):
  """
  Step all parameters (amplitude, field coeffs) simultaneously.  True metropolis algorithm.
  Much faster than sequential (Gibbs) sampling but low acceptance rate at low temp.
  :param wavenumber:
  :param kappa:
  :param amplitude:
  :param field_coeffs:
  :param surface_energy:
  :param field_energy:
  :param system_energy:
  :return:
  """
  # TODO: record acceptance rate, aim for 20-60 %
  proposed_amplitude = amplitude + sampling_dist(amplitude_sampling_width )
  proposed_field_coeffs = copy.copy(field_coeffs)
  for index in field_coeffs:
    proposed_field_coeffs[index] += rand_complex(field_sampling_width)
  if abs(proposed_amplitude) >= 1:
    return amplitude, field_coeffs, surface_energy, field_energy
  new_field_energy = system_energy.calc_field_energy(proposed_field_coeffs, proposed_amplitude, radius=radius, n=n, alpha=alpha,
                                                     C=C, u=u,
                                                     wavenumber=wavenumber, amplitude_change=True)
  new_surface_energy = system_energy.calc_surface_energy(proposed_amplitude, wavenumber=wavenumber, radius=radius,
                                                         gamma=gamma,
                                                         kappa=kappa, amplitude_change=False)
  if metropolis_decision(temp, (field_energy + surface_energy), (new_field_energy + new_surface_energy)):
    field_energy = new_field_energy
    surface_energy = new_surface_energy
    amplitude = proposed_amplitude
    field_coeffs = proposed_field_coeffs
  return amplitude, field_coeffs, surface_energy, field_energy

def metropolis_decision(temp, old_energy, proposed_energy):
  """
  Considering energy difference and temperature, return decision to accept or reject step
  :param temp: system temperature
  :param old_energy: current system total energy
  :param proposed_energy: system total energy after proposed change
  :return: True to accept value, False to reject
  """
  diff = proposed_energy - old_energy
  if diff <= 0:
    return True  # choice was made that 0 difference -> accept change
  elif diff > 0 and temp == 0:
    return False
  else:
    probability = math.exp(- 1 * diff / temp)
    assert probability >= 0
    assert probability <= 1
    if random.uniform(0, 1) <= probability:
      return True
    else:
      return False


def run(field_coeffs, wavenumber, kappa, amplitude,
        amp_steps, fieldsteps_per_ampstep, field_max_stepsize):  # constants for this run
  """
  TODO : name this function better
  A single run with a given set of system constants and a given starting point
  :param field_coeffs:
  :param wavenumber:
  :param kappa:
  :param amplitude:
  :param amp_steps:
  :param fieldsteps_per_ampstep:
  :param field_max_stepsize:
  :return:
  """
  se = ce.System_Energy() # this object exists to store results of numerical integration between steps
  field_energy = se.calc_field_energy(field_coeffs, amplitude, radius=radius, n=n, alpha=alpha, C=C, u=u,
                                      wavenumber=wavenumber)
  surface_energy = se.calc_surface_energy(amplitude, wavenumber, kappa=kappa, radius=radius, gamma=gamma,
                                          amplitude_change=False)
  # TODO : don't have different unmbers amp and field steps.  this was done from mistaken POV of emulating dynamics
  for i in range(amp_steps):
    for j in range(fieldsteps_per_ampstep):
      field_coeffs, field_energy = step_fieldcoeffs_sequential(wavenumber=wavenumber,
                                                               amplitude=amplitude,
                                                               field_coeffs=field_coeffs, field_energy=field_energy,
                                                               surface_energy=surface_energy,
                                                               field_max_stepsize=field_max_stepsize)
    amplitude, field_energy, surface_energy = step_amplitude(wavenumber=wavenumber, kappa=kappa,
                                                             amplitude=amplitude, field_coeffs=field_coeffs,
                                                             field_energy=field_energy, surface_energy=surface_energy)
  return field_coeffs, abs(amplitude)


def loop_wavenumber_kappa(wavenumber_range, kappa_range, amp_steps, converge_stop,
                          fieldsteps_per_ampstep):
  """
  A set of runs looping over a grid of wavenumber, bending rigdity values
  :param wavenumber_range:
  :param kappa_range:
  :param amp_steps:
  :param converge_stop:
  :param fieldsteps_per_ampstep:
  :return:
  """
  # TODO: maybe convergence check
  converge_times = []
  results = []
  for wavenumber in wavenumber_range:
    results_line = []
    times_line = []
    for kappa in kappa_range:
      # reset to starting values 0 for each run
      amplitude = 0
      initial_field_coeffs = dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
      # run
      field_coeffs, amplitude = run(field_coeffs=initial_field_coeffs,
                                    wavenumber=wavenumber, field_max_stepsize=field_sampling_width,
                                    kappa=kappa, amplitude=amplitude,
                                    amp_steps=amp_steps, fieldsteps_per_ampstep=fieldsteps_per_ampstep)
      results_line.append(amplitude)
      print(kappa, wavenumber, amplitude)
    results.append(results_line)
    converge_times.append(times_line)
  return (np.array(converge_times), np.array(results))


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

def rand_complex(maxamplitude=1):
  """
  random complex number in/on unit (or other) circle
  default amplitude : 1
  distribution not uniform wrt area of circle
  :return:
  """
  amplitude = random.uniform(0,maxamplitude)
  phase = random.uniform(0, 2*math.pi)
  return cmath.rect(amplitude, phase)

def record_amplitude_vs_time(kappa, wavenumber, n_steps, method = "simultaneous"):
  """
  for examining a single run over time.
  :param kappa:
  :param wavenumber:
  :param amp_steps:
  :param fieldsteps_per_ampstep:
  :return:
  """
  field_coeffs = dict([(i, rand_complex()) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
  amplitude = .6 # TODO: optionally pass in initial amplitude, field
  amplitudes = [amplitude]
  se = ce.System_Energy()  # object stores A,B,D integral values -> less recalculate
  field_energy = se.calc_field_energy(field_coeffs, amplitude, radius=radius, n=n, alpha=alpha, C=C, u=u,
                                      wavenumber=wavenumber)
  surface_energy = se.calc_surface_energy(amplitude, wavenumber, kappa=kappa, radius=radius, gamma=gamma,
                                          amplitude_change=False)
  if method == "sequential":
    for i in range(n_steps):
      field_coeffs, field_energy = step_fieldcoeffs_sequential(wavenumber=wavenumber,
                                                               amplitude=amplitude,
                                                               field_coeffs=field_coeffs, field_energy=field_energy,
                                                               surface_energy=surface_energy,
                                                               field_max_stepsize=field_sampling_width,
                                                               system_energy=se)
      amplitude, field_energy, surface_energy = step_amplitude(wavenumber=wavenumber, kappa=kappa,
                                                             amplitude=amplitude, field_coeffs=field_coeffs,
                                                             field_energy=field_energy, surface_energy=surface_energy,
                                                             system_energy=se)
      amplitudes.append(amplitude)
  elif method == "simultaneous":
    for i in range(n_steps):
      amplitude, field_coeffs, field_energy, surface_energy = step_all(wavenumber=wavenumber, kappa=kappa,
                                                             amplitude=amplitude, field_coeffs=field_coeffs,
                                                             field_energy=field_energy, surface_energy=surface_energy,
                                                             system_energy=se)
      amplitudes.append(amplitude)
      print(amplitude, field_coeffs)
  plt.scatter(range(len(amplitudes)), amplitudes)
  plt.savefig("amplitudes_vs_time.png")
  return amplitudes


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


# TODO: where to set system variables?
# coefficients
alpha = -1
C = 1
u = 1
n = 1
kappa = 1
gamma = 1
temp = .1

# system dimensions
radius = 1
wavenumber = 1.5

# simulation details
num_field_coeffs = 3
#user-define jump distribution
#sampling_dist = lambda width :  random.uniform(-1*x, x)
sampling_dist = lambda width : random.gauss(0, width)
amplitude_sampling_width = .025 #
field_sampling_width = .025

if __name__ == "__main__":
  # specify type, range of plot; title of experiment
  loop_type = ("wavenumber", "kappa")
  experiment_title = loop_type[0] + "_" + loop_type[1]
  range1 = np.arange(0.005, 1.2, .05)
  range2 = np.arange(0, 1, .05)
  n_steps = 1000

  assert (alpha <= 0)

  record_amplitude_vs_time(kappa, wavenumber, n_steps, method="sequential")

  # run_experiment(loop_type, experiment_title,
  # range1, range2, amp_steps, converge_stop, fieldsteps_per_ampstep)

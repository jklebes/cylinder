import copy
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calc_energy as ce

def step_fieldcoeffs_all(wavenumber, field_coeffs, field_energy, amplitude, field_max_stepsize):
  """
  try stepping each of the field coeffs once, in a random order
  :param temp:
  :param wavenumber:
  :param field_coeffs:
  :param field_energy:
  :param amplitude:
  :return:
  """
  indices_randomized = [*field_coeffs] # unpack iterable into list - like list() but faster
  random.shuffle(indices_randomized)
  for index in indices_randomized:
    field_coeffs, field_energy = step_fieldcoeff(index, wavenumber,
                                                 field_coeffs, field_energy, amplitude, field_max_stepsize)
  return field_coeffs, field_energy

def step_fieldcoeff(field_coeff_index, wavenumber,
                    field_coeffs, field_energy, amplitude, field_max_stepsize):
  """
  steps one field coeff
  :param temp:
  :param wavenumber:
  :param field_coeffs:
  :param field_energy:
  :param amplitude:
  :return:
  """
  proposed_field_coeffs =copy.copy(field_coeffs)
  proposed_field_coeffs[field_coeff_index]+=field_max_stepsize*complex(random.uniform(-1,1), random.uniform(-1,1))

  new_field_energy = ce.calc_field_energy(proposed_field_coeffs, amplitude,radius=radius,n=n, alpha=alpha,C=C,u=u,
                                          wavenumber=wavenumber)
  diff = new_field_energy - field_energy

  # TODO: look at this logic
  if diff <= 0 or (temp!=0 and random.uniform(0, 1) <= math.exp(-diff / temp)):
    #accept change
    field_energy=new_field_energy
    field_coeffs= proposed_field_coeffs
  return field_coeffs, field_energy

def step_amplitude(wavenumber, kappa, amplitude, field_coeffs, surface_energy, field_energy):
  """

  :param temp:
  :param wavenumber:
  :param kappa:
  :param amplitude:
  :param field_coeffs:
  :param surface_energy:
  :param field_energy:
  :return:
  """
  proposed_amplitude = amplitude + amplitude_max_stepsize*random.uniform(-1,1)
  new_field_energy = ce.calc_field_energy(field_coeffs, proposed_amplitude,radius=radius,n=n, alpha=alpha,C=C,u=u,
                                          wavenumber=wavenumber)
  new_surface_energy=ce.calc_surface_energy(proposed_amplitude, wavenumber=wavenumber, radius=radius,gamma=gamma,
                                            kappa=kappa, amplitude_change=False)
  # TODO: check this logic
  if decide_change(temp, (field_energy + surface_energy) ,(new_field_energy + new_surface_energy)):
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
  return amplitude, surface_energy, field_energy

def metropolis_decision(temp, old_energy, proposed_energy):
  # TODO: check this logic
  diff= proposed_energy - old_energy
  if temp==0:
    probability = 1 if diff<=0 else 1 #choice was made that 0 difference -> accept change
  else:
    try:
      probability = min(math.exp(- 1* diff / temp), 1)
    except OverflowError:
      if -1*diff/temp >0:
        probability = 1
      else:
        raise(OverflowError)
  assert probability >= 0
  assert probability <= 1
  if random.uniform(0, 1) <= probability:
      return True
  else:
    return False

def run(field_coeffs, wavenumber, kappa, amplitude,
        amp_steps, fieldsteps_per_ampstep, field_max_stepsize): #constants for this run
  """
  one run
  loop over this for statistics or to try differnt system parameters
  :param temp:
  :param initial_field_coeffs:
  :param wavenumber:
  :param kappa:
  :param initial_amplitude:
  :param amp_steps:
  :param fieldsteps_per_ampstep:
  :return:
  """
  field_energy = ce.calc_field_energy(field_coeffs, amplitude,radius=radius,n=n, alpha=alpha,C=C,u=u,
                                          wavenumber=wavenumber)
  surface_energy = ce.calc_surface_energy(amplitude, wavenumber, kappa=kappa, radius=radius, gamma=gamma,
                                          amplitude_change=False )
  for i in range(amp_steps):
    if abs(amplitude) > 1: break
    for j in range(fieldsteps_per_ampstep):
      field_coeffs, field_energy = step_fieldcoeffs_all(wavenumber=wavenumber,
                                                        amplitude=amplitude,
                                                    field_coeffs = field_coeffs, field_energy=field_energy,
                                                        field_max_stepsize= field_max_stepsize)
    amplitude, field_energy, surface_energy = step_amplitude( wavenumber=wavenumber, kappa=kappa,
                                                             amplitude=amplitude, field_coeffs=field_coeffs,
                                                             field_energy=field_energy, surface_energy=surface_energy)
  return field_coeffs, abs(amplitude)



def loop_wavenumber_kappa( wavenumber_range, kappa_range, amp_steps, converge_stop,
                           fieldsteps_per_ampstep):
  # TODO: maybe convergence check
  converge_times = []
  results =[]
  for wavenumber in wavenumber_range:
    results_line = []
    times_line = []
    for kappa in kappa_range:
      #reset to starting values 0 for each run
      amplitude = 0
      initial_field_coeffs = dict([(i, 0+0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
      #run
      field_coeffs, amplitude= run(field_coeffs=initial_field_coeffs,
                                   wavenumber=wavenumber, field_max_stepsize=field_max_stepsize,
                                   kappa=kappa, amplitude=amplitude,
                                   amp_steps=amp_steps, fieldsteps_per_ampstep=fieldsteps_per_ampstep)
      results_line.append(amplitude)
      print(kappa, wavenumber, amplitude)
    results.append(results_line)
    converge_times.append(times_line)
  return (np.array(converge_times), np.array(results))

def plot_save(wavenumber_range, kappa_range, results, title):
  df = pd.DataFrame(index= wavenumber_range, columns=kappa_range, data=results)
  df.to_csv(title+".csv")
  plt.imshow(results,extent=[min(kappa_range), max(kappa_range), max(wavenumber_range), min(wavenumber_range)])
  plt.colorbar()
  plt.savefig(title+".png")
  plt.close()

def record_amplitude_vs_time(temp, kappa, wavenumber, amp_steps, fieldsteps_per_ampstep):
  """
  for examining single run
  :param temp:
  :param kappa:
  :param wavenumber:
  :param amp_steps:
  :param fieldsteps_per_ampstep:
  :return:
  """
  initial_field_coeffs = dict([(i, 0 + 0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
  initial_amplitude=0
  amplitudes = [initial_amplitude]
  field_coeffs = initial_field_coeffs
  amplitude = initial_amplitude
  field_energy = ce.calc_field_energy(field_coeffs, initial_amplitude, wavenumber, kappa)
  surface_energy = ce.calc_surface_energy(initial_amplitude, wavenumber, kappa, amplitude_change=False )
  for i in range(amp_steps):
    if abs(amplitude) > 1: break
    for j in range(fieldsteps_per_ampstep):
      step_fieldcoeffs_all(temp, wavenumber=wavenumber)
    step_amplitude(temp, wavenumber=wavenumber, kappa=kappa)
    amplitudes.append(amplitude)
  return amplitudes

def run_experiment(type, experiment_title, range1, range2, amp_steps, converge_stop, fieldsteps_per_ampstep, plot=True):
  # TODO: lookup from type, decide on loop function
  converge_time, results = loop_wavenumber_kappa(wavenumber_range=range1, kappa_range=range2, amp_steps=amp_steps, converge_stop=converge_stop
                                                 ,fieldsteps_per_ampstep=fieldsteps_per_ampstep)
  if plot:
    plot_save(wavenumber_range=range1, kappa_range=range2, results=results, title=experiment_title)
    plot_save(wavenumber_range=range1, kappa_range=range2, results=converge_time,
              title=experiment_title + "_convergence_time_")
  # TODO: save data
  print(converge_time, results)

#TODO: how to set system variables?
  # coefficients
alpha=-1
C=1
u=1
n = 1
kappa = 1
gamma=1
temp=0.001

  # system dimensions
radius = 1
wavenumber = 1

  #simulation details
num_field_coeffs = 3
amplitude_max_stepsize =.05
field_max_stepsize =.05

if __name__ == "__main__":

  #specify type, range of plot; title of experiment
  loop_type = ("wavenumber", "kappa")
  experiment_title =  loop_type[0]+"_"+loop_type[1]
  range1 = np.arange(0.005, 1.2, .05)
  range2 = np.arange(0,1, .05)
  amp_steps = 500
  converge_stop = True
  fieldsteps_per_ampstep= 1

  run_experiment(loop_type, experiment_title,
                 range1, range2, amp_steps, converge_stop, fieldsteps_per_ampstep)

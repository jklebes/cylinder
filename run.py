import copy
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calc_energy as ce

def step_fieldcoeffs(temp, wavenumber, field_coeffs, field_energy, amplitude):
  # TODO: do all, but in a random order?
  field_coeff_index = random.randint(-1*num_field_coeffs, num_field_coeffs)
  proposed_field_coeffs =copy.copy(field_coeffs)
  proposed_field_coeffs[field_coeff_index]+=field_max_stepsize*complex(random.uniform(-1,1), random.uniform(-1,1))

  new_field_energy = ce.calc_field_energy(proposed_field_coeffs, amplitude, wavenumber,  radius, n=n,
                                          alpha=alpha, C=C, u=u, amplitude_change=False)
  diff = new_field_energy - field_energy

  # TODO: look at this logic
  if diff <= 0 or (temp!=0 and random.uniform(0, 1) <= math.exp(-diff / temp)):
    #accept change
    field_energy=new_field_energy
    field_coeffs= proposed_field_coeffs
  return field_coeffs, field_energy

def step_amplitude(temp, wavenumber, kappa, amplitude, field_coeffs, surface_energy, field_energy):
  proposed_amplitude = amplitude + amplitude_max_stepsize*random.uniform(-1,1)
  new_field_energy = ce.calc_field_energy(field_coeffs, proposed_amplitude,wavenumber=wavenumber, radius=radius, n=n,
                                          alpha=alpha, C=C, u=u)
  new_surface_energy=ce.calc_surface_energy(proposed_amplitude, wavenumber=wavenumber, radius=radius,
                                            kappa=kappa, gamma=gamma, amplitude_change=False)
  diff = (new_field_energy + new_surface_energy) - (field_energy + surface_energy)
  # TODO: check this logic
  if temp==0:
    probability = 1 if diff<=0 else 1 #choice was made that 0 difference -> accept change
  else:
    probability = min(math.exp(-diff / temp), 1)
  assert probability >= 0
  assert probability <= 1
  if random.uniform(0, 1) <= probability:
      # accept change
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
  return amplitude, surface_energy, field_energy

#one run
#loop over this for statistics or to try idffernt system parameters
def run(temp, initial_field_coeffs, wavenumber, kappa, initial_amplitude=0, amp_steps=100, fieldsteps_per_ampstep=10): #constants for this run
  field_coeffs = initial_field_coeffs
  amplitude = initial_amplitude
  field_energy = ce.calc_field_energy(field_coeffs, initial_amplitude, wavenumber, radius=radius, n=n,
                                      alpha=alpha, C=C, u=u)
  surface_energy = ce.calc_surface_energy(initial_amplitude, wavenumber, radius=radius, kappa=kappa, gamma=gamma,
                                          amplitude_change=False )
  for i in range(amp_steps):
    if abs(amplitude) > 1: break
    for j in range(fieldsteps_per_ampstep):
      field_coeffs, field_energy = step_fieldcoeffs(temp, wavenumber=wavenumber, amplitude=amplitude,
                                                    field_coeffs = field_coeffs, field_energy=field_energy)
    amplitude, field_energy, surface_energy = step_amplitude(temp, wavenumber=wavenumber, kappa=kappa,
                                                             amplitude=amplitude, field_coeffs=field_coeffs,
                                                             field_energy=field_energy, surface_energy=surface_energy)
  return field_coeffs, abs(amplitude)



def loop_wavenumber_kappa(temp, wavenumber_range, kappa_range, amp_steps, fieldsteps_per_ampstep):
  # TODO: maybe convergence check
  converge_times = []
  results =[]
  for wavenumber in wavenumber_range:
    results_line = []
    times_line = []
    for kappa in kappa_range:
      initial_field_coeffs = field_coeffs = dict([(i, 0+0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
      field_coeffs, amplitude= run(temp, initial_field_coeffs, wavenumber, kappa,
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
  global field_energy
  global surface_energy
  global field_coeffs
  global amplitude
  field_coeffs = initial_field_coeffs
  amplitude = initial_amplitude
  field_energy = ce.calc_field_energy(field_coeffs, initial_amplitude, wavenumber, kappa)
  surface_energy = ce.calc_surface_energy(initial_amplitude, wavenumber, kappa, amplitude_change=False )
  for i in range(amp_steps):
    if abs(amplitude) > 1: break
    for j in range(fieldsteps_per_ampstep):
      step_fieldcoeffs(temp, wavenumber=wavenumber)
    step_amplitude(temp, wavenumber=wavenumber, kappa=kappa)
    amplitudes.append(amplitude)
  return amplitudes

def run_experiment(type, range1, range2, amp_steps, converge_stop, fieldsteps_per_ampstep, temp, plot=True):
  # TODO: lookup from type, decide on loop function
  converge_time, results =  loop_wavenumber_kappa(temp=temp, wavenumber_range=range1, kappa_range=range2, amp_steps=amp_steps
                        , fieldsteps_per_ampstep=fieldsteps_per_ampstep)
  if plot:
    title = "wavenumber_kappa_alpha0u0c0_tmp"
    plot_save(wavenumber_range=range1, kappa_range=range2, results=results, title=title)
    plot_save(wavenumber_range=range1, kappa_range=range2, results=converge_time,
              title=title + "_convergence_time_")


if __name__ == "__main__":
  # set system variables here
  global alpha
  global C
  global u
  global gamma
  global radius
  global amplitude_max_stepsize
  global field_max_stepsize
  global n
  alpha=0
  C=0
  u=0
  n = 1
  gamma=1
  radius = 1
  num_field_coeffs = 3
  amplitude_max_stepsize =.05
  field_max_stepsize =.05
  temp=0.001

  #specify type, range of plot; title of experiment
  loop_type = ("wavenumber", "kappa")
  experiment_title =  loop_type[0]+"_"+loop_type[1]
  range1 = np.arange(0.005, 1.2, .05)
  range2 = np.arange(0,1, .05)
  amp_steps = 500
  converge_stop = True
  fieldsteps_per_ampstep=10
  run_experiment(loop_type, range1, range2, amp_steps, converge_stop, fieldsteps_per_ampstep, temp=temp)

import copy
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calc_energy as ce

def step_fieldcoeffs(temp, wavenumber):
  global field_energy
  global field_coeffs
  field_coeff_index = random.randint(-1*num_field_coeffs, num_field_coeffs)
  proposed_field_coeffs =copy.copy(field_coeffs)
  proposed_field_coeffs[field_coeff_index]+=field_max_stepsize*complex(random.uniform(-1,1), random.uniform(-1,1))
  #print(field_coeff_index, proposed_field_coeffs)
  new_field_energy = ce.calc_field_energy(proposed_field_coeffs, amplitude, wavenumber,  amplitude_change=False)
  #print(new_field_energy, field_energy)
  diff = new_field_energy - field_energy
  if diff <= 0 or (temp!=0 and random.uniform(0, 1) <= math.exp(-diff / temp)):
    #accept change
    field_energy=new_field_energy
    field_coeffs= proposed_field_coeffs
  #except ZeroDivisionError: #if temperature is 0
   # if diff <=0:
    #  # accept change
     # field_energy = new_field_energy
      #field_coeffs = proposed_field_coeffs

def step_amplitude(temp, wavenumber, kappa):
  global field_energy
  global surface_energy
  global amplitude
  proposed_amplitude = amplitude + amplitude_max_stepsize*random.uniform(-1,1)
  new_field_energy = ce.calc_field_energy(field_coeffs, proposed_amplitude, wavenumber=wavenumber)
  new_surface_energy=ce.calc_surface_energy(proposed_amplitude, wavenumber=wavenumber, kappa=kappa,amplitude_change=False)
  diff = (new_field_energy + new_surface_energy) - (field_energy + surface_energy)
  if diff <= 0 or (temp!=0 and random.uniform(0, 1) <= math.exp(-diff / temp)):
      # accept change
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
      #print(amplitude, field_energy, surface_energy)
  #except ZeroDivisionError:  # if temperature is 0 or diff is large negative (proposed new energy is much less)
    #if diff <= 0:
      # accept change
      #field_energy = new_field_energy
      #surface_energy = new_surface_energy
      #amplitude = proposed_amplitude

#one run
#loop over this for statistics or to try idffernt system parameters
def run(temp, initial_field_coeffs, wavenumber, kappa, initial_amplitude=0, amp_steps=100, fieldsteps_per_ampstep=10): #constants for this run
  global field_energy
  global surface_energy
  global field_coeffs
  global amplitude
  field_coeffs = initial_field_coeffs
  amplitude = initial_amplitude
  field_energy = ce.calc_field_energy(field_coeffs, initial_amplitude, wavenumber, kappa)
  surface_energy = ce.calc_surface_energy(initial_amplitude, wavenumber, kappa, amplitude_change=False, )
  for i in range(amp_steps):
    if abs(amplitude) > 1: break
    for j in range(fieldsteps_per_ampstep):
      step_fieldcoeffs(temp, wavenumber=wavenumber)
    step_amplitude(temp, wavenumber=wavenumber, kappa=kappa)
  return field_coeffs, abs(amplitude)

#one run
#loop over this for statistics or to try idffernt system parameters
def run_amp_converge(temp, initial_field_coeffs, wavenumber, kappa, initial_amplitude=0,
                     amp_steps=500, fieldsteps_per_ampstep=10, converged_tol = .05, converged_n=50): #constants for this run
  global field_energy
  global surface_energy
  global field_coeffs
  global amplitude
  field_coeffs = initial_field_coeffs
  amplitude = initial_amplitude
  field_energy = ce.calc_field_energy(field_coeffs, initial_amplitude, wavenumber, kappa)
  surface_energy = ce.calc_surface_energy(initial_amplitude, wavenumber, kappa, amplitude_change=False, )
  converged_count=0
  past_values = [amplitude]
  for i in range(amp_steps):
    #print("amplitude", amplitude)
    if abs(amplitude) > 1 or converged_count >= converged_n :
      if converged_count >= converged_n :
        print("converged on ", amplitude, " after ", i, " steps ")
      break
    #print("no break")
    for j in range(fieldsteps_per_ampstep):
      step_fieldcoeffs(temp, wavenumber=wavenumber)
    step_amplitude(temp, wavenumber=wavenumber, kappa=kappa)
    if abs(sum(past_values)/len(past_values)-amplitude)<converged_tol:
      converged_count +=1
    else:
      converged_count = 0
    past_values.append(amplitude)
    if len(past_values) > converged_n:
      past_values.pop(0)
  return i, abs(sum(past_values)/len(past_values))

def loop_wavenumber_kappa(temp, wavenumber_range, kappa_range, amp_steps=1000, fieldsteps_per_ampstep=100, amp_converge=True):
  converge_times = []
  results =[]
  for wavenumber in wavenumber_range:
    results_line = []
    times_line = []
    for kappa in kappa_range:
      initial_field_coeffs = field_coeffs = dict([(i, 0+0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
      if amp_converge:
        time, amplitude = run_amp_converge(temp, initial_field_coeffs, wavenumber, kappa,
                                      amp_steps=amp_steps, fieldsteps_per_ampstep=fieldsteps_per_ampstep)
        times_line.append(time)
      else:
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

def record_amplitude_vs_time(temp, kappa, wavenumber, amp_steps=1000, fieldsteps_per_ampstep=5):
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

if __name__ == "__main__":
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

  wavenumber_range = np.arange(0.005, 1.2, .05)
  kappa_range = np.arange(0,1, .05)
  temp=0.001
  converge_time, results = loop_wavenumber_kappa(temp=temp, wavenumber_range=wavenumber_range, kappa_range=kappa_range)
  print(results)
  #wavenumber = .1
  #kappa=.3
  #amplitudes = record_amplitude_vs_time(temp, wavenumber=wavenumber, kappa=kappa)
  title="wavenumber_kappa_alpha0u0c0_tmp"
  plot_save(wavenumber_range=wavenumber_range, kappa_range=kappa_range, results=results, title=title)
  plot_save(wavenumber_range=wavenumber_range, kappa_range=kappa_range, results=converge_time, title=title+"_convergence_time_")
  #plot_save(wavenumber_range, kappa_range, results, title)
  #plt.scatter(range(len(amplitudes)), amplitudes, marker='x')
  #plt.xlabel="steps"
  #plt.ylabel("a")
  #plt.savefig(title+".png")
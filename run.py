import copy
import math
import random
import pandas as pd
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

A_integrals = dict()
B_integrals = dict()

def sqrt_g_theta(radius, amplitude, wavenumber, z):
  return radius * (1+ amplitude*math.sin(wavenumber*z))

def sqrt_g_z(radius, amplitude, wavenumber, z):
  return math.sqrt(1+ (radius * amplitude * wavenumber * math.cos(wavenumber*z))**2)

def radius_rescale_factor(amplitude):
  return (1+amplitude**2/2)**(-0.5)

def A_integrand_img_part(diff, amplitude, z,wavenumber):
  """
  :param diff: coefficient difference (i-j) or (i+i'-j-j') in e^i(i-j)kz 
  :param amplitude: a
  :return: float
  """
  if amplitude==0:
    return (math.sin(diff * wavenumber* z) *#img part of e^(i ...)
                          radius ) #sqrt(g_theta theta) = radius
                          # radius rescale factor = 1
                          # and sqrt(g_zz) =1
  else:
    return (math.sin(diff * wavenumber * z) *  # real part of e^(i ...)
            sqrt_g_theta(radius, amplitude, wavenumber, z) *  # sqrt(g_theta theta)
            radius_rescale_factor(amplitude)*  # r(a) adjustment for volume conservation
            sqrt_g_z(radius, amplitude, wavenumber, z)) # and sqrt(g_zz)

def A_integrand_real_part(diff, amplitude, z, wavenumber):
  """
  :param diff: coefficient difference (i-j) or (i+i'-j-j') in e^i(i-j)kz 
  :param amplitude: a
  :return: float
  """
  if amplitude==0:
    return (math.cos( diff * wavenumber* z) *#real part of e^(i ...)
                          radius ) #sqrt(g_theta theta) = radius
                          # r(a) due to volume conservation = r
                          # and sqrt(g_zz) =0
  else:
    return (math.cos(  diff * wavenumber* z) *#real part of e^(i ...)
            sqrt_g_theta(radius, amplitude, wavenumber, z) * #sqrt(g_theta theta)
            radius_rescale_factor(amplitude)* # r(a) adjustment for volume conservation
            sqrt_g_z(radius, amplitude, wavenumber, z))# and sqrt(g_zz)


def B_integrand_img_part(i,j, amplitude, z, wavenumber):
  if amplitude == 0:
    z_part = (i * j * wavenumber ** 2 * math.sin((i - j) * wavenumber * z) *  # |d e^... |^2
              radius)  # sqrt(g_theta theta) = radius
    # r(a) rescale due to volume conservation = *1
    # and 1/sqrt(g_zz) =1
    return (z_part)
  else:
    z_part = (i * j * wavenumber ** 2 * math.sin((i - j) * wavenumber * z) *  # |d e^... |^2
              sqrt_g_theta(radius, amplitude, wavenumber, z) *  # sqrt(g_theta theta)
              radius_rescale_factor(amplitude) *  # r(a) due to volume conservation
              sqrt_g_z(radius, amplitude, wavenumber, z))  # and 1/sqrt(g_zz)
    return (z_part)


def B_integrand_real_part(i, j , amplitude, z, wavenumber):
  if amplitude == 0:
    z_part = (i * j * wavenumber ** 2 * math.cos((i - j) * wavenumber * z) *  # |d e^... |^2
              radius)  # sqrt(g_theta theta) = radius
              # r(a) due to volume conservation
              # and 1/sqrt(g_zz) =1
    return(z_part)
  else:
    z_part = (i * j * wavenumber ** 2 * math.cos((i - j) * wavenumber * z) *  # |d e^... |^2
              sqrt_g_theta(radius, amplitude, wavenumber, z) *  # sqrt(g_theta theta)
              radius_rescale_factor(amplitude) *  # r(a) due to volume conservation
              1/sqrt_g_z(radius, amplitude, wavenumber, z))  # and 1/sqrt(g_zz) from metric ddeterminant, index raising g^zz
    theta_part = ((n*wavenumber*radius*amplitude*math.cos(wavenumber*z))**2 * #part of A_theta^2
                  1.0/sqrt_g_z(radius, amplitude, wavenumber, z) * #sqrt(g_zz) annd with 1/g_zz normalizing A_theta^2
                  1.0/sqrt_g_theta(radius, amplitude, wavenumber, z)* #sqrt(g_thth) and g^thth
                  radius_rescale_factor(amplitude)) # two from r^2 in A_th, -2 from g^thth, 1 from sqrt_g_thth
    return (z_part + theta_part)

def evaluate_A_integrals(amplitude, wavenumber):
  global A_integrals
  for diff in range(-4*num_field_coeffs, 4*num_field_coeffs+1):
    img_part, error = integrate.quad(lambda z: A_integrand_img_part(diff, amplitude, z, wavenumber=wavenumber),
                              0, 2 *math.pi / wavenumber)
    real_part, error = integrate.quad(lambda z: A_integrand_real_part(diff, amplitude, z, wavenumber=wavenumber),
                               0, 2 *math.pi / wavenumber)
    A_integrals[diff]= complex(real_part, img_part)

def evaluate_A_integral_0(amplitude, wavenumber):
  global A_integrals
  img_part, error = integrate.quad(lambda z: A_integrand_img_part(0, amplitude, z, wavenumber=wavenumber),
                              0, 2 *math.pi / wavenumber)
  if not math.isclose(img_part, 0, abs_tol=1e-9):
    print("surface area = A[diff=0] with imaginary component", img_part)
  real_part, error = integrate.quad(lambda z: A_integrand_real_part(0, amplitude, z, wavenumber=wavenumber),
                               0, 2 *math.pi / wavenumber)
  A_integrals[0]= complex(real_part, img_part)

def evaluate_B_integrals(amplitude, wavenumber):
  global B_integrals
  B_integrals = dict()
  for i in range(-1*num_field_coeffs, num_field_coeffs+1):
    for j in range(-1 * num_field_coeffs, num_field_coeffs + 1):
      img_part, error = integrate.quad(lambda z: B_integrand_img_part(i, j , amplitude, z, wavenumber=wavenumber),
                                       0, 2 * math.pi / wavenumber)
      real_part, error = integrate.quad(lambda z: B_integrand_real_part(i, j , amplitude, z, wavenumber=wavenumber),
                                        0, 2 * math.pi / wavenumber) #throws integrationwarning for amplitude >.9
      B_integrals[(i,j)] = complex(real_part, img_part)


def calc_field_energy(field_coeffs, amplitude, wavenumber, amplitude_change=True):
  #some memoization: repurpose A integrals as D integrals
  # and don't reevaluate until amplitude change
  if not A_integrals or amplitude_change:
    #print("amplitude", amplitude)
    evaluate_A_integrals(amplitude, wavenumber=wavenumber)
    evaluate_B_integrals(amplitude, wavenumber=wavenumber)
    #print("A integrals", A_integrals)
    #print("B integrals", B_integrals)
  A_complex_energy = 0+0j
  for i in range(-1 * num_field_coeffs, num_field_coeffs + 1):
    for j in range(-1 * num_field_coeffs, num_field_coeffs + 1):
      A_complex_energy+= (field_coeffs[i]*field_coeffs[j].conjugate()*A_integrals[i-j])
  if not math.isclose(A_complex_energy.imag, 0, abs_tol=1e-9):
    print("nonzero imaginary energy component in A", A_complex_energy)
  B_complex_energy = 0 + 0j
  for i in range(-1 * num_field_coeffs, num_field_coeffs + 1):
      for j in range(-1 * num_field_coeffs, num_field_coeffs + 1):
        B_complex_energy += (field_coeffs[i] * field_coeffs[j].conjugate() * B_integrals[(i,j)])
  if not math.isclose(B_complex_energy.imag, 0, abs_tol=1e-9):
    print("nonzero imaginary energy component in B", B_complex_energy)
  #same reduce / list comprehension as loops this time because there are 4 variables
  D_complex_energy = 0+0j #identity of complex sum
  for i1 in range(-1 * num_field_coeffs, num_field_coeffs + 1):
    for i2 in range(-1 * num_field_coeffs, num_field_coeffs + 1):
      for j1 in range(-1 * num_field_coeffs, num_field_coeffs + 1):
        for j2 in range(-1 * num_field_coeffs, num_field_coeffs + 1):
          D_complex_energy += (field_coeffs[i1]*field_coeffs[i2]*
                                field_coeffs[j1].conjugate()*field_coeffs[j2].conjugate()*
                                                A_integrals[(i1+i2-j1-j2)])
  if not math.isclose(D_complex_energy.imag, 0, abs_tol=1e-8):
    print("nonzero imaginary energy component in D", D_complex_energy)
  return alpha*A_complex_energy.real + C * B_complex_energy.real + 0.5*u*D_complex_energy.real

def Kzz_integrand(amplitude, z, wavenumber):
  return( (amplitude * wavenumber**2 * math.sin(wavenumber*z))**2 *
          radius**3 * (1+amplitude * math.sin(wavenumber*z)) *
          (1+amplitude**2/2)**(-3/2.0)*
          1/(1+(amplitude*wavenumber*math.cos(wavenumber*z))**2))

def Kthth_integrand(amplitude, z, wavenumber):
  return( 1/((radius )**2)* #part of K_th^th^2
          1/sqrt_g_z(radius, amplitude, wavenumber, z)*# part of K_th^th^2*sqrt_g_zz
          sqrt_g_theta(radius, amplitude, wavenumber, z)*
          1/radius_rescale_factor(amplitude)) # one radius in sqrt_g_theta and -2 in Kthth


def calc_bending_energy(amplitude, wavenumber):
  if amplitude==0:
    Kthth_integral, error= integrate.quad(lambda z: 1.0/radius**2,
                                       0, 2 * math.pi / wavenumber)
    return Kthth_integral
  else:
    Kzz_integral, error = integrate.quad(lambda z: Kzz_integrand(amplitude, z, wavenumber=wavenumber),
                                       0, 2 * math.pi / wavenumber)
    Kthth_integral, error= integrate.quad(lambda z: Kthth_integrand( amplitude, z, wavenumber=wavenumber),
                                       0, 2 * math.pi / wavenumber)
    return(Kzz_integral+Kthth_integral)

def calc_surface_energy(amplitude, wavenumber, kappa, amplitude_change=True):
  if (not A_integrals) or amplitude_change:
    evaluate_A_integral_0(amplitude)
  #print('surface area', A_integrals[0].real, "a", amplitude)
  return gamma* A_integrals[0].real + kappa*calc_bending_energy(amplitude, wavenumber)

def step_fieldcoeffs(temp, wavenumber):
  global field_energy
  global field_coeffs
  field_coeff_index = random.randint(-1*num_field_coeffs, num_field_coeffs)
  proposed_field_coeffs =copy.copy(field_coeffs)
  proposed_field_coeffs[field_coeff_index]+=field_max_stepsize*complex(random.uniform(-1,1), random.uniform(-1,1))
  #print(field_coeff_index, proposed_field_coeffs)
  new_field_energy = calc_field_energy(proposed_field_coeffs, amplitude, wavenumber,  amplitude_change=False)
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
  new_field_energy = calc_field_energy(field_coeffs, proposed_amplitude, wavenumber=wavenumber)
  new_surface_energy=calc_surface_energy(proposed_amplitude, wavenumber=wavenumber, kappa=kappa,amplitude_change=False)
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
  field_energy = calc_field_energy(field_coeffs, initial_amplitude, wavenumber, kappa)
  surface_energy = calc_surface_energy(initial_amplitude, wavenumber, kappa, amplitude_change=False, )
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
  field_energy = calc_field_energy(field_coeffs, initial_amplitude, wavenumber, kappa)
  surface_energy = calc_surface_energy(initial_amplitude, wavenumber, kappa, amplitude_change=False, )
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
    #print('avg',sum(past_values)/len(past_values), amplitude)
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
  field_energy = calc_field_energy(field_coeffs, initial_amplitude, wavenumber, kappa)
  surface_energy = calc_surface_energy(initial_amplitude, wavenumber, kappa, amplitude_change=False )
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
  global num_field_coeffs
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
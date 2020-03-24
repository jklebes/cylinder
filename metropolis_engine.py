import cmath
import copy
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import system as ce

class MetropolisEngine():
  def __init__(self, method, num_field_coeffs, sampling_dist, sampling_widths, temp):
    self.method = method
    self.sampling_dist = sampling_dist
    self.num_field_coeffs = num_field_coeffs
    if isinstance(sampling_widths, float) or isinstance(sampling_widths, int):
      self.sampling_width_amplitude = sampling_widths
      self.sampling_width_coeffs = dict([(i, sampling_widths) for i in range(-num_field_coeffs, num_field_coeffs+1)])
    else:
      self.sampling_width_amplitude = sampling_widths[0]
      self.sampling_width_coeffs = sampling_widths[1]
    self.temp=temp

    self.acceptance_rate = None # to be calculated
  
  def step_fieldcoeffs_sequential(self, wavenumber, amplitude, field_coeffs, field_energy, surface_energy, 
                                  system):
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
    :param system:
    :return:
    """
    indices_randomized = [*field_coeffs]  # unpack iterable into list - like list() but faster
    # TODO : find out whether randomizing the order is helpful
    random.shuffle(indices_randomized)
    for index in indices_randomized:
      field_coeffs, field_energy = self.step_fieldcoeff(index, wavenumber,
                                                   field_coeffs, field_energy, surface_energy, amplitude,
                                                   system)
    print(field_energy, field_coeffs)
    return field_coeffs, field_energy


  def step_fieldcoeff(self, field_coeff_index, wavenumber,
                      field_coeffs, field_energy, surface_energy, amplitude, system):
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
    proposed_field_coeff = field_coeffs[field_coeff_index] + self.gaussian_complex(self.sampling_width_coeffs[field_coeff_index])
    new_field_energy = system.calc_field_energy_diff(field_coeff_index, proposed_field_coeff, field_coeffs,
                                                            amplitude,  amplitude_change=False)
    if metropolis_decision(temp, field_energy + surface_energy, new_field_energy + surface_energy):
      field_energy = new_field_energy
      field_coeffs[field_coeff_index] = proposed_field_coeff
    return field_coeffs, field_energy

  def step_amplitude(self, amplitude, field_coeffs, surface_energy, field_energy, system):
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
    proposed_amplitude = amplitude + self.sampling_dist(amplitude_sampling_width )
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

  def step_all(self, amplitude, field_coeffs, surface_energy, field_energy, system):
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
    proposed_amplitude = amplitude + self.sampling_dist(self.sampling_width_amplitude)
    proposed_field_coeffs = copy.copy(field_coeffs)
    for index in field_coeffs:
      proposed_field_coeffs[index] += self.gaussian_complex(self.sampling_width_coeffs[index])
    if abs(proposed_amplitude) >= 1:
      return amplitude, field_coeffs, surface_energy, field_energy
    new_field_energy = system.calc_field_energy(proposed_field_coeffs, proposed_amplitude, 
                                                       amplitude_change=True)
    new_surface_energy = system.calc_surface_energy(proposed_amplitude, amplitude_change=False)

    if self.metropolis_decision((field_energy + surface_energy), (new_field_energy + new_surface_energy)):
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
      field_coeffs = proposed_field_coeffs
    return amplitude, field_coeffs, surface_energy, field_energy

  def metropolis_decision(self, old_energy, proposed_energy):
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
    elif diff > 0 and self.temp == 0:
      return False
    else:
      probability = math.exp(- 1 * diff / self.temp)
      assert probability >= 0
      assert probability <= 1
      if random.uniform(0, 1) <= probability:
        return True
      else:
        return False

  def gaussian_complex(self, sigma):
    amplitude = random.gauss(0, sigma)
    phase = random.uniform(0, 2*math.pi)
    return cmath.rect(amplitude, phase)
  
  def set_temperature(self, new_temp):
    self.temp=new_temp

import cmath
import copy
import math
import random
import numpy as np

class MetropolisEngine():
  """
  base version: separate fixed proposal distribution widths for amplitude of fluctuation, magnitude of field coeffs
  draws steps from n separate gaussian (or other) distributions.
  Any constant (in run time), global (in parameter space) proposal distribution definitely makes sampling ergodic.
  """
  def __init__(self, num_field_coeffs, sampling_width=0.05, temp=0):
    self.temp=temp
    if isinstance(sampling_width, float) or isinstance(sampling_width, int):
      self.sampling_width_amplitude = sampling_width
      self.sampling_width_field_coeffs = dict([(i, sampling_width) for i in range(-num_field_coeffs, num_field_coeffs+1)])
    elif isinstance(sampling_width, tuple) or isinstance(sampling_width, list):
      self.sampling_width_amplitude = sampling_width[0]
      self.sampling_width_field_coeffs = sampling_width[1]
    else:
      print("couldnt interpret sampling width argument of type ", type(sampling_width))
    self.accepted_counter =0

  def step_fieldcoeffs_sequential(self, amplitude, field_coeffs, field_energy, surface_energy, 
                                  system):
    """
    Try stepping each field coefficient c_i once, in a random order
    -> this is Gibbs sampling
    I chose sequential for greater acceptance rates
    Disadvantage: 2n+1 recalculations, of the products and sums in system_energy.calc_field_energy
     (not of numerical integration)
    :param field_coeffs:
    :param field_energy:
    :param surface_energy:
    :param amplitude:
    :param system:
    :return:
    """
    indices_randomized = [*field_coeffs]  # unpack iterable into list - like list() but faster
    # TODO : find out whether randomizing the order is helpful
    random.shuffle(indices_randomized)
    for index in indices_randomized:
      field_coeffs, field_energy = self.step_fieldcoeff(index, field_coeffs, field_energy, surface_energy, amplitude, system)
    print(field_energy, field_coeffs)
    return field_coeffs, field_energy


  def step_fieldcoeff(self, field_coeff_index, field_coeffs, field_energy, surface_energy, amplitude, system, amplitude_change=False):
    """
    Stepping a single field coefficient c_i: generate a random complex value.  Accept or reject.
    :param field_coeff_index:
    :param field_coeffs:
    :param field_energy:
    :param surface_energy:
    :param amplitude:
    :param system_energy:
    :param amplitude_change: this parameter is here isolated use of fct in unit testing.  default False should be enough for real use.
    :return:
    """
    proposed_field_coeff = field_coeffs[field_coeff_index] + self.gaussian_complex(self.sampling_width_field_coeffs[field_coeff_index])
    new_field_energy = system.calc_field_energy_diff(field_coeff_index, proposed_field_coeff, field_coeffs, amplitude, amplitude_change)
    if self.metropolis_decision(field_energy + surface_energy, new_field_energy + surface_energy):
      field_energy = new_field_energy
      field_coeffs[field_coeff_index] = proposed_field_coeff
    return field_coeffs, field_energy

  def step_amplitude(self, amplitude, field_coeffs, surface_energy, field_energy, system):
    """
    Stepping amplitude by metropolis algorithm.
    :param amplitude:
    :param field_coeffs:
    :param surface_energy:
    :param field_energy:
    :param system:
    :return:
    """
    proposed_amplitude = amplitude + self.sampling_dist(amplitude_sampling_width )
    if abs(proposed_amplitude) >= 1:
      # don't accept.
      # like an infinite energy barrier to self-intersection.
      # does not violate symmetric jump distribution, because this is like
      # an energy-landscape-based decision after generation
      return amplitude, surface_energy, field_energy
    new_field_energy = system.calc_field_energy(field_coeffs, proposed_amplitude, radius=radius, n=n, alpha=alpha,
                                                       C=C, u=u,
                                                       wavenumber=wavenumber)
    new_surface_energy = system.calc_surface_energy(proposed_amplitude, wavenumber=wavenumber, radius=radius,
                                                           gamma=gamma,
                                                           kappa=kappa, amplitude_change=False)
    if self.metropolis_decision(temp, (field_energy + surface_energy), (new_field_energy + new_surface_energy)):
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
    proposed_amplitude = amplitude + random.gauss(0, self.sampling_width_amplitude)
    if abs(proposed_amplitude) >= 1:
      return amplitude, field_coeffs, surface_energy, field_energy
    proposed_field_coeffs = copy.copy(field_coeffs)
    for key in proposed_field_coeffs:
      proposed_field_coeffs[key] += self.gaussian_complex(self.sampling_width_field_coeffs[key])
    #calculate energy of poprosed state 
    new_field_energy = system.calc_field_energy(proposed_field_coeffs, proposed_amplitude, amplitude_change=True)
    new_surface_energy = system.calc_surface_energy(proposed_amplitude, amplitude_change=False)
    if self.metropolis_decision((field_energy + surface_energy), (new_field_energy + new_surface_energy)):
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
      field_coeffs = proposed_field_coeffs
    #output system properties, energy
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
    assert(new_temp >= 0)
    self.temp=new_temp

class AdaptiveMetropolisEngine(MetropolisEngine):
  def __init__(self, initial_field_coeffs, initial_amplitude, initial_covariance_matrix=None, initial_sampling_width=.0001, temp=0):
    self.num_field_coeffs = max(initial_field_coeffs)
    self.temp=temp
    
    #adaptive scheme
    # amplitude (float) and field_coeffs (dict of complex numbers) rearranged to list 'state' for covariance matrix calculation
    self.step_counter =1
    self.accepted_counter =0
    self.param_space_dims = 2*len(initial_field_coeffs)+1 #perturbation amplitude, real and img parts of ci
    state=[initial_amplitude]
    state.extend([initial_field_coeffs[key].real for key in initial_field_coeffs])
    state.extend([initial_field_coeffs[key].imag for key in initial_field_coeffs])
    self.state = np.array(state)
    self.mean = self.state
    print("initialized", self.state, self.mean)

    if initial_covariance_matrix is None:
      #initialize plausible 
      #identity matrix
      self.initial_covariance_matrix = initial_sampling_width*np.identity(self.param_space_dims)
    else:
      self.initial_covariance_matrix=initial_covariance_matrix
    #dimensions match dimensions of paramter space
    #assert(self.initial_covariance_matrix.shape()==(self.param_space_dims, self.param_space_dims))
    #self.covariance_matrix will be constantly updated
    #while self.initial_convariance_matrix is a static version for gathering data in the first n steps
    self.covariance_matrix = copy.copy(self.initial_covariance_matrix)

# TODO: how to handle not using new covariance matrix for the first n steps

  def update_covariance_matrix(self):     
    """
    adaptie Metropoplis scheme after Haario, Saksman & Tamminen  2001
    """
    #add to covariance matrix calculation 
    small_number =0.001
    sd = 2.4**2/self.param_space_dims

    t=self.step_counter
    #update mean
    old_mean = copy.copy(self.mean)
    self.mean *= (t-1) /t 
    self.mean += self.state/t
    print("updated mean", self.mean)
    # eq (3) [Haario2001]
    self.covariance_matrix *= ((t-1)/t)
    #print("multiplied cov by",  ((t-1)/t))
    self.covariance_matrix += sd /(self.step_counter+1) *(t*np.outer(old_mean,old_mean) - (t+1)*np.outer(self.mean,self.mean) + np.outer(self.state,self.state) + small_number*np.identity(self.param_space_dims))
    #print("added", t*np.outer(old_mean,old_mean)[-1])
    print("cov")
    print(self.covariance_matrix)


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
    #draw from proposal dist: multivariate gaussian with runningly updated covariance matrix of system
    assert(amplitude == self.state[0])
    if self.step_counter < 300:
      covariance_matrix = self.initial_covariance_matrix
    else:
      covariance_matrix = self.covariance_matrix
    proposed_state =  np.random.multivariate_normal(self.state, covariance_matrix) #use self.state instead of taking in outside information
    #print("proposed state", proposed_state)
    #rearrange state to amplitude, field_coeffs dict for energy calculation
    proposed_amplitude = proposed_state[0]
    if abs(proposed_amplitude) >= 1:
      return amplitude, field_coeffs, surface_energy, field_energy
    proposed_field_coeffs = dict([(key, complex(re,im)) for (key,(re,im)) in zip(range(-self.num_field_coeffs, self.num_field_coeffs+1), zip((proposed_state[1:1+2*self.num_field_coeffs+1]),(proposed_state[2*self.num_field_coeffs+2:])))])
    #calculate energy of poposed state 
    new_field_energy = system.calc_field_energy(proposed_field_coeffs, proposed_amplitude, amplitude_change=True)
    new_surface_energy = system.calc_surface_energy(proposed_amplitude, amplitude_change=False)
    if self.metropolis_decision((field_energy + surface_energy), (new_field_energy + new_surface_energy)):
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
      field_coeffs = proposed_field_coeffs
      self.state=proposed_state
      self.accepted_counter +=1
      #print("changed state", self.state, "at step", self.step_counter)
      print("acceptance rate" , self.accepted_counter / self.step_counter)
    #update covariance matrix with new (bzw same) state
    self.step_counter+=1
    self.update_covariance_matrix()
    #output system properties, energy
    return amplitude, field_coeffs, surface_energy, field_energy


class AmplitudeAdaptiveMetropolisEngine(AdaptiveMetropolisEngine):
  def __init__(self, initial_field_coeffs, initial_amplitude, initial_covariance_matrix=None,
               initial_sampling_width=.0001, temp=0):
    self.num_field_coeffs = max(initial_field_coeffs)
    self.temp = temp

    # adaptive scheme
    self.step_counter = 1
    self.accepted_counter = 0
    self.param_space_dims = len(initial_field_coeffs) + 1  # perturbation amplitude, real and img parts of ci
    state = [initial_amplitude]
    state.extend([abs(initial_field_coeffs[key]) for key in initial_field_coeffs])
    self.state = np.array(state)
    self.mean = self.state
    print("initialized", self.state, self.mean)

    if initial_covariance_matrix is None:
      # initialize plausible
      # identity matrix
      self.initial_covariance_matrix = initial_sampling_width * np.identity(self.param_space_dims)
    else:
      self.initial_covariance_matrix = initial_covariance_matrix
    self.covariance_matrix = copy.copy(self.initial_covariance_matrix)

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
    # draw from proposal dist: multivariate gaussian with runningly updated covariance matrix of system
    assert (amplitude == self.state[0])
    if self.step_counter < 300:
      covariance_matrix = self.initial_covariance_matrix
    else:
      covariance_matrix = self.covariance_matrix
    state=[abs(amplitude)]
    state.extend([abs(field_coeffs[key]) for key in field_coeffs])
    state=np.array(state)
    proposed_state = np.random.multivariate_normal(state,
                                                   covariance_matrix)
    # rearrange state to amplitude, field_coeffs dict for energy calculation
    proposed_amplitude = proposed_state[0]
    if abs(proposed_amplitude) >= 1:
      return amplitude, field_coeffs, surface_energy, field_energy
    proposed_field_coeffs = dict([(key, cmath.rect(r, random.uniform(-math.pi, math.pi))) for (key, r) in
                                  zip(range(-self.num_field_coeffs, self.num_field_coeffs + 1),
                                      proposed_state[1:])])
    # calculate energy of proposed state
    new_field_energy = system.calc_field_energy(proposed_field_coeffs, proposed_amplitude, amplitude_change=True)
    new_surface_energy = system.calc_surface_energy(proposed_amplitude, amplitude_change=False)
    if self.metropolis_decision((field_energy + surface_energy), (new_field_energy + new_surface_energy)):
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
      field_coeffs = proposed_field_coeffs
      self.state = proposed_state
      self.accepted_counter += 1
      # print("changed state", self.state, "at step", self.step_counter)
      print("acceptance rate", self.accepted_counter / self.step_counter)
    # update covariance matrix with new (bzw same) state
    self.step_counter += 1
    self.update_covariance_matrix()
    # output system properties, energy
    return amplitude, field_coeffs, surface_energy, field_energy


class ComplexAdaptiveMetropolisEngine(AdaptiveMetropolisEngine):
  def __init__(self, initial_field_coeffs, initial_amplitude, initial_covariance_matrix=None,
               initial_sampling_width=.0001, temp=0):
    self.num_field_coeffs = max(initial_field_coeffs)
    self.temp = temp

    # adaptive scheme
    self.step_counter = 1
    self.accepted_counter = 0
    self.param_space_dims = len(initial_field_coeffs) + 1  # perturbation amplitude, real and img parts of ci
    state = [initial_amplitude]
    state.extend([abs(initial_field_coeffs[key]) for key in initial_field_coeffs])
    self.state = np.array(state)
    self.mean = self.state
    print("initialized", self.state, self.mean)

    if initial_covariance_matrix is None:
      # initialize plausible
      # identity matrix
      self.initial_covariance_matrix = initial_sampling_width * np.identity(self.param_space_dims)
    else:
      self.initial_covariance_matrix = initial_covariance_matrix
    self.covariance_matrix = copy.copy(self.initial_covariance_matrix)

  def multivariate_normal_complex(self, mean, covariance_matrix):


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
    # draw from proposal dist: multivariate gaussian with runningly updated covariance matrix of system
    assert (amplitude == self.state[0])
    if self.step_counter < 300:
      covariance_matrix = self.initial_covariance_matrix
    else:
      covariance_matrix = self.covariance_matrix
    state=[abs(amplitude)]
    state.extend([abs(field_coeffs[key]) for key in field_coeffs])
    state=np.array(state)
    # TODO : how to sample from complex covariance matrix?
    proposed_state = np.random.multivariate_normal(state,
                                                   covariance_matrix)
    # rearrange state to amplitude, field_coeffs dict for energy calculation
    proposed_amplitude = proposed_state[0]
    if abs(proposed_amplitude) >= 1:
      return amplitude, field_coeffs, surface_energy, field_energy
    proposed_field_coeffs = dict([(key, cmath.rect(r, random.uniform(-math.pi, math.pi))) for (key, r) in
                                  zip(range(-self.num_field_coeffs, self.num_field_coeffs + 1),
                                      proposed_state[1:])])
    # calculate energy of proposed state
    new_field_energy = system.calc_field_energy(proposed_field_coeffs, proposed_amplitude, amplitude_change=True)
    new_surface_energy = system.calc_surface_energy(proposed_amplitude, amplitude_change=False)
    if self.metropolis_decision((field_energy + surface_energy), (new_field_energy + new_surface_energy)):
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
      field_coeffs = proposed_field_coeffs
      self.state = proposed_state
      self.accepted_counter += 1
      # print("changed state", self.state, "at step", self.step_counter)
      print("acceptance rate", self.accepted_counter / self.step_counter)
    # update covariance matrix with new (bzw same) state
    self.step_counter += 1
    self.update_covariance_matrix()
    # output system properties, energy
    return amplitude, field_coeffs, surface_energy, field_energy

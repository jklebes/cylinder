import cmath
import copy
import math
import random
import numpy as np
import scipy.stats

class MetropolisEngine():
  """
  base version: proposal distribution is fixed in width and covariance.
  By default the covariance matrix of the multivariate proposal distribution is the dientity matrix, equivalent to drawing from n independent gaussian distribution.
  A different fixed covariance matrix can alo be chosen.
  draws steps from n separate gaussian (or other) distributions.
  Using just this base class with constant (in run time), global (in parameter space) proposal distribution guarantees ergodic sampling.
  Subclass StaticCovarianceAdaptiveMetropolisEngine is recommended minimum level of adaptiveness.
  """

  def __init__(self, num_field_coeffs, sampling_width=0.05, temp=0, covariance_matrix=None, target_acceptance=.3):
    """
    :param num_field_coeffs: number of field coefficient degrees of freedom.
    :param sampling_width: float, width of proposal distribution.  A factor scaling covariance matrix in multivariate gaussian proposal distribution.
    :param temp: system temperature.  Optional, default 0.
    :param covariance_matrix: initial or constant covariance matrix of multivariate gaussian proposal distribution. Optional, defaults to identity matrix. Should reflect correlations between the parameters if known.  Dimensions should match number of free parameters: in general number of field coefficients + 1.  2*number of field coefficients+1 Where phase/ampltitude or real/imag parts of field coefficients are treated as separate degrees of freedom.  complex entries in subclass ComplexAdaptiveMetropolisEngine.
    """
    self.temp=temp
    self.num_field_coeffs = num_field_coeffs
    max_field_coeffs = num_field_coeffs//2
    self.keys_ordered = list(range(-max_field_coeffs, max_field_coeffs+1))
    self.sampling_width = sampling_width
    self.param_space_dims = self.num_field_coeffs+1 #in more basic case perturbation amplitude, magnitudes only of ci.  subclasses may consider both magnitude and phase / both real and img parts of cis and increase this number.
    self.mean = np.zeros(self.param_space_dims)
    if covariance_matrix == None:
      self.covariance_matrix = np.identity(self.param_space_dims)
    else:
      self.covariance_matrix = covariance_matrix
    self.accepted_counter =1
    self.step_counter =1
    self.target_acceptance = target_acceptance

  def step_sequential(self, amplitude, field_coeffs, field_energy, surface_energy, 
                                  system):
    """
    Try stepping each field coefficient c_i once, in a random order
    -> this is Gibbs sampling
    I chose sequential for greater acceptance rates
    Not yet implemented.
    """
    amplitude = self.step_amplitude()
    indices_randomized = [*field_coeffs]  # unpack iterable into list - like list() but faster
    # TODO : find out whether randomizing the order is helpful
    random.shuffle(indices_randomized)
    for index in indices_randomized:
      field_coeffs, field_energy = self.step_fieldcoeff(index, field_coeffs, field_energy, surface_energy, amplitude, system)
    print(field_energy, field_coeffs)
    return amplitude, field_coeffs, surface_energy, field_energy


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
    covariance_matrix_entry = self.covariance_matrix[field_coeff_index, field_coeff_index] 
    field_coeff = field_coeffs[field_coeff_index]
    proposed_field_coeff = self.modify_phase(abs(field_coeff) + random.gauss(0, self.sampling_width * covariance_matrix_entry), cmath.phase(field_coeff))
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
    proposed_amplitude = draw_amplitude_from_proposal_distriution(amplitude)
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
      # TODO: how to handle separate acceptance rates?
    return amplitude, surface_energy, field_energy

  def step_all(self, amplitude, field_coeffs, surface_energy, field_energy, system):
    """
    Step all parameters (amplitude, field coeffs) simultaneously.  True metropolis algorithm.
    Generic to any proposal distribution and adaptive algorithm
    :param amplitude:
    :param field_coeffs:
    :param surface_energy:
    :param field_energy:
    :param system: object which holds functions for calculating energy.  May store pre-calculated values and shortcuts.
    :return: new state and energy in the form tuple (ampltiude, field_coeffs (dict), surface_energy, field_energy).  Identical to input parameters if step was rejected.  Also modifies step counter and acceptance rate counter.
    """
    proposed_amplitude, proposed_field_coeffs = self.draw_all_from_proposal_distribution(amplitude, field_coeffs)
    if abs(proposed_amplitude) >= 1:
      return amplitude, field_coeffs, surface_energy, field_energy
    new_field_energy = system.calc_field_energy(proposed_field_coeffs, proposed_amplitude, amplitude_change=True)
    new_surface_energy = system.calc_surface_energy(proposed_amplitude, amplitude_change=False)
    accept = self.metropolis_decision((field_energy + surface_energy), (new_field_energy + new_surface_energy))
    if accept:
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
      field_coeffs = proposed_field_coeffs
      self.accepted_counter +=1
    self.step_counter += 1
    self.update_proposal_distribution(accept, amplitude, field_coeffs)
    #output system properties, energy
    return amplitude, field_coeffs, surface_energy, field_energy

  def draw_field_coeffs_from_proposal_distribution(self,ield_coeffs):
    """implement later"""
    # TODO : generalize to multivariate gaussian
    proposed_field_coeffs = copy.copy(field_coeffs)
    for key in proposed_field_coeffs:
      proposed_field_coeffs[key] += self.gaussian_complex(self.sampling_width_field_coeffs[key])
    return proposed_field_coeffs

  def draw_amplitude_from_proposal_distriution(self,amplitude):
    """ implement later"""
    # TODO : genrealize to multivariate gaussian
    proposed_amplitude = amplitude + random.gauss(0, self.sampling_width_amplitude)
    return proposed_amplitude

  def draw_all_from_proposal_distribution(self,amplitude, field_coeffs):
    """
    draw from multivariate gaussian distribution with sampling_width * covariance matrix
    exactly equivlent to drawing from n independent gaussian distributions when covariance matrix is identity matrix
    :param amplitude: current amplitude
    :param field_coeffs: dict of current (complex) field coeff values
    In this implementation only ampltiude of perturbation and magnitude of field coeffs is modifified by adaptive metropolis algorithm with (possible adapted) step sizes and covariance matrix.  Phases of field coeffs are independently adjusted by fixed-width, uncoupled gaussian distribution around their current state at each step.
    """
    state = [amplitude]
    state.extend([abs(field_coeffs[key]) for key in field_coeffs])
    old_phases = [cmath.phase(field_coeffs[key]) for key in field_coeffs]
    proposed_addition =  np.random.multivariate_normal([0]*len(state), self.sampling_width*self.covariance_matrix)
    proposed_amplitude = state[0]+proposed_addition[0]
    proposed_field_coeffs = dict([(key, self.modify_phase(proposed_amplitude, old_phase)) for (key,(proposed_amplitude, old_phase)) in zip(self.keys_ordered, zip(proposed_addition[1:], old_phases))])
    return proposed_amplitude, proposed_field_coeffs
  
  def update_proposal_distribution(self, accept, amplitude, field_coeffs):
    """
    no change to sampling width or covariance matrix in base class.  override in subclasses
    :param accept: True or False outcome of accepting latest step or not
    """
    pass
  
  def metropolis_decision(self, old_energy, proposed_energy):
    """
    Considering energy difference and temperature, return decision to accept or reject step
    :param old_energy: current system total energy
    :param proposed_energy: system total energy after proposed change
    :return: bool True if step will be accepted; False to reject
    """
    diff = proposed_energy - old_energy
    if diff <= 0:
      return True  # choice was made that 0 difference -> accept change
    elif diff > 0 and self.temp == 0:
      return False
    else:
      print(self.temp)
      probability = math.exp(- 1 * diff / self.temp)
      assert probability >= 0
      assert probability <= 1
      if random.uniform(0, 1) <= probability:
        return True
      else:
        return False

  def modify_phase(self, amplitude, phase, phase_sigma = .5):
    """
    takes a random number and changes phase by gaussian proposal distribution
    :param phase_sigma: width of gaussian distribution to modify phase by.  Optional, default .5 (pi)
    """
    # TODO : pass option to set phase_sigma upwards
    new_phase = phase + random.gauss(0, phase_sigma)
    return cmath.rect(amplitude, new_phase)

  def random_complex(self, r):
    """
    a random complex number with a random (uniform between -pi and pi) phase and exactly the given ampltiude r
    :param r: magnitude of complex number
    :return: cmath complex number object in standard real, imaginary represenation
    """
    phi = random.uniform(-math.pi, math.pi)
    return cmath.rect(r, phi)

  def gaussian_complex(self, sigma=1):
    """
    a random complex number with completely random phase and amplitude drawn from a gaussian distribution with given width sigma (default 1)
    :param sigma: width of gaussian distribution (centered around 0) of possible magnitude of complex number
    :return: cmath complex number object in standard real, imaginary represenation 
    """
    amplitude = random.gauss(0, sigma)
    phase = random.uniform(0, 2*math.pi)
    return cmath.rect(amplitude, phase)
  
  def set_temperature(self, new_temp):
    """
    change metropolis engine temperature
    :param new_temp: new temperature value, float >=0 .
    """
    assert(new_temp >= 0)
    self.temp=new_temp


class StaticCovarianceAdaptiveMetropolisEngine(MetropolisEngine):
  """ Reasoning and algorithm from 
  Garthwaite, Fan, & Sisson 2010: arxiv.org/abs/1006.3690v1
  here first without updating covariance matrix - assuming covariance is approximately the identity matrix (default) indicating no correlations, or a covariance matrix from known correlations of system parameters is given.
  """

  def __init__(self,num_field_coeffs, sampling_width=0.05, temp=0, covariance_matrix = None):
    super().__init__(num_field_coeffs, sampling_width, temp, covariance_matrix)
    
    # alpha (a constant used later) := -phi^(-1)(target acceptance /2) 
    # ,where  phi is the cumulative density function of the standard normal distribution
    # norm.ppf ('percent point function') is the inverse of the cumulative density function of standard normal distribution
    self.alpha = -1*scipy.stats.norm.ppf(self.target_acceptance/2)
    self.m = self.param_space_dims

  def update_proposal_distribution(self, accept, amplitude, field_coeffs):
    step_number_factor = self.step_counter/self.m
    steplength_c = self.sampling_width * ((1-(1/self.m)) * math.sqrt(2*math.pi) * math.exp(self.alpha**2/2) / 2*self.alpha + 1/(self.m*self.target_acceptance*(1-self.target_acceptance)))
    if accept:
      self.sampling_width += steplength_c * (1-self.target_acceptance)/step_number_factor
    else:
      self.sampling_width -= steplength_c * self.target_acceptance / step_number_factor
  # TODO: test for convergence of sampling_width, c->0

class RobbinsMonroAdaptiveMetropolisEngine(StaticCovarianceAdaptiveMetropolisEngine):
  """ Reasoning and algorithm from 
  Garthwaite, Fan, & Sisson 2010: arxiv.org/abs/1006.3690v1
  without updating covariance matrix - assuming covariance is approximately identity matrix
  """

  def __init__(self, initial_field_coeffs, initial_amplitude=0, sampling_width=0.05, temp=0, method = "simultaneous", covariance_matrix = None):
    self.num_field_coeffs = len(initial_field_coeffs)
    super().__init__(self.num_field_coeffs, sampling_width, temp, covariance_matrix)
    #start collecting data for updating covariance matrix
    mean=[initial_amplitude]
    mean.extend([abs(initial_field_coeffs[key]) for key in self.keys_ordered])
    self.mean = np.array(mean)
    print(self.mean)

  def update_proposal_distribution(self, accept, amplitude, field_coeffs):
    #update sampling width
    super().update_proposal_distribution(accept, amplitude, field_coeffs)
    # TODO: update covarience matrix
    self.update_mean(amplitude, field_coeffs)

  def update_mean(self, amplitude, field_coeffs):
    state = [abs(amplitude)]
    state.extend([abs(field_coeffs[key]) for key in self.keys_ordered])
    state=np.array(state)
    self.mean *= (self.step_counter -1)/self.step_counter
    self.mean += state/self.step_counter

  # TODO: test for convergence of sampling_width, c->0

class FiniteAdaptiveMetropolisEngine(MetropolisEngine):
  def __init__(self,  num_field_coeffs, sampling_width=0.05, temp=0):
    # init superclass 
    self.param_space_dims = 2*len(initial_field_coeffs)+1 #perturbation amplitude, real and img parts of ci
    self.adaption_size = 1

  #how to handle running first n steps differently?
  
  def update_proposal_distribution(self, accept):
    if step_counter < n:
      if accept:
        self.sampling_width_amplitude += 0
        self.sampling_width_field_coeffs += 0
      else:
        pass
      self.adaption_size *= .08
    print("acceptance_rate", self.accept_counter / self.step_counter, "target", self.target)

class RealImgAdaptiveMetropolisEngine(MetropolisEngine):
  def __init__(self, initial_field_coeffs, initial_amplitude, initial_covariance_matrix=None, initial_sampling_width=.0001, temp=0):
    self.num_field_coeffs = max(initial_field_coeffs)
    self.temp=temp
    
    #adaptive scheme
    # amplitude (float) and field_coeffs (dict of complex numbers) rearranged to list 'state' for covariance matrix calculation
    self.step_counter =0
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
    sd = (2.4**2)/self.param_space_dims

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


class AmplitudePhaseAdaptiveMetropolisEngine(RobbinsMonroAdaptiveMetropolisEngine):
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


class ComplexAdaptiveMetropolisEngine(RobbinsMonroAdaptiveMetropolisEngine):
  def __init__(self, initial_field_coeffs, initial_amplitude =0 , sampling_width=.001, initial_covariance_matrix=None, temp=0):
    super().__init__(initial_field_coeffs, sampling_width, temp, initial_covariance_matrix)

  def multivariate_normal_complex(self, mean, covariance_matrix):
    pass


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

  def __init__(self, initial_field_coeffs, initial_amplitude=None, sampling_width=0.05, temp=0, covariance_matrix=None,
               target_acceptance=.3):
    """
    :param num_field_coeffs: number of field coefficient degrees of freedom.
    :param sampling_width: float, width of proposal distribution.  A factor scaling covariance matrix in multivariate gaussian proposal distribution.
    :param temp: system temperature.  Optional, default 0.
    :param covariance_matrix: initial or constant covariance matrix of multivariate gaussian proposal distribution. Optional, defaults to identity matrix. Should reflect correlations between the parameters if known.  Dimensions should match number of free parameters: in general number of field coefficients + 1.  2*number of field coefficients+1 Where phase/ampltitude or real/imag parts of field coefficients are treated as separate degrees of freedom.  complex entries in subclass ComplexAdaptiveMetropolisEngine.
    """
    self.temp = temp
    assert (self.temp is not None)
    self.num_field_coeffs = len(initial_field_coeffs)
    max_field_coeffs = max(initial_field_coeffs)
    self.keys_ordered = list(range(-max_field_coeffs, max_field_coeffs + 1))
    self.sampling_width = sampling_width
    self.param_space_dims = self.num_field_coeffs + 1  # in more basic case perturbation amplitude, magnitudes only of ci.  subclasses may consider both magnitude and phase / both real and img parts of cis and increase this number.
    self.mean = np.zeros(self.param_space_dims)
    if covariance_matrix is None:
      self.covariance_matrix = .03 * np.identity(
        self.param_space_dims)  # initial default guess needs to be of the right order of magnitude of variances, or covariance matrix doesnt stabilize withing first 200 steps before sigma starts adjusting
    else:
      self.covariance_matrix = covariance_matrix
    self.accepted_counter = 1
    self.step_counter = 1
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
      field_coeffs, field_energy = self.step_fieldcoeff(index, field_coeffs, field_energy, surface_energy, amplitude,
                                                        system)
    print(field_energy, field_coeffs)
    return amplitude, field_coeffs, surface_energy, field_energy

  def step_fieldcoeff(self, field_coeff_index, field_coeffs, field_energy, surface_energy, amplitude, system,
                      amplitude_change=False):
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
    proposed_field_coeff = self.modify_phase(
      abs(field_coeff) + random.gauss(0, self.sampling_width * covariance_matrix_entry), cmath.phase(field_coeff))
    new_field_energy = system.calc_field_energy_diff(field_coeff_index, proposed_field_coeff, field_coeffs, amplitude,
                                                     amplitude_change)
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
    proposed_amplitude = self.draw_amplitude_from_proposal_distriution(amplitude)
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
    if self.metropolis_decision((field_energy + surface_energy), (new_field_energy + new_surface_energy)):
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
    # print("propsed state", proposed_amplitude, proposed_field_coeffs)
    if abs(proposed_amplitude) >= 1:
      return amplitude, field_coeffs, surface_energy, field_energy
    new_field_energy = system.calc_field_energy(proposed_field_coeffs, proposed_amplitude, amplitude_change=True)
    new_surface_energy = system.calc_surface_energy(proposed_amplitude, amplitude_change=False)
    accept = self.metropolis_decision((field_energy + surface_energy), (new_field_energy + new_surface_energy))
    # print("propsed c0", proposed_field_coeffs[0])
    # print("field_energy", new_field_energy, "old_field_energy", field_energy)
    if accept:
      # print("accepted")
      field_energy = new_field_energy
      surface_energy = new_surface_energy
      amplitude = proposed_amplitude
      field_coeffs = proposed_field_coeffs
      self.accepted_counter += 1
    self.step_counter += 1
    self.update_proposal_distribution(accept, amplitude, field_coeffs)
    # output system properties, energy
    return amplitude, field_coeffs, surface_energy, field_energy

  def update_proposal_distribution(accept, amplitude, field_coeffs):
    """ does nothinng in non-adaptive base class """
    pass

  def draw_field_coeffs_from_proposal_distribution(self, ield_coeffs):
    """implement later"""
    # TODO : generalize to multivariate gaussian
    proposed_field_coeffs = copy.copy(field_coeffs)
    for key in proposed_field_coeffs:
      proposed_field_coeffs[key] += self.gaussian_complex(self.sampling_width_field_coeffs[key])
    return proposed_field_coeffs

  def draw_amplitude_from_proposal_distriution(self, amplitude):
    """ implement later"""
    # TODO : genrealize to multivariate gaussian
    proposed_amplitude = amplitude + random.gauss(0, self.sampling_width_amplitude)
    return proposed_amplitude

  def draw_all_from_proposal_distribution(self, amplitude, field_coeffs):
    """
    draw from multivariate gaussian distribution with sampling_width * covariance matrix
    exactly equivlent to drawing from n independent gaussian distributions when covariance matrix is identity matrix
    :param amplitude: current amplitude
    :param field_coeffs: dict of current (complex) field coeff values
    In this implementation only ampltiude of perturbation and magnitude of field coeffs is modifified by adaptive metropolis algorithm with (possible adapted) step sizes and covariance matrix.  Phases of field coeffs are independently adjusted by fixed-width, uncoupled gaussian distribution around their current state at each step.
    """
    state = self.construct_state(amplitude, field_coeffs)
    old_phases = [cmath.phase(field_coeffs[key]) for key in field_coeffs]
    proposed_addition = np.random.multivariate_normal([0] * len(state),
                                                      self.sampling_width ** 2 * self.covariance_matrix,
                                                      check_valid='raise')
    proposed_amplitude = amplitude + (-1 if amplitude < 0 else 1) * proposed_addition[0]
    proposed_field_coeff_amplitude = [original + addition for (original, addition) in
                                      zip(state[1:], proposed_addition[1:])]
    proposed_field_coeffs = dict(
      [(key, self.modify_phase(proposed_amplitude, old_phase)) for (key, (proposed_amplitude, old_phase)) in
       zip(self.keys_ordered, zip(proposed_field_coeff_amplitude, old_phases))])
    return proposed_amplitude, proposed_field_coeffs

  def metropolis_decision(self, old_energy, proposed_energy):
    """
    Considering energy difference and temperature, return decision to accept or reject step
    :param old_energy: current system total energy
    :param proposed_energy: system total energy after proposed change
    :return: bool True if step will be accepted; False to reject
    """
    diff = proposed_energy - old_energy
    assert (self.temp is not None)
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

  def construct_state(self, amplitude, field_coeffs):
    """
    helper function to construct position in parameter space as np array 
    By default the parameters we track for correlation matrix, co-adjustemnt of step sizes are [abs(amplitude), {abs(field_coeff_i)}] (in this order)
    override in derived classes where different representation of the parameters is tracked.
    """
    state = [abs(amplitude)]  # TODO: abs?
    state.extend([abs(field_coeffs[key]) for key in self.keys_ordered])
    return np.array(state)

  def modify_phase(self, amplitude, phase, phase_sigma=.5):
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

  @staticmethod
  def gaussian_complex(sigma=1):
    """
    a random complex number with completely random phase and amplitude drawn from a gaussian distribution with given width sigma (default 1)
    :param sigma: width of gaussian distribution (centered around 0) of possible magnitude of complex number
    :return: cmath complex number object in standard real, imaginary represenation 
    """
    amplitude = random.gauss(0, sigma)
    phase = random.uniform(0, 2 * math.pi)
    return cmath.rect(amplitude, phase)

  def set_temperature(self, new_temp):
    """
    change metropolis engine temperature
    :param new_temp: new temperature value, float >=0 .
    """
    assert (new_temp >= 0)
    self.temp = new_temp


class StaticCovarianceAdaptiveMetropolisEngine(MetropolisEngine):
  """ Reasoning and algorithm from 
  Garthwaite, Fan, & Sisson 2010: arxiv.org/abs/1006.3690v1
  here first without updating covariance matrix - assuming covariance is approximately the identity matrix (default) indicating no correlations, or a covariance matrix from known correlations of system parameters is given.
  """

  def __init__(self, initial_field_coeffs, initial_amplitude=None, sampling_width=0.05, temp=0, covariance_matrix=None):
    super().__init__(initial_field_coeffs, initial_amplitude, sampling_width, temp, covariance_matrix)

    # alpha (a constant used later) := -phi^(-1)(target acceptance /2) 
    # ,where  phi is the cumulative density function of the standard normal distribution
    # norm.ppf ('percent point function') is the inverse of the cumulative density function of standard normal distribution
    self.alpha = -1 * scipy.stats.norm.ppf(self.target_acceptance / 2)
    self.m = self.param_space_dims

  def update_proposal_distribution(self, accept, amplitude, field_coeffs):
    step_number_factor = max((self.step_counter / self.m, 200))
    steplength_c = self.sampling_width * (
        (1 - (1 / self.m)) * math.sqrt(2 * math.pi) * math.exp(self.alpha ** 2 / 2) / 2 * self.alpha + 1 / (
        self.m * self.target_acceptance * (1 - self.target_acceptance)))
    if accept:
      self.sampling_width += steplength_c * (1 - self.target_acceptance) / step_number_factor
    else:
      self.sampling_width -= steplength_c * self.target_acceptance / step_number_factor
    assert (self.sampling_width) > 0
  # TODO: test for convergence of sampling_width, c->0


class RobbinsMonroAdaptiveMetropolisEngine(StaticCovarianceAdaptiveMetropolisEngine):
  """ Reasoning and algorithm from 
  Garthwaite, Fan, & Sisson 2010: arxiv.org/abs/1006.3690v1
  without updating covariance matrix - assuming covariance is approximately identity matrix
  """

  def __init__(self, initial_field_coeffs, initial_amplitude=0, sampling_width=0.05, temp=0, method="simultaneous",
               covariance_matrix=None):
    super().__init__(initial_field_coeffs=initial_field_coeffs, initial_amplitude=initial_amplitude,
                     sampling_width=sampling_width, temp=temp, covariance_matrix=covariance_matrix)
    # start collecting data for updating covariance matrix
    self.mean = self.construct_state(initial_amplitude, initial_field_coeffs)

  def update_proposal_distribution(self, accept, amplitude, field_coeffs):
    # update sampling width
    super().update_proposal_distribution(accept, amplitude, field_coeffs)
    new_state = self.construct_state(amplitude, field_coeffs)
    old_mean = copy.copy(self.mean)
    assert (isinstance(new_state, np.ndarray))
    self.update_mean(state=new_state)
    if self.step_counter > 100:
      self.update_covariance_matrix(old_mean, new_state)
    # print("covariace_matrix", self.covariance_matrix)

  def update_mean(self, state):
    assert (isinstance(state, np.ndarray))
    self.mean *= (self.step_counter - 1) / self.step_counter
    self.mean += state / self.step_counter

  def update_covariance_matrix(self, old_mean, state):
    i = self.step_counter
    small_number = self.sampling_width ** 2 / i
    # print("added",  np.outer(old_mean,old_mean) - i/(i-1)*np.outer(self.mean,self.mean) + np.outer(state,state)/(i-1) + small_number*np.identity(self.param_space_dims))
    # print("multiplied", (i-2)/(i-1)*self.covariance_matrix)
    # print("result",  (i-2)/(i-1)*self.covariance_matrix +  np.outer(old_mean,old_mean)- i/(i-1)*np.outer(self.mean,self.mean) + np.outer(state,state)/(i-1) + small_number*np.identity(self.param_space_dims))
    self.covariance_matrix *= (i - 2) / (i - 1)
    self.covariance_matrix += np.outer(old_mean, old_mean) - i / (i - 1) * np.outer(self.mean, self.mean) + np.outer(
      state, state) / (i - 1) + small_number * np.identity(self.param_space_dims)
  # TODO: test for convergence of sampling_width, c->0

# a simpler class that stops adapting and starts measuring after n steps?
class FiniteAdaptiveMetropolisEngine(MetropolisEngine):
  def __init__(self, initial_field_coeffs, amplitude, sampling_width=0.05, temp=0):
    # init superclass
    self.adaption_size = 1

  def update_proposal_distribution(self, accept):
    if self.step_counter < n:
      if accept:
        self.sampling_width_amplitude += 0
        self.sampling_width_field_coeffs += 0
      else:
        pass
      self.adaption_size *= .08
    print("acceptance_rate", self.accept_counter / self.step_counter, "target", self.target)


########## three schemes for more accurate covariance matrix ###########
class RealImgAdaptiveMetropolisEngine(RobbinsMonroAdaptiveMetropolisEngine):
  def __init__(self, initial_field_coeffs, initial_amplitude, initial_covariance_matrix=None,
               initial_sampling_width=.0001, temp=0):
    self.param_space_dims = 2 * len(initial_field_coeffs) + 1
    print("initialized", self.state, self.mean)

  def construct_state(self):
    pass

class AmplitudePhaseAdaptiveMetropolisEngine(RobbinsMonroAdaptiveMetropolisEngine):
  def __init__(self, initial_field_coeffs, initial_amplitude, initial_covariance_matrix=None,
               initial_sampling_width=.0001, temp=0):
    pass

  def construct_state(self):
    pass

class ComplexAdaptiveMetropolisEngine(RobbinsMonroAdaptiveMetropolisEngine):
  def __init__(self, initial_field_coeffs, initial_amplitude=0, sampling_width=.001, initial_covariance_matrix=None,
               temp=0):
    super().__init__(initial_field_coeffs=initial_field_coeffs, sampling_width=sampling_width, temp=temp,
                     covariance_matrix=initial_covariance_matrix)

  def multivariate_normal_complex(self, mean, covariance_matrix):
    pass

  def construct_state(self):
    pass
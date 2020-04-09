import unittest
import run
import numpy as np
import system
import metropolis_engine
import random
import math
import copy

class TestBaseMetropolis(unittest.TestCase):

  def setUp(self): 
    self.field_coeffs=dict([(i, 0+0j) for i in range(-3,4)])
    self.me = metropolis_engine.MetropolisEngine(num_field_coeffs=len(self.field_coeffs.keys()) )
    self.sys_basic = system.System(radius=1, wavenumber=1, kappa=1, gamma=1, alpha=-1, u=1, C=1, n=1)

  def tearDown(self):
    pass
  
  def test_init_covariance_matrix_default(self):
    num_field_coeffs=3
    me = metropolis_engine.MetropolisEngine(num_field_coeffs=num_field_coeffs)
    self.assertEqual(me.covariance_matrix[1][3],0)
    self.assertEqual(me.covariance_matrix[2][2],1)
 
  def test_step_fieldcoeff(self):
    pass

  def test_step_amplitude(self):
    pass
  
  def test_step_fieldcoeffs_sequential(self):
    pass

  def test_step_all(self):
    pass

  def test_gaussian_complex(self):
    # TODO : statistical test?
    pass

  def test_set_temperature(self):
    me = metropolis_engine.MetropolisEngine(num_field_coeffs=3, temp=0)
    self.assertEqual(me.temp, 0.0)
    me.set_temperature(0.50)
    self.assertEqual(me.temp, 0.5)
    me.set_temperature(0.0)
    self.assertEqual(me.temp, 0)

  def test_metropolis_decision_1(self):
    old_energy = 12.45
    new_energy = -1451.0
    self.me.set_temperature(.001)
    self.assertTrue(self.me.metropolis_decision(old_energy, new_energy), "note: very rarely True by chance")

  def test_metropolis_decision_2(self):
    old_energy = -145.4
    new_energy = -145.3
    self.me.set_temperature(0)
    self.assertFalse(self.me.metropolis_decision(old_energy, new_energy))

  def test_metropolis_decision_2b(self):
    old_energy = -145.4
    new_energy = -145.5
    self.me.set_temperature(0)
    self.assertTrue(self.me.metropolis_decision( old_energy, new_energy))

  def test_metropolis_decision_3(self):
    old_energy = 0
    new_energy = 0
    self.me.set_temperature(0)
    self.assertTrue(self.me.metropolis_decision( old_energy, new_energy), "equal energies defined as accepted")

  def test_metropolis_decision_4(self):
    old_energy = 14.3
    new_energy = 14.3
    self.me.set_temperature(1.5)
    self.assertTrue(self.me.metropolis_decision(old_energy, new_energy), "equal energies defined as accepted")


class TestStaticCovarianceAdaptiveMetropolis(unittest.TestCase):

  def setUp(self): 
    self.field_coeffs=dict([(i, 0+0j) for i in range(-3,4)])
    self.me_identity = metropolis_engine.StaticCovarianceAdaptiveMetropolisEngine(num_field_coeffs=len(self.field_coeffs ))
    self.me_correlated = metropolis_engine.StaticCovarianceAdaptiveMetropolisEngine(num_field_coeffs=len(self.field_coeffs ))
    self.sys_basic = system.System(radius=1, wavenumber=1, kappa=1, gamma=1, alpha=-1, u=1, C=1, n=1)

  def tearDown(self):
    pass
   
  def test_step_counter(self):
    counter_initial = self.me_identity.step_counter
    self.me_identity.step_all(amplitude=-.12, field_coeffs=self.field_coeffs, surface_energy=-21.0, field_energy=34.2, system=self.sys_basic)
    counter_after = self.me_identity.step_counter
    self.assertEqual(counter_initial+1, counter_after)


  def test_initial_mean(self):
    amplitude = .5
    field_coeffs=dict([(i,self.me_identity.random_complex(random.uniform(0,1))) for i in range(-3,4)])
    me  = metropolis_engine.RobbinsMonroAdaptiveMetropolisEngine(field_coeffs, amplitude)
    mean_initial=me.mean
    self.assertEqual(mean_initial[0], amplitude) #initial amplitude
    self.assertEqual(mean_initial[2], abs(field_coeffs[-2]))
    self.assertEqual(mean_initial[-1], abs(field_coeffs[3]))

  def test_two_step_mean(self):
    amplitude = .5
    field_coeffs = dict([(i, 0+0j) for i in range(-3,4)])
    me = metropolis_engine.RobbinsMonroAdaptiveMetropolisEngine(field_coeffs, amplitude)
    new_amplitude = 0.8
    new_field_coeffs = dict([(i, 0-0.6j) for i in range(-3, 4)])
    me.step_counter +=1
    new_state = [new_amplitude]
    new_state.extend([abs(new_field_coeffs[key]) for key in range(-3,4)])
    new_state=np.array(new_state)
    me.update_mean(new_state)
    self.assertEqual(me.mean[0], 0.65)
    self.assertEqual(me.mean[4], 0.3)


  def test_initial_covariance_default(self):
    amplitude = .5
    field_coeffs=dict([(i,self.me_identity.random_complex(random.uniform(0,1))) for i in range(-3,4)])
    me  = metropolis_engine.RobbinsMonroAdaptiveMetropolisEngine(field_coeffs, amplitude)
    cov_initial=me.covariance_matrix
    self.assertEqual(cov_initial[0,0], 1)
    self.assertEqual(cov_initial[2,2], 1)
    self.assertEqual(cov_initial[1,0], 0)

  def test_initial_covariance(self):
    pass


  def test_two_step_covariance(self):
    amplitude = 0
    field_coeffs = dict([(i, 0+0j) for i in range(-3,4)])
    me = metropolis_engine.RobbinsMonroAdaptiveMetropolisEngine(field_coeffs, amplitude)
    old_mean = copy.copy(me.mean)
    new_amplitude = -1 #covariance matrix calculation should work with abs(amplitude) due to symmetry a <-> -a
    new_field_coeffs = dict([(i, 0-1j) for i in range(-3, 4)])
    me.step_counter +=1
    new_state = [abs(new_amplitude)]
    new_state.extend([abs(new_field_coeffs[key]) for key in range(-3,4)])
    new_state=np.array(new_state)
    me.update_mean(new_state)
    me.update_covariance_matrix(old_mean, new_state)
    #after this update, should be uniform covariance matrix with positive coefficients everywhere because the whole set of observable is identical at each of the two steps
    # the value should be 0*identity matrix + old mean**2 -2 new_mean **2 + 1 new_state**2 
    # 0+0 - 2*.5**2 + 1*1**2 = -1/2 + 1 = 1/2
    self.assertEqual(me.covariance_matrix[0,0], 0.5)
    self.assertEqual(me.covariance_matrix[2,2], 0.5)
    self.assertEqual(me.covariance_matrix[0,3], 0.5) #off-diagonal: ampltiude with a field coeff


  def test_two_step_covariance2(self):
    amplitude = 0
    field_coeffs = dict([(i, 1+0j) for i in range(-3,4)])
    me = metropolis_engine.RobbinsMonroAdaptiveMetropolisEngine(field_coeffs, amplitude)
    old_mean = copy.copy(me.mean)
    new_amplitude = -1 #covariance matrix calculation should work with abs(amplitude) due to symmetry a <-> -a
    new_field_coeffs = dict([(i, 0+0j) for i in range(-3, 4)])
    me.step_counter +=1
    new_state = [abs(new_amplitude)]
    new_state.extend([abs(new_field_coeffs[key]) for key in range(-3,4)])
    new_state=np.array(new_state)
    me.update_mean(new_state)
    me.update_covariance_matrix(old_mean, new_state)
    # some anticorrelated: expect + or - 0.5
    self.assertEqual(me.covariance_matrix[0,0], 0.5)
    self.assertEqual(me.covariance_matrix[2,2], 0.5)
    self.assertEqual(me.covariance_matrix[0,3], -0.5)

  def test_100th_step_covariance(self):
    amplitude = 0
    field_coeffs = dict([(i, 0+0j) for i in range(-3,4)])
    me = metropolis_engine.RobbinsMonroAdaptiveMetropolisEngine(field_coeffs, amplitude)
    me.step_counter = 99 #100th step - covariance has been identity until now; start adapting
    old_mean = copy.copy(me.mean) # assume mean has been all 0s for 100 steps
    new_amplitude = -1 #covariance matrix calculation should work with abs(amplitude) due to symmetry a <-> -a
    new_field_coeffs = dict([(i, 1+0j) for i in range(-3, 4)])
    me.step_counter +=1
    new_state = [abs(new_amplitude)]
    new_state.extend([abs(new_field_coeffs[key]) for key in range(-3,4)])
    new_state=np.array(new_state)
    me.update_mean(new_state)
    me.update_covariance_matrix(old_mean, new_state)
    # covriance matrix_100 = 98/99 identity matrix + [0s]*[0s] - 100/99 [mean 99 0s and 1x 1] + 1/99 [1s]
    # = .9998989.. on diagonal
    # 0 -100/99 * .1**2 +0 = 0.010 off diagonal
    self.assertAlmostEqual(me.covariance_matrix[0,0], 0.99989898989)
    self.assertAlmostEqual(me.covariance_matrix[2,2], 0.99989898989)
    self.assertAlmostEqual(me.covariance_matrix[0,3], 0.010)
    self.assertAlmostEqual(me.covariance_matrix[4,3], .010) # other off diagonal


if __name__ == '__main__':
  unittest.main()

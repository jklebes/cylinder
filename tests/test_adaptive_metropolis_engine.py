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
    me.update_mean(new_amplitude, new_field_coeffs)
    self.assertEqual(me.mean[0], 0.65)
    self.assertEqual(me.mean[4], 0.3)

if __name__ == '__main__':
  unittest.main()

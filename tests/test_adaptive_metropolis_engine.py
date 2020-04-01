import unittest
import run
import numpy as np
import system
import metropolis_engine
import random
import math
import copy

class TestAdaptiveMetropolis(unittest.TestCase):

  def setUp(self): 
    self.field_coeffs=dict([(i, 0+0j) for i in range(-3,4)])
    self.me = metropolis_engine.AdaptiveMetropolisEngine(initial_amplitude =0, initial_field_coeffs=self.field_coeffs )
    self.sys_basic = system.System(radius=1, wavenumber=1, kappa=1, gamma=1, alpha=-1, u=1, C=1, n=1)

  def tearDown(self):
    pass
  
  def test_init_covariance_matrix_default(self):
    num_field_coeffs=3
    me = metropolis_engine.AdaptiveMetropolisEngine(initial_amplitude=0, initial_field_coeffs=self.field_coeffs)
    self.assertEqual(me.covariance_matrix[1][3],0)
    self.assertEqual(me.covariance_matrix[2][2],1)

  
  def test_init_covariance_matrix(self):
    num_field_coeffs=3
    covariance_matrix = [[4.3, 0, 0.2],[2, 2.1, -4.3], [-0.01, 0.2, 0.25]]
    me = metropolis_engine.AdaptiveMetropolisEngine(initial_amplitude=0, initial_field_coeffs=dict([(i, 0+0j) for i in range(-1,2)]), initial_covariance_matrix=covariance_matrix)
    self.assertEqual(me.covariance_matrix[1][1],2.1)
    self.assertEqual(me.covariance_matrix[0][2],0.2)
 
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
if __name__ == '__main__':
  unittest.main()

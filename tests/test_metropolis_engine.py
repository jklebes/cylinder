import unittest
import run
import numpy as np
import system
import metropolis_engine
import random
import math
import copy

class TestMetropolisSteps(unittest.TestCase):

  def setUp(self):
    self.me = metropolis_engine.MetropolisEngine( num_field_coeffs=3)
    self.sys_basic = system.System(radius=1, wavenumber=1, kappa=1, gamma=1, alpha=-1, u=1, C=1, n=1)

  def tearDown(self):
    pass
  
  def test_init_sampling_widths_float(self):
    num_field_coeffs=3
    me = metropolis_engine.MetropolisEngine(num_field_coeffs=num_field_coeffs, sampling_widths=.5)
    self.assertEqual(me.sampling_width_amplitude,.5)
    self.assertEqual(me.sampling_width_coeffs[-num_field_coeffs],.5)
    self.assertEqual(me.sampling_width_coeffs[num_field_coeffs],.5)
    self.assertEqual(me.sampling_width_coeffs[0],.5)

  
  def test_init_sampling_widths_tuple(self):
    num_field_coeffs=3
    me = metropolis_engine.MetropolisEngine(num_field_coeffs=num_field_coeffs, sampling_widths=(.21, {-3:.7, -2:.3, -1:.1, 0:1.4, 1:3, 2:5, 3:.001}))
    self.assertEqual(me.sampling_width_amplitude,.21)
    self.assertEqual(me.sampling_width_coeffs[-num_field_coeffs],.7)
    self.assertEqual(me.sampling_width_coeffs[num_field_coeffs],.001)
    self.assertEqual(me.sampling_width_coeffs[0],1.4)
 
  def test_step_fieldcoeff(self):
    num_field_coeffs=3
    # TODO: make this common line a function
    field_coeffs = dict([(i, complex(0,0)) for i in range(-num_field_coeffs, num_field_coeffs+1)])
    field_energy = 4443716734.2 # pretend last field energy was very high; change of field coeff will be accepted
    index = -2
    surface_energy = 12.5 #some constant value
    amplitude = .012
    system = self.sys_basic
    new_field_coeffs, new_field_energy = self.me.step_fieldcoeff(index, field_coeffs, field_energy, surface_energy, amplitude, system)
    #new_field_coeffs[index] changed
    self.assertNotEqual(new_field_coeffs[index], complex(0,0))
    # new energy decreased (certain with temp=0 and unrealistically high old energy value)
    self.assertLess(new_field_energy, field_energy)
  
  #TODO : more variations

  def test_step_amplitude(self):
    pass
  
  def test_step_fieldcoeffs_sequential(self):
    pass

  def test_step_all(self):
    pass


class TestMetropolisHelperFunctions(unittest.TestCase):

  def setUp(self):
    self.me = metropolis_engine.MetropolisEngine( num_field_coeffs=3, sampling_dist=random.gauss, sampling_widths=.5, temp=0)

  def tearDown(self):
    pass
  
  def test_gaussian_complex(self):
    # TODO : statistical test?
    pass

  def test_set_temperature(self):
    me = metropolis_engine.MetropolisEngine(num_field_coeffs=3, sampling_dist=random.gauss, sampling_widths=.5, temp=0)
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

import unittest
import run
import numpy as np
import system
import metropolis_engine
import random
import math
import copy

class TestMetropolis(unittest.TestCase):

  def setUp(self):
    self.me = metropolis_engine.MetropolisEngine(method='sequential', num_field_coeffs=3, sampling_dist=random.gauss, sampling_widths=.5, temp=0)

  def tearDown(self):
    # self.widget.dispose()
    pass
  
  
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

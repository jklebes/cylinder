import unittest
import run
import numpy as np
import calc_energy as ce
import random
import math
import copy

class TestRun(unittest.TestCase):

  def setUp(self):
    pass

  def tearDown(self):
    # self.widget.dispose()
    pass
  
  def test_loop_wavenumber_kappa_dims(self):
    """

    """
    range1 = np.arange(0.005, .5, .3)
    range2 = np.arange(0, .2, .3)
    amp_steps = 1
    converge_stop = True
    fieldsteps_per_ampstep = 1
    amplitude_results = run.loop_wavenumber_kappa(wavenumber_range=range1, kappa_range=range2,
                                                   amp_steps=amp_steps, 
                                                   fieldsteps_per_ampstep=fieldsteps_per_ampstep,converge_stop=converge_stop,
                                                  )
    self.assertEqual(amplitude_results.shape, (len(range1), len(range2)), "dimensions of amplitude output")
  
  def test_metropolis_decision_1(self):
    old_energy = 12.45
    new_energy = -1451.0
    temp= .001
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy), "note: very rarely True by chance")

  def test_metropolis_decision_2(self):
    old_energy = -145.4
    new_energy = -145.3
    temp = 0
    self.assertFalse(run.metropolis_decision(temp, old_energy, new_energy))

  def test_metropolis_decision_2b(self):
    old_energy = -145.4
    new_energy = -145.5
    temp = 0
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy))

  def test_metropolis_decision_3(self):
    old_energy = 0
    new_energy = 0
    temp= 0
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy), "equal energies defined as accepted")

  def test_metropolis_decision_4(self):
    old_energy = 14.3
    new_energy = 14.3
    temp=1.5
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy), "equal energies defined as accepted")

if __name__ == '__main__':
    unittest.main()

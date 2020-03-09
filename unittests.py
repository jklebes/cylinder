import unittest
import run
import numpy as np
import calc_energy as ce
import random

class TestRun(unittest.TestCase):

  def setUp(self):
    pass

  def tearDown(self):
    # self.widget.dispose()
    pass
  """
  def test_loop_wavenumber_kappa_dims(self):
    range1 = np.arange(0.005, .5, .3)
    range2 = np.arange(0, .2, .3)
    amp_steps = 1
    converge_stop = True
    fieldsteps_per_ampstep = 1
    converge_time, amplitude_results = run.loop_wavenumber_kappa(wavenumber_range=range1, kappa_range=range2,
                                                   amp_steps=amp_steps
                                                   , fieldsteps_per_ampstep=fieldsteps_per_ampstep,converge_stop=converge_stop,
                                                  )
    #self.assertEqual(converge_time.shape, (len(range1), len(range2)), "dimensions of convergetime output")
    self.assertEqual(amplitude_results.shape, (len(range1), len(range2)), "dimensions of amplitude output")
  """
  def test_decide_change1(self):
    old_energy = 12.45
    new_energy = -1451.0
    temp= .001
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy), "note: very rarely True by chance")

  def test_decide_change2(self):
    old_energy = -145.4
    new_energy = -145.3
    temp = 0
    self.assertFalse(run.metropolis_decision(temp, old_energy, new_energy))

  def test_decide_change2b(self):
    old_energy = -145.4
    new_energy = -145.5
    temp = 0
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy))

  def test_decide_change3(self):
    old_energy = 0
    new_energy = 0
    temp= 0
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy), "equal energies defined as accepted")

  def test_decide_change4(self):
    old_energy = 14.3
    new_energy = 14.3
    temp=1.5
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy), "equal energies defined as accepted")

class TestCalcEnergy(unittest.TestCase):

  def setUp(self):
    #self.widget = Widget('The widget')
    pass
  
  def tearDown(self):
    #self.widget.dispose()
    pass

  def test_calc_field_energy_zerofield(self):
    num_field_coeffs = 3
    field_coeffs= dict([(i, 0+0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
    amplitude = 0
    wavenumber=1
    n=1
    radius = 1
    alpha= -1
    C=1
    u=1
    energy = ce.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_zerofield2(self):
    num_field_coeffs = 3
    field_coeffs= dict([(i, 0+0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
    amplitude = .3
    wavenumber=1
    n=1
    radius = 1
    alpha= -1
    C=1
    u=1
    energy = ce.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_1(self):
    num_field_coeffs = 3
    field_coeffs = dict([(i, complex(random.uniform(-1,1), random.uniform(-1,1))) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    amplitude = 0
    wavenumber = 1
    n = 1
    radius = 1
    alpha = 0
    C = 0
    u = 0
    energy = ce.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_2(self):
    num_field_coeffs = 3
    field_coeffs = dict([(i, complex(random.uniform(-1,1), random.uniform(-1,1))) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    wavenumber = 1
    n = 1
    radius = 1
    alpha = -1
    C = 1
    u = 1
    energy_flat = ce.calc_field_energy(field_coeffs, 0, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    energy_perturbed = ce.calc_field_energy(field_coeffs, .2, wavenumber, radius, n, alpha, C, u,
                                       amplitude_change=True)
    # TODO : this sometimes fails
    self.assertGreater(energy_perturbed, energy_flat)

  def test_calc_field_energy_3(self):
    """
    should be more energy cost when there is a nonzero field bending rigidity
    :return:
    """
    num_field_coeffs = 3
    field_coeffs = dict([(i, complex(random.uniform(-1, 1), random.uniform(-1, 1))) for i in
                           range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    wavenumber = 1
    n = 1
    radius = 1
    alpha = -1
    amplitude = 0.1
    u = 1
    energy_0C = ce.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, 0, u, amplitude_change=True)
    energy_C = ce.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, 1, u,
                                              amplitude_change=True)
    self.assertGreater(energy_C, energy_0C)

if __name__ == '__main__':
    unittest.main()

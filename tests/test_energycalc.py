import unittest
import run
import numpy as np
import calc_energy as ce
import random
import math
import copy


class TestIntegrandFactors(unittest.TestCase):

  def setUp(self):
    #self.system_energy = ce.System_Energy()
    pass
  def tearDown(self):
    #self.widget.dispose()
    pass

  def test_sqrt_g_theta_flat(self):
    radius=1
    amplitude=0
    wavenumber=1
    system_energy = ce.System_Energy()
    for z in np.arange(0, 2 * math.pi / wavenumber, .3):
      flat = system_energy.sqrt_g_theta(radius, amplitude, wavenumber, z)
      self.assertEqual(flat, 1)
      small_radius = system_energy.sqrt_g_theta(.5, amplitude, wavenumber, z)
      self.assertLess(small_radius, 1)

  def test_sqrt_g_z(self):
    radius = 1
    amplitude = 0
    wavenumber = 1
    system_energy = ce.System_Energy()
    for z in np.arange(0, 2 * math.pi / wavenumber, .3):
      flat = system_energy.sqrt_g_z(radius, amplitude, wavenumber, z)
      self.assertEqual(flat, 1)
      perturbed = system_energy.sqrt_g_z(radius, .3, wavenumber, z)
      self.assertGreater(perturbed, 1)
      perturbed2 = system_energy.sqrt_g_z(radius, -.5, wavenumber, z)
      self.assertGreater(perturbed2, 1)

  def test_n_A_theta(self):
    system_energy = ce.System_Energy()
    radius=1
    amplitude=0
    wavenumber=1
    n=1
    for z in np.arange(0, 2 * math.pi / wavenumber, .3):
      flat = system_energy.n_A_theta_squared(radius, amplitude, wavenumber, z, n=n)
      self.assertEqual(flat, 0)
      perturbed = system_energy.n_A_theta_squared(radius, .12, wavenumber, z, n=n)
      self.assertGreater(perturbed, 0)
      perturbed2 = system_energy.n_A_theta_squared(radius, -.3, wavenumber, z, n=n)
      self.assertGreater(perturbed2, 0)
      perturbed_n6 = system_energy.n_A_theta_squared(radius, -.3, wavenumber, z, n=6)
      self.assertGreater(perturbed_n6, perturbed2)

  def test_radius_rescale_factor(self):
    system_energy = ce.System_Energy()
    flat = system_energy.radius_rescaled(amplitude=0,radius=1)
    self.assertEqual(flat, 1)
    perturbed = system_energy.radius_rescaled(amplitude=-.2,radius=1)
    self.assertLess(perturbed, 1)
    perturbed2 = system_energy.radius_rescaled(amplitude=0.9,radius=1)
    self.assertLess(perturbed2, 1)

  def test_A_integrand_psi0(self):
    diff = 0
    radius = 1
    amplitude = 0
    wavenumber = 1
    system_energy = ce.System_Energy()
    for z in np.arange(0, 2 * math.pi / wavenumber, .2):
      img = system_energy.A_integrand_img_part( diff, amplitude, z, wavenumber, radius)
      real = system_energy.A_integrand_real_part(diff, amplitude, z, wavenumber, radius)
      self.assertEqual(img, 0)
      self.assertEqual(real, 1) # surface area per unit area? without factor of 2pi

class Test_Calc_Energy(unittest.TestCase):

  def test_evaluate_A_integrals_zerofield_zeroamplitude(self):
    system_energy = ce.System_Energy()
    num_field_coeffs=1
    amplitude= 0
    radius=1
    wavenumber=1
    field_coeffs=dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    self.assertDictEqual(system_energy.A_integrals, dict([])) #an empty dict before using anything
    system_energy.evaluate_A_integrals(amplitude, wavenumber, field_coeffs, radius)
    #not an empty dict anymore
    self.assertNotEqual(system_energy.A_integrals, dict([]))
    # surface area (A_integrals[0]) is as expected
    self.assertEqual(system_energy.A_integrals[0], complex(2*math.pi/wavenumber, 0))
    # complex parts of A_1 are 0 
    self.assertAlmostEqual(system_energy.A_integrals[1], complex(0, 0))
    self.assertAlmostEqual(system_energy.A_integrals[-1], complex(0, 0))

  def test_evaluate_A_integrals_zerofield_amplitude_change(self):
    system_energy = ce.System_Energy()
    num_field_coeffs = 1
    nonzero_amplitude = 0.32
    radius = 1
    wavenumber = 1
    field_coeffs=dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    #evaluate at new ammpitude
    system_energy.evaluate_A_integrals(nonzero_amplitude, wavenumber, field_coeffs, radius)
    #pertrubed  - should change value to different surface area
    self.assertNotEqual(system_energy.A_integrals[0].real, complex(2 * math.pi/wavenumber, 0).real)

  def test_evaluate_A_integrals_zerofield_wavenumber_change(self):
    system_energy = ce.System_Energy()
    num_field_coeffs=1
    nonzero_amplitude = -.232
    field_coeffs=dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    radius = 1
    unit_wavenumber = 1
    low_wavenumber = .8
    high_wavenumber = 1.2
    #low wavenumber - greater surface area
    system_energy.evaluate_A_integrals(nonzero_amplitude, low_wavenumber, field_coeffs, radius)
    self.assertGreater(system_energy.A_integrals[0].real, complex(2 * math.pi, 0).real)
    self.assertAlmostEqual(system_energy.A_integrals[0].imag, complex(2 * math.pi, 0).imag)
    #high wavenumber - less surface area
    system_energy.evaluate_A_integrals(nonzero_amplitude, high_wavenumber, field_coeffs, radius)
    self.assertLess(system_energy.A_integrals[0].real, complex(2 * math.pi, 0).real)
    self.assertAlmostEqual(system_energy.A_integrals[0].imag, complex(2 * math.pi, 0).imag)

  def test_B_integrand_psi0(self):
    i,j = 0,0
    radius = 1
    amplitude = 0
    wavenumber = 1
    n=1
    system_energy = ce.System_Energy()
    for z in np.arange(0, 2 * math.pi / wavenumber, .2):
      img = system_energy.B_integrand_img_part( i,j, amplitude, z, wavenumber, radius)
      real = system_energy.B_integrand_real_part(i,j, amplitude, z, wavenumber, radius, n)
      self.assertEqual(img, 0)
      self.assertEqual(real, 0) 

  def test_evaluate_B_integrals_zerofield_zeroamplitude(self):
    system_energy = ce.System_Energy()
    num_field_coeffs = 1
    amplitude = 0
    radius = 1
    wavenumber = 1
    n=1
    field_coeffs = dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    self.assertDictEqual(system_energy.B_integrals, dict([]))  # an empty dict before using anything
    system_energy.evaluate_B_integrals(amplitude, wavenumber, field_coeffs, radius, n=n)
    self.assertNotEqual(system_energy.B_integrals, dict([])) #not an empty dict
    #expected values at a=0, c_i=0
    self.assertEqual(system_energy.B_integrals[(0,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(1,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(-1,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(1, 1)], complex(radius*2*math.pi/wavenumber, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(-1, -1)], complex(radius*2*math.pi/wavenumber, 0))

  def test_evaluate_B_integrals_zerofield_amplitude_change(self):
    system_energy = ce.System_Energy()
    num_field_coeffs=1
    new_amplitude=.243
    wavenumber=1
    n=1
    radius=1
    amplitude = .352
    field_coeffs = dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    system_energy.evaluate_B_integrals(new_amplitude, wavenumber, field_coeffs, radius, n=n)
    # pertrubed  - should change value to greater
    # TODO : ?
    self.assertNotEqual(system_energy.B_integrals[(0,0)].real, complex(0, 0).real)

  def test_evaluate_B_integrals_field_zeroamplitude(self):
    system_energy = ce.System_Energy()
    num_field_coeffs = 1
    amplitude = 0
    radius = 1
    wavenumber = 1
    n=1
    field_coeffs = dict([(i, run.rand_complex()) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    self.assertDictEqual(system_energy.B_integrals, dict([]))  # an empty dict before using anything
    system_energy.evaluate_B_integrals(amplitude, wavenumber, field_coeffs, radius, n=n)
    self.assertNotEqual(system_energy.B_integrals, dict([])) #not an empty dict
    #expected values at a=0
    self.assertEqual(system_energy.B_integrals[(0,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(1,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(-1,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(1, 1)], complex(radius*2*math.pi/wavenumber, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(-1, -1)], complex(radius*2*math.pi/wavenumber, 0))
  
  def test_evaluate_B_integrals_field_amplitude_change(self):
    # evaluate at new ammpitude
    system_energy = ce.System_Energy()
    num_field_coeffs = 1
    new_amplitude= -.243
    radius =1
    wavenumber = 1
    n=1
    field_coeffs = dict([(i, run.rand_complex()) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    system_energy.evaluate_B_integrals(new_amplitude, wavenumber, field_coeffs, radius, n=n)
    # pertrubed  - should change value to greater
    self.assertNotEqual(system_energy.B_integrals[(0,0)].real, complex(0, 0).real)

  def test_calc_field_energy_zerofield_zeroamplitude(self):
    num_field_coeffs = 3
    field_coeffs= dict([(i, 0+0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
    amplitude = 0
    wavenumber=1
    n=1
    radius = 1
    alpha= -1
    C=1
    u=1
    system_energy = ce.System_Energy()
    energy = system_energy.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_zerofield_amplitude(self):
    num_field_coeffs = 3
    field_coeffs= dict([(i, 0+0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
    amplitude = .3
    wavenumber=1
    n=1
    radius = 1
    alpha= -1
    C=1
    u=1
    system_energy = ce.System_Energy()
    energy = system_energy.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_randfield_zeroamplitude(self):
    num_field_coeffs = 3
    field_coeffs = dict([(i, run.rand_complex()) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    amplitude = 0
    wavenumber = 1
    n = 1
    radius = 1
    alpha = 0
    C = 0
    u = 0
    system_energy = ce.System_Energy()
    energy = system_energy.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_Cchange(self):
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
    zeroC = 0
    unitC = 1
    system_energy = ce.System_Energy()
    energy_0C = system_energy.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, zeroC, u, amplitude_change=True)
    energy_C = system_energy.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, unitC, u,
                                              amplitude_change=True)
    self.assertGreater(energy_C, energy_0C)

class Test_Energy_Diff(unittest.TestCase):
  pass

if __name__ == '__main__':
    unittest.main()

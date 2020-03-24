import unittest
import run
import numpy as np
import system
import random
import math
import copy


class TestIntegrandFactors(unittest.TestCase):

  def setUp(self):
    self.sys_basic = system.System(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)
    self.sys_small_radius = system.System(radius=0.5, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)
    self.sys_n6 = system.System(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=6)

  def tearDown(self):
    #self.widget.dispose()
    pass

  def test_sqrt_g_theta_flat(self):
    wavenumber = self.sys_basic.wavenumber
    for z in np.arange(0, 2 * math.pi / wavenumber, .3):
      flat = self.sys_basic.sqrt_g_theta(amplitude=0, z=z)
      self.assertEqual(flat, 1)
      small_radius = self.sys_small_radius.sqrt_g_theta(amplitude=0, z=z)
      self.assertLess(small_radius, 1)

  def test_sqrt_g_z(self):
    wavenumber = self.sys_basic.wavenumber
    for z in np.arange(0, 2 * math.pi / wavenumber, .3):
      flat = self.sys_basic.sqrt_g_z(amplitude=0, z=z)
      self.assertEqual(flat, 1)
      perturbed = self.sys_basic.sqrt_g_z(amplitude=.3, z=z)
      self.assertGreater(perturbed, 1)
      perturbed2 = self.sys_basic.sqrt_g_z(amplitude= -.5, z=z)
      self.assertGreater(perturbed2, 1)

  def test_n_A_theta(self):
    wavenumber = self.sys_basic.wavenumber
    for z in np.arange(0, 2 * math.pi / wavenumber, .3):
      flat = self.sys_basic.n_A_theta_squared(amplitude=0, z=z)
      self.assertEqual(flat, 0)
      perturbed = self.sys_basic.n_A_theta_squared(amplitude=.12, z=z)
      self.assertGreater(perturbed, 0)
      perturbed2 = self.sys_basic.n_A_theta_squared(amplitude=-.3, z=z)
      self.assertGreater(perturbed2, 0)
      perturbed_n6 = self.sys_n6.n_A_theta_squared(amplitude=-.3, z=z)
      self.assertGreater(perturbed_n6, perturbed2)

  def test_radius_rescale_factor(self):
    flat = self.sys_basic.radius_rescaled(amplitude=0)
    self.assertEqual(flat, 1)
    perturbed = self.sys_basic.radius_rescaled(amplitude=-.2)
    self.assertLess(perturbed, 1)
    perturbed2 = self.sys_basic.radius_rescaled(amplitude=0.9)
    self.assertLess(perturbed2, 1)

  def test_A_integrand_psi0(self):
    diff = 0
    amplitude = 0
    wavenumber=self.sys_basic.wavenumber
    for z in np.arange(0, 2 * math.pi / wavenumber, .2):
      img = self.sys_basic.A_integrand_img_part( diff, amplitude, z)
      real = self.sys_basic.A_integrand_real_part(diff, amplitude, z)
      self.assertEqual(img, 0)
      self.assertEqual(real, 1) # surface area per unit area? without factor of 2pi

class Test_Calc_Energy(unittest.TestCase):

  def setUp(self):
    self.sys_basic = system.System(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)
    self.sys_low_wvn = system.System(radius=1, wavenumber=0.8, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)
    self.sys_high_wvn = system.System(radius=1, wavenumber=1.2, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)

  def tearDown(self):
    #self.widget.dispose()
    pass
  
  def test_evaluate_A_integrals_zerofield_zeroamplitude(self):
    num_field_coeffs=1
    amplitude= 0
    wavenumber =self.sys_basic.wavenumber
    field_coeffs=dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    self.assertDictEqual(self.sys_basic.A_integrals, dict([])) #an empty dict before using anything
    self.sys_basic.evaluate_A_integrals(amplitude, field_coeffs)
    #not an empty dict anymore
    self.assertNotEqual(self.sys_basic.A_integrals, dict([]))
    # surface area (A_integrals[0]) is as expected
    self.assertEqual(self.sys_basic.A_integrals[0], complex(2*math.pi/wavenumber, 0))
    # complex parts of A_1 are 0 
    self.assertAlmostEqual(self.sys_basic.A_integrals[1], complex(0, 0))
    self.assertAlmostEqual(self.sys_basic.A_integrals[-1], complex(0, 0))

  def test_evaluate_A_integrals_zerofield_amplitude_change(self):
    nonzero_amplitude = 0.32
    wavenumber =self.sys_basic.wavenumber
    num_field_coeffs=1
    field_coeffs=dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    #evaluate at new ammpitude
    self.sys_basic.evaluate_A_integrals(nonzero_amplitude, field_coeffs)
    #pertrubed  - should change value to different surface area
    self.assertNotEqual(self.sys_basic.A_integrals[0].real, complex(2 * math.pi/wavenumber, 0).real)

  def test_evaluate_A_integrals_zerofield_wavenumber_change(self):
    num_field_coeffs=1
    nonzero_amplitude = -.232
    field_coeffs=dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    #low wavenumber - greater surface area
    self.sys_low_wvn.evaluate_A_integrals(nonzero_amplitude, field_coeffs)
    self.assertGreater(self.sys_low_wvn.A_integrals[0].real, complex(2 * math.pi, 0).real)
    self.assertAlmostEqual(self.sys_low_wvn.A_integrals[0].imag, complex(2 * math.pi, 0).imag)
    #high wavenumber - less surface area
    self.sys_high_wvn.evaluate_A_integrals(nonzero_amplitude, field_coeffs)
    self.assertLess(self.sys_high_wvn.A_integrals[0].real, complex(2 * math.pi, 0).real)
    self.assertAlmostEqual(self.sys_high_wvn.A_integrals[0].imag, complex(2 * math.pi, 0).imag)

  def test_B_integrand_psi0(self):
    i,j = 0,0
    amplitude=0
    for z in np.arange(0, 2 * math.pi / self.sys_basic.wavenumber, .2):
      img = self.sys_basic.B_integrand_img_part( i,j, amplitude, z)
      real = self.sys_basic.B_integrand_real_part(i,j, amplitude, z)
      self.assertEqual(img, 0)
      self.assertEqual(real, 0) 

  def test_evaluate_B_integrals_zerofield_zeroamplitude(self):
    wavenumber = self.sys_basic.wavenumber
    radius = self.sys_basic.radius
    num_field_coeffs = 1
    field_coeffs = dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    # TODO : change this to fill on initialize?
    self.assertDictEqual(self.sys_basic.B_integrals, dict([]))  # an empty dict before using anything
    self.sys_basic.evaluate_B_integrals(amplitude=0, field_coeffs=field_coeffs)
    self.assertNotEqual(self.sys_basic.B_integrals, dict([])) #not an empty dict
    #expected values at a=0, c_i=0
    self.assertEqual(self.sys_basic.B_integrals[(0,0)], complex(0, 0))
    self.assertAlmostEqual(self.sys_basic.B_integrals[(1,0)], complex(0, 0))
    self.assertAlmostEqual(self.sys_basic.B_integrals[(-1,0)], complex(0, 0))
    self.assertAlmostEqual(self.sys_basic.B_integrals[(1, 1)], complex(radius*2*math.pi/wavenumber, 0))
    self.assertAlmostEqual(self.sys_basic.B_integrals[(-1, -1)], complex(radius*2*math.pi/wavenumber, 0))

  def test_evaluate_B_integrals_zerofield_amplitude_change(self):
    num_field_coeffs=1
    new_amplitude=.243
    new_amplitude2 = .352
    field_coeffs = dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    self.sys_basic.evaluate_B_integrals(new_amplitude,field_coeffs)
    # pertrubed  - should change value to greater
    # TODO : ?
    self.assertNotEqual(self.sys_basic.B_integrals[(0,0)].real, complex(0, 0).real)

  def test_evaluate_B_integrals_field_zeroamplitude(self):
    num_field_coeffs = 1
    amplitude = 0
    wavenumber = self.sys_basic.wavenumber
    radius = self.sys_basic.radius
    field_coeffs = dict([(i, run.rand_complex()) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    self.assertDictEqual(self.sys_basic.B_integrals, dict([]))  # an empty dict before using anything
    self.sys_basic.evaluate_B_integrals(amplitude, field_coeffs)
    self.assertNotEqual(self.sys_basic.B_integrals, dict([])) #not an empty dict
    #expected values at a=0
    self.assertEqual(self.sys_basic.B_integrals[(0,0)], complex(0, 0))
    self.assertAlmostEqual(self.sys_basic.B_integrals[(1,0)], complex(0, 0))
    self.assertAlmostEqual(self.sys_basic.B_integrals[(-1,0)], complex(0, 0))
    self.assertAlmostEqual(self.sys_basic.B_integrals[(1, 1)], complex(radius*2*math.pi/wavenumber, 0))
    self.assertAlmostEqual(self.sys_basic.B_integrals[(-1, -1)], complex(radius*2*math.pi/wavenumber, 0))
  
  def test_evaluate_B_integrals_field_amplitude_change(self):
    # evaluate at new ammpitude
    num_field_coeffs = 1
    new_amplitude= -.243
    field_coeffs = dict([(i, run.rand_complex()) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    self.sys_basic.evaluate_B_integrals(new_amplitude, field_coeffs)
    # pertrubed  - should change value to greater
    self.assertNotEqual(self.sys_basic.B_integrals[(0,0)].real, complex(0, 0).real)

  def test_calc_field_energy_zerofield_zeroamplitude(self):
    num_field_coeffs = 3
    field_coeffs= dict([(i, 0+0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
    amplitude = 0
    energy = self.sys_basic.calc_field_energy(field_coeffs, amplitude, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_zerofield_amplitude(self):
    num_field_coeffs = 3
    field_coeffs= dict([(i, 0+0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
    amplitude = .3
    energy = self.sys_basic.calc_field_energy(field_coeffs, amplitude, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_zerofieldconstants(self):
    num_field_coeffs = 3
    field_coeffs = dict([(i, run.rand_complex()) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    amplitude = 0
    sys_zerofieldcoeffs = system.System(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=0, C=0, u=0, n=1)
    energy = sys_zerofieldcoeffs.calc_field_energy(field_coeffs, amplitude, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_Cchange(self):
    """
    should be more energy cost when there is a nonzero field bending rigidity
    :return:
    """
    num_field_coeffs = 3
    field_coeffs = dict([(i, run.rand_complex()) for i in
                           range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    amplitude = 0.1
    sys_zeroC = system.System(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=0, u=1, n=1)
    energy_0C = sys_zeroC.calc_field_energy(field_coeffs, amplitude, amplitude_change=True)
    energy_C = self.sys_basic.calc_field_energy(field_coeffs, amplitude, amplitude_change=True)
    self.assertGreater(energy_C, energy_0C)

class Test_Energy_Diff(unittest.TestCase):
  def setUp(self):
    self.sys_basic = system.System(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)

  def test_calc_energy_diff_singlecoefftozero(self):
    num_field_coeffs =0
    field_coeffs = dict([(i, run.rand_complex()) for i in
                           range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    amplitude = 0 
    energy_before = self.sys_basic.calc_field_energy(field_coeffs, amplitude, amplitude_change=True)
    index = 0
    #new field coeffs with one value (c_1) changed to a different random complex value
    new_field_coeffs = copy.copy(field_coeffs)
    new_field_coeff = 0+0j
    new_field_coeffs[index] = new_field_coeff
    assert(field_coeffs[index]!=new_field_coeffs[index])
    energy_after_diff =  self.sys_basic.calc_field_energy_diff(index, new_field_coeff, field_coeffs, amplitude)
    energy_after =  self.sys_basic.calc_field_energy(new_field_coeffs, amplitude, amplitude_change=False)
    self.assertEqual(energy_after, 0)
    self.assertEqual(energy_after_diff, 0)

  def test_calc_energy_diff(self):
    num_field_coeffs =1
    field_coeffs = dict([(i, complex(random.uniform(-1, 1), random.uniform(-1, 1))) for i in
                           range(-1 * num_field_coeffs, num_field_coeffs + 1)])

if __name__ == '__main__':
  unittest.main()

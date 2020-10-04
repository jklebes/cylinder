import scipy.integrate as integrate
import unittest
import run
import numpy as np
import surfaces_and_fields.system_cylinder1D as system
import surfaces_and_fields.system_cylinder2D as system2D
import random
import math
import copy

class TestIntegrandFactors(unittest.TestCase):

  def setUp(self):
    self.sys_basic = system.Cylinder1D(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)
    self.sys_small_radius = system.Cylinder1D(radius=0.5, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)
    self.sys_n6 = system.Cylinder1D(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=6)

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

class Test_Calc_Field_Energy(unittest.TestCase):

  def setUp(self):
    self.sys_basic = system.Cylinder1D(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)
    self.sys_low_wvn = system.Cylinder1D(radius=1, wavenumber=0.8, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)
    self.sys_high_wvn = system.Cylinder1D(radius=1, wavenumber=1.2, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)

  def tearDown(self):
    #self.widget.dispose()
    pass


class Test_Surface_Energy(unittest.TestCase):

  def setUp(self):
    self.sys_basic = system.Cylinder1D(radius=1, wavenumber=1.001, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1, num_field_coeffs=0) # not exactly wavenumber 1 to know expected stability: surface area should decrease with a !=0
    self.sys_low_wvn = system.Cylinder1D(radius=1, wavenumber=0.8, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1, num_field_coeffs=0)
    self.sys_high_wvn = system.Cylinder1D(radius=1, wavenumber=1.2, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1, num_field_coeffs=0)

  def tearDown(self):
    #self.widget.dispose()
    pass
 
  def test_Kzz(self): 
    pass

  def test_Kthth(self): 
    pass
  def test_surface_area(self):
    pass

  def test_curvature_energy_zerofield(self):
    pass
  
  def test_curvature_energy_short_cylinder(self):
    pass


if __name__ == '__main__':
  unittest.main()

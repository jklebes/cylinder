import scipy.integrate as integrate
import unittest
import run
import numpy as np
import surfaces_and_fields.system_cylinder1D as cylinder1D
import surfaces_and_fields.On_lattice_simple as On_lattice
import random
import math
import copy
import matplotlib.pyplot as plt

class TestLattice(unittest.TestCase):

  def setUp(self):
    """
    with num_field_coeffs=1, the A matrix is 3x3
    """
    self.radius=1
    self.wavenumber=1
    self.sys_basic = cylinder1D.Cylinder1D(radius=self.radius, wavenumber=self.wavenumber, gamma=1, kappa=1, 
                                           alpha=-1, C=1, u=1, n=1, num_field_coeffs=1)
    #prompt fillling of tmp_A_integrals[diff], tmp_A_matrix[j,j'], with a!=0
    #so we can see if it's filled "the wrong way around" when assymetrix
    self.sys_basic.evaluate_A_integrals(amplitude=.9)
    

  def tearDown(self):
    #self.widget.dispose()
    pass

  #def test_fill_A_integrals_dims(self):
    #expected- A_integrals is 1D array with 9 elements
    #self.assertAlmostEqual(self.sys_basic.tmp_A_matrix[1,0].real, 0)

 
class TestLatticeDerivative(unittest.TestCase): 
    pass

class TestLatticeScaling(unittest.TestCase): 
    def setUp(self):
        self.lattice_fine = On_lattice.Lattice(radius=1, wavenumber=1, amplitude=0, gamma=1, kappa=0,
                                            intrinsic_curvature=0, alpha=-1, u=1, C=1, n=1,
                                            temperature=0.01, temperature_lattice=0.01, dims=(500,250))
        self.lattice_coarse = On_lattice.Lattice(radius=1, wavenumber=1, amplitude=0, gamma=1, kappa=0,
                                            intrinsic_curvature=0, alpha=-1, u=1, C=1, n=1,
                                            temperature=0.01, temperature_lattice=0.01,dims=(50,25))
        #fill both with field=1(real) everywhere (overwriting random initiatlize), to check reuslting energy density
        self.lattice_fine.lattice=np.ones((self.lattice_fine.z_len, self.lattice_fine.th_len), dtype=complex)
        self.lattice_coarse.lattice=np.ones((self.lattice_coarse.z_len, self.lattice_coarse.th_len),dtype=complex)
        self.lattice_fine.psi_squared = np.ones((self.lattice_fine.z_len, self.lattice_fine.th_len)) 
        self.lattice_fine.dz = np.zeros((self.lattice_fine.z_len, self.lattice_fine.th_len), dtype=complex)
        self.lattice_fine.dth = np.zeros((self.lattice_fine.z_len, self.lattice_fine.th_len), dtype=complex)
        self.lattice_coarse.psi_squared = np.ones((self.lattice_coarse.z_len, self.lattice_coarse.th_len)) 
        self.lattice_coarse.dz = np.zeros((self.lattice_coarse.z_len, self.lattice_coarse.th_len), dtype=complex)
        self.lattice_coarse.dth = np.zeros((self.lattice_coarse.z_len, self.lattice_coarse.th_len), dtype=complex)

    def test_1_fill(self):
        """check that both were filled with 1s"""
        self.assertEqual(self.lattice_fine.lattice[3,4], 1+0j)
        self.assertEqual(self.lattice_fine.z_len,500)
        self.assertEqual(self.lattice_fine.th_len,250)
        self.assertEqual(self.lattice_coarse.lattice[3,4], 1+0j)
        self.assertEqual(self.lattice_coarse.z_len,50)
        self.assertEqual(self.lattice_coarse.th_len,25)

    def test_pixel_dims(self):
        """test that pixel length and number add up to object length"""
        pixel_len = self.lattice_fine.z_pixel_len
        pixel_number = self.lattice_fine.z_len
        self.assertEqual(pixel_len*pixel_number, 2*math.pi/1)
        pixel_len = self.lattice_fine.th_pixel_len
        pixel_number = self.lattice_fine.th_len
        self.assertEqual(pixel_len*pixel_number, 2*math.pi*1)

    def test_field_energy_fine(self):
        """expected field energy in both cases -1/2 alpha^2/u x surface area = - pi r 2pi / k = -2 pi^2"""
        energy_fine = self.lattice_fine.surface_field_energy(amplitude=0)
        self.assertAlmostEqual(energy_fine, -.5*4*math.pi**2)

    def test_field_energy_coarse(self):
        energy_coarse = self.lattice_coarse.surface_field_energy(amplitude=0)
        self.assertAlmostEqual(energy_coarse, -.5*4*math.pi**2)

    def test_field_energy_scaling(self):
        """check its the same o fine and coarse latttice"""
        energy_fine = self.lattice_fine.surface_field_energy(amplitude=0)
        energy_coarse = self.lattice_coarse.surface_field_energy(amplitude=0)
        self.assertAlmostEqual(energy_fine, energy_coarse)
    
    def test_z_derivative_energy_scaling(self):
        #a field that rises from 0 to 1 (in the middle) and back to 0
        half_z_len=self.lattice_fine.z_len//2
        for z in range(half_z_len):
            self.lattice_fine.lattice[z]=np.ones((self.lattice_fine.th_len), dtype=complex)*(z/half_z_len)
            self.lattice_fine.psi_squared[z] = self.lattice_fine.lattice[z]**2
            #what I think the z-direction slope should be: 1/half_z_len
            self.lattice_fine.dz[z] = np.ones((self.lattice_fine.th_len), dtype=complex)*1/(math.pi)
            self.lattice_fine.dth[z] = np.zeros((self.lattice_fine.th_len), dtype=complex)
        #other half
        second_half_zlen = self.lattice_fine.z_len-half_z_len
        for z in range(half_z_len, self.lattice_fine.z_len):
            self.lattice_fine.lattice[z]=np.ones((self.lattice_fine.th_len), dtype=complex)*(1-((half_z_len-z)/second_half_zlen))
            self.lattice_fine.psi_squared[z] = self.lattice_fine.lattice[z]**2
            self.lattice_fine.dz[z] = np.ones((self.lattice_fine.th_len), dtype=complex)*-1/(math.pi)
            self.lattice_fine.dth[z] = np.zeros((self.lattice_fine.th_len), dtype=complex)
        half_z_len=self.lattice_coarse.z_len//2
        for z in range(half_z_len):
            self.lattice_coarse.lattice[z]=np.ones((self.lattice_coarse.th_len), dtype=complex)*(z/half_z_len)
            self.lattice_coarse.psi_squared[z] = self.lattice_coarse.lattice[z]**2
            #what I think the z-direction slope should be: 1/half object's length
            self.lattice_coarse.dz[z] = np.ones((self.lattice_coarse.th_len), dtype=complex)*1/(math.pi)
            self.lattice_coarse.dth[z] = np.zeros((self.lattice_coarse.th_len), dtype=complex)
        #other half
        second_half_zlen = self.lattice_coarse.z_len-half_z_len
        for z in range(half_z_len, self.lattice_coarse.z_len):
            self.lattice_coarse.lattice[z]=np.ones((self.lattice_coarse.th_len), dtype=complex)*(1-((half_z_len-z)/second_half_zlen))
            self.lattice_coarse.psi_squared[z] = self.lattice_coarse.lattice[z]**2
            self.lattice_coarse.dz[z] = np.ones((self.lattice_coarse.th_len), dtype=complex)*-1/(math.pi)
            self.lattice_coarse.dth[z] = np.zeros((self.lattice_coarse.th_len), dtype=complex)
        #self.assertAlmostEqual(self.lattice_coarse.dz[5][5], self.lattice_fine.dz[5][5])
        #self.assertAlmostEqual(self.lattice_coarse.dz[-5][5], self.lattice_fine.dz[-5][5])
        energy_fine = self.lattice_fine.surface_field_energy(amplitude=0)
        energy_coarse = self.lattice_coarse.surface_field_energy(amplitude=0)
        #expected: ~ (slope =1/pi) * area -1/2 * area = ((1/pi)**2-1/2)*area = (1/pi-1/2)*4*pi**2
        #large deviations allowed - we just don't want wrong scaling by factors 10^n in either direction
        #allow 20% of expected value:
        delta = abs((1/math.pi**2 -.5)* 4*math.pi**2) *.2
        self.assertAlmostEqual(energy_fine, energy_coarse, delta=delta)

 
if __name__ == '__main__':
  unittest.main()

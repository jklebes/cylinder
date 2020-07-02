import scipy.integrate as integrate
import unittest
import run
import numpy as np
import system1D
import system2D
import random
import math
import copy
import metropolis_engine

class TestIntegrandFactors(unittest.TestCase):

  def setUp(self):
    self.sys_basic = system1D.System(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)
    self.sys_small_radius = system1D.System(radius=0.5, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1)
    self.sys_n6 = system1D.System(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=6)

  def tearDown(self):
    #self.widget.dispose()
    pass

class Test_Compare_Calc_Field_Energy(unittest.TestCase):

  def setUp(self):
    self.sys_basic = system1D.System(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1, num_field_coeffs = 3) 
                                                                                 # 1D field z-wavenumber = [-n ,..., 0, ...,  +n]
    self.sys_basic_2d = system2D.System2D(radius=1, wavenumber=1, gamma=1, kappa=1, alpha=-1, C=1, u=1, n=1, num_field_coeffs = (3, 0)) 
                                                                                 # functionally identical system with 2D field 
                                                                                 # theta-wavenumebr = [0] only, z-wavenumber = [-n ,..., 0, ...,  +n]
    random_values = [(random.uniform(-1,1)+random.uniform(-1,1)*1j) for i in range(7)] #random complex coeffs
    self.field_coeffs = np.array(random_values)
    self.field_coeffs_2d = np.array([random_values]) #same values, as 7x1 2D array

  def tearDown(self):
    #self.widget.dispose()
    pass
  
  def test_compare_A_integrals_zero_amplitude(self):
    self.sys_basic.evaluate_A_integrals(amplitude=0)
    self.sys_basic_2d.evaluate_A_integrals(amplitude=0)
    A_matrix_1d = self.sys_basic.tmp_A_matrix
    A_matrix_2d = self.sys_basic_2d.tmp_A_matrix
    self.assertEqual(A_matrix_1d[0,0], A_matrix_2d[0,0])
    self.assertEqual(A_matrix_1d[2,2], A_matrix_2d[2,2])
    self.assertEqual(A_matrix_1d[0,1], A_matrix_2d[0,1])
    self.assertEqual(A_matrix_1d[2,0], A_matrix_2d[2,0])

  def test_compare_A_energy_zero_amplitude(self):
    self.sys_basic.evaluate_A_integrals(amplitude=0)
    self.sys_basic_2d.evaluate_A_integrals(amplitude=0)
    #how energy is calcualated in 1D case 
    A_complex_energy_1d = np.einsum("ji, i, j -> ", self.sys_basic.tmp_A_matrix, self.field_coeffs, self.field_coeffs.conjugate())
    # how energy is calculated in 2d case
    A_complex_energy_2d = 0+0j
    # hybrid einsum and loop for Acc* and Dccc*c* sums
    for beta in range(self.sys_basic_2d.len_arrays_theta):
        A_complex_energy_2d += np.einsum("ji, i, j -> ", self.sys_basic_2d.tmp_A_matrix, self.field_coeffs_2d[beta], self.field_coeffs_2d[beta].conjugate())
    self.assertEqual(A_complex_energy_1d.real, A_complex_energy_2d.real)


  def test_compare_D_energy_zero_amplitude(self):
    self.sys_basic.evaluate_A_integrals(amplitude=0)
    self.sys_basic_2d.evaluate_A_integrals(amplitude=0)
    #how energy is calcualated in 1D case 
    D_complex_energy_1d =  np.einsum("klij, i, j, k, l -> ", self.sys_basic.tmp_D_matrix, self.field_coeffs, self.field_coeffs, self.field_coeffs.conjugate(), self.field_coeffs.conjugate())
    # how energy is calculated in 2d case
    D_complex_energy_2d = 0+0j
    # hybrid einsum and loop for Acc* and Dccc*c* sums
    for beta1 in range(self.sys_basic_2d.len_arrays_theta): # TODO: better to generate once and save allowed tuples [beta1, beta2, betaprime1, betaprime2]
        for beta2 in range(self.sys_basic_2d.len_arrays_theta):
            for betaprime1 in range(self.sys_basic_2d.len_arrays_theta):
               #selection rule beta1 + beta2 == beta'1 + beta'2
               betaprime2 = beta1+beta2-betaprime1
               try: # betaprime2 may be out of range
                 D_complex_energy_2d +=  np.einsum("klij, i, j, k, l -> ", self.sys_basic_2d.tmp_D_matrix, self.field_coeffs_2d[beta1], self.field_coeffs_2d[beta2], self.field_coeffs_2d[betaprime1].conjugate(), self.field_coeffs_2d[betaprime2].conjugate())
               except KeyError:
                 pass
    self.assertEqual(D_complex_energy_1d.real, D_complex_energy_2d.real)

  def test_compare_B_matrices_zero_amplitude(self):
    self.sys_basic.evaluate_B_integrals(amplitude=0)
    self.sys_basic_2d.evaluate_B_integrals(amplitude=0)
    B_matrix_1d = self.sys_basic.tmp_B_integrals
    B_matrix_2d = self.sys_basic_2d.tmp_B_matrix
    self.assertEqual(B_matrix_1d[0,0], B_matrix_2d[0,0,0,0])
    self.assertEqual(B_matrix_1d[1,2], B_matrix_2d[1,2,0,0])
    self.assertEqual(B_matrix_1d[2,2], B_matrix_2d[2,2,0,0])

  def test_compare_A_energy_zero_amplitude(self):
    self.sys_basic.evaluate_B_integrals(amplitude=0)
    self.sys_basic_2d.evaluate_B_integrals(amplitude=0)
    #how energy is calcualated in 1D case 
    B_complex_energy_1d = np.einsum("ij, i, j -> ", self.sys_basic.tmp_B_integrals, self.field_coeffs.conjugate(), self.field_coeffs)
    # how energy is calculated in 2d case
    B_complex_energy_2d = np.einsum("ijab, ai, bj -> ", self.sys_basic_2d.tmp_B_matrix, self.field_coeffs_2d.conjugate(), self.field_coeffs_2d)
    self.assertEqual(B_complex_energy_1d.real, B_complex_energy_2d.real)

  def test_field_energy_zero_amplitude(self):
    energy1D= self.sys_basic.calc_field_energy_amplitude_change(field_coeffs=self.field_coeffs, amplitude=0)
    energy2D= self.sys_basic_2d.calc_field_energy_amplitude_change(field_coeffs=self.field_coeffs_2d, amplitude=0)
    self.assertAlmostEqual(energy1D, energy2D)

  def save_matrices_zero_amplitude(self):
    energy1D_tmp= self.sys_basic.calc_field_energy_amplitude_change(field_coeffs=self.field_coeffs, amplitude=0)
    energy2D_tmp= self.sys_basic_2d.calc_field_energy_amplitude_change(field_coeffs=self.field_coeffs_2d, amplitude=0)
    self.sys_basic.save_temporary_matrices()
    self.sys_basic_2d.save_temporary_matrices()
    energy1D= self.sys_basic.calc_field_energy(field_coeffs=self.field_coeffs)
    energy2D= self.sys_basic_2d.calc_field_energy(field_coeffs=self.field_coeffs_2d)
    self.assertAlmostEqual(energy1D, energy2D)
  
  def test_compare_A_integrals(self):
    self.sys_basic.evaluate_A_integrals(amplitude=-.8)
    self.sys_basic_2d.evaluate_A_integrals(amplitude=-.8)
    A_matrix_1d = self.sys_basic.tmp_A_matrix
    A_matrix_2d = self.sys_basic_2d.tmp_A_matrix
    self.assertEqual(A_matrix_1d[0,0], A_matrix_2d[0,0])
    self.assertEqual(A_matrix_1d[2,2], A_matrix_2d[2,2])
    self.assertEqual(A_matrix_1d[0,1], A_matrix_2d[0,1])
    self.assertEqual(A_matrix_1d[2,0], A_matrix_2d[2,0])

  def test_compare_A_energy(self):
    self.sys_basic.evaluate_A_integrals(amplitude=.22)
    self.sys_basic_2d.evaluate_A_integrals(amplitude=.22)
    #how energy is calcualated in 1D case 
    A_complex_energy_1d = np.einsum("ji, i, j -> ", self.sys_basic.tmp_A_matrix, self.field_coeffs, self.field_coeffs.conjugate())
    # how energy is calculated in 2d case
    A_complex_energy_2d = 0+0j
    # hybrid einsum and loop for Acc* and Dccc*c* sums
    for beta in range(self.sys_basic_2d.len_arrays_theta):
        A_complex_energy_2d += np.einsum("ji, i, j -> ", self.sys_basic_2d.tmp_A_matrix, self.field_coeffs_2d[beta], self.field_coeffs_2d[beta].conjugate())
    self.assertEqual(A_complex_energy_1d.real, A_complex_energy_2d.real)


  def test_compare_D_energy(self):
    self.sys_basic.evaluate_A_integrals(amplitude=-.5)
    self.sys_basic_2d.evaluate_A_integrals(amplitude=-.5)
    #how energy is calcualated in 1D case 
    D_complex_energy_1d =  np.einsum("klij, i, j, k, l -> ", self.sys_basic.tmp_D_matrix, self.field_coeffs, self.field_coeffs, self.field_coeffs.conjugate(), self.field_coeffs.conjugate())
    # how energy is calculated in 2d case
    D_complex_energy_2d = 0+0j
    # hybrid einsum and loop for Acc* and Dccc*c* sums
    for beta1 in range(self.sys_basic_2d.len_arrays_theta): # TODO: better to generate once and save allowed tuples [beta1, beta2, betaprime1, betaprime2]
        for beta2 in range(self.sys_basic_2d.len_arrays_theta):
            for betaprime1 in range(self.sys_basic_2d.len_arrays_theta):
               #selection rule beta1 + beta2 == beta'1 + beta'2
               betaprime2 = beta1+beta2-betaprime1
               try: # betaprime2 may be out of range
                 D_complex_energy_2d +=  np.einsum("klij, i, j, k, l -> ", self.sys_basic_2d.tmp_D_matrix, self.field_coeffs_2d[beta1], self.field_coeffs_2d[beta2], self.field_coeffs_2d[betaprime1].conjugate(), self.field_coeffs_2d[betaprime2].conjugate())
               except KeyError:
                 pass
    self.assertEqual(D_complex_energy_1d.real, D_complex_energy_2d.real)

  def test_compare_B_matrices(self):
    self.sys_basic.evaluate_B_integrals(amplitude=.6)
    self.sys_basic_2d.evaluate_B_integrals(amplitude=.6)
    B_matrix_1d = self.sys_basic.tmp_B_integrals
    B_matrix_2d = self.sys_basic_2d.tmp_B_matrix
    self.assertAlmostEqual(B_matrix_1d[3,3], B_matrix_2d[3,3,0,0]) #central element - just A_theta part
    self.assertAlmostEqual(B_matrix_1d[0,0], B_matrix_2d[0,0,0,0])
    self.assertAlmostEqual(B_matrix_1d[1,2], B_matrix_2d[1,2,0,0])
    self.assertAlmostEqual(B_matrix_1d[2,2], B_matrix_2d[2,2,0,0])

  def test_compare_A_energy(self):
    self.sys_basic.evaluate_B_integrals(amplitude=-.23)
    self.sys_basic_2d.evaluate_B_integrals(amplitude=-.23)
    #how energy is calcualated in 1D case 
    B_complex_energy_1d = np.einsum("ij, i, j -> ", self.sys_basic.tmp_B_integrals, self.field_coeffs.conjugate(), self.field_coeffs)
    # how energy is calculated in 2d case
    B_complex_energy_2d = np.einsum("ijab, ai, bj -> ", self.sys_basic_2d.tmp_B_matrix, self.field_coeffs_2d.conjugate(), self.field_coeffs_2d)
    self.assertAlmostEqual(B_complex_energy_1d.real, B_complex_energy_2d.real)

  def test_field_energy(self):
    energy1D= self.sys_basic.calc_field_energy_amplitude_change(field_coeffs=self.field_coeffs, amplitude=.4)
    energy2D= self.sys_basic_2d.calc_field_energy_amplitude_change(field_coeffs=self.field_coeffs_2d, amplitude=.4)
    self.assertAlmostEqual(energy1D, energy2D)

  def test_save_matrices(self):
    energy1D_tmp= self.sys_basic.calc_field_energy_amplitude_change(field_coeffs=self.field_coeffs, amplitude=-.35)
    energy2D_tmp= self.sys_basic_2d.calc_field_energy_amplitude_change(field_coeffs=self.field_coeffs_2d, amplitude=-.35)
    self.sys_basic.save_temporary_matrices()
    self.sys_basic_2d.save_temporary_matrices()
    energy1D= self.sys_basic.calc_field_energy(field_coeffs=self.field_coeffs)
    energy2D= self.sys_basic_2d.calc_field_energy(field_coeffs=self.field_coeffs_2d)
    self.assertAlmostEqual(energy1D, energy2D)


if __name__ == '__main__':
  unittest.main()

import scipy.integrate as integrate
import unittest
import run
import numpy as np
import surfaces_and_fields.system_cylinder1D as cylinder1D
import random
import math
import copy

class TestAMatrixOneElement(unittest.TestCase):

  def setUp(self):
    """
    with num_field_coeffs=0, there will be 1 field mode j=0 and one 
    matrix element of A j=0, j'=0
    """
    self.radius=1
    self.wavenumber=1
    self.expected_A_integral_0=math.pi*2*self.radius/self.wavenumber +0j
    self.sys_basic = cylinder1D.Cylinder1D(radius=self.radius, wavenumber=self.wavenumber, gamma=1, kappa=1, 
                                           alpha=-1, C=1, u=1, n=1, num_field_coeffs=0)
    #prompt fillling of tmp_A_integrals[diff], tmp_A_matrix[j,j']
    self.sys_basic.evaluate_A_integrals(amplitude=0)
    


  def tearDown(self):
    #self.widget.dispose()
    pass

  def test_fill_A_integrals(self):
    #expected- A_integrals is 1D array with 1 element
    self.assertEqual(len(self.sys_basic.tmp_A_integrals), 1) 
    #the element A_integrals[diff=0] is the surface area of the cylinder: 
    # 2pi*r_0*2pi/k except for a factor of 2pi (as complex number)
    self.assertEqual(self.sys_basic.tmp_A_integrals[0], self.expected_A_integral_0)

  def test_fill_A_matrix(self):
    #expected: 2D array with 1 element 
    self.assertEqual(self.sys_basic.tmp_A_matrix.shape, (1,1)) 
    self.assertEqual(self.sys_basic.tmp_A_matrix[0,0], self.expected_A_integral_0)

class TestAMatrix9Element(unittest.TestCase):

  def setUp(self):
    """
    with num_field_coeffs=1, the A matrix is 3x3
    """
    self.radius=1
    self.wavenumber=1
    self.expected_A_integral_0=math.pi*2*self.radius/self.wavenumber
    self.sys_basic = cylinder1D.Cylinder1D(radius=self.radius, wavenumber=self.wavenumber, gamma=1, kappa=1, 
                                           alpha=-1, C=1, u=1, n=1, num_field_coeffs=1)
    #prompt evaluation of integrals, fill of matrix elemnts
    self.sys_basic.evaluate_A_integrals(amplitude=0)
    

  def tearDown(self):
    #self.widget.dispose()
    pass

  def test_fill_A_integrals_dims(self):
    #expected- A_integrals is 1D array with 9 elements
    # differences between 4 coeffs randing from -1 to 1 can go from -4 to +4
    self.assertEqual(len(self.sys_basic.tmp_A_integrals), 9) 

  def test_fill_A_matrix_dims(self):
    #expected: 2D array with 9: indexed j from -1 to 1; j' from -1 to 1
    self.assertEqual(self.sys_basic.tmp_A_matrix.shape, (3,3)) 

  def test_fill_A_integrals_0element(self):
    #expected- A_integrals is 1D array with 9 element, middle element j-j'=0 is surfcae area
    self.assertEqual(self.sys_basic.tmp_A_integrals[4], self.expected_A_integral_0)

  def test_fill_A_matrix_0element(self):
    #expected: 2D array 3x3 with middle element j=0, j'=0 is surface area
    self.assertEqual(self.sys_basic.tmp_A_matrix[1,1], self.expected_A_integral_0)
    #elemtns j=-1,j'=-1, j==j' and j=+1,j'=+1, j==j' should also be surface area
    self.assertEqual(self.sys_basic.tmp_A_matrix[0,0], self.expected_A_integral_0)
    self.assertEqual(self.sys_basic.tmp_A_matrix[2,2], self.expected_A_integral_0)

class TestAMatrixAssymetric(unittest.TestCase):

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

  def test_fill_A_integrals_dims(self):
    #expected- A_integrals is 1D array with 9 elements
    # differences between 4 coeffs randing from -1 to 1 can go from -4 to +4
    self.assertEqual(len(self.sys_basic.tmp_A_integrals), 9) 

  def test_fill_A_matrix_dims(self):
    #expected: 2D array with 9: indexed j from -1 to 1; j' from -1 to 1
    self.assertEqual(self.sys_basic.tmp_A_matrix.shape, (3,3)) 

  def test_fill_A_matrix(self):
    #expected order of filling from A_integrals:
    # A_integral[j-j'], where j is wavevector of Psi_j and j' of h.c. Psi_j'
    # and A_matrix is indexed [j,j']
    self.assertEqual(self.sys_basic.tmp_A_integrals[(1-0)+4], self.sys_basic.tmp_A_matrix[1+1,0+1]) 
    self.assertEqual(self.sys_basic.tmp_A_integrals[(0-1)+4], self.sys_basic.tmp_A_matrix[0+1,1+1]) 
    self.assertEqual(self.sys_basic.tmp_A_integrals[(0-0)+4], self.sys_basic.tmp_A_matrix[0+1,0+1]) 
    self.assertEqual(self.sys_basic.tmp_A_integrals[(-1-0)+4], self.sys_basic.tmp_A_matrix[-1+1,0+1]) 
    self.assertEqual(self.sys_basic.tmp_A_integrals[(0-(-1))+4], self.sys_basic.tmp_A_matrix[0+1,-1+1]) 

  def test_fill_A_integrals_0element(self):
    #expected- middle element of 3x3 matrix is middle element of A_integrals (len 9)
    self.assertEqual(self.sys_basic.tmp_A_integrals[4], self.sys_basic.tmp_A_matrix[1,1]) 
    

  def test_fill_A_matrix_hermitian(self):
    conjugate_transpose = self.sys_basic.tmp_A_matrix.conjugate().transpose()
    self.assertEqual(self.sys_basic.tmp_A_matrix[0,1], conjugate_transpose[0,1])
  
  def test_fill_A_matrix_real_diagonal(self):
    self.assertAlmostEqual(self.sys_basic.tmp_A_matrix[2,2].imag, 0)
    self.assertAlmostEqual(self.sys_basic.tmp_A_matrix[1,1].imag, 0)
    self.assertAlmostEqual(self.sys_basic.tmp_A_matrix[0,0].imag, 0)

  def test_fill_A_matrix_imag_offdiagonal(self):
    self.assertAlmostEqual(self.sys_basic.tmp_A_matrix[1,2].real, 0)
    self.assertAlmostEqual(self.sys_basic.tmp_A_matrix[0,1].real, 0)
    self.assertAlmostEqual(self.sys_basic.tmp_A_matrix[1,0].real, 0)
  
  def test_A_matrix_einsum(self):
    field_coeffs = np.array([(random.uniform(-1,1)+random.uniform(-1,1)*1j) for i in range(3)])
    A_complex_energy_einsum = np.einsum("ij, i, j -> ", self.sys_basic.tmp_A_matrix, field_coeffs, field_coeffs.conjugate()) # watch out for how A,D are transpose of expected
    A_complex_energy_loop = 0+0j
    for j in range(3): #j
      for j2 in range(3): #j', the one associacted with complex conjugate
        A_complex_energy_loop+= field_coeffs[j]*field_coeffs[j2].conjugate()*self.sys_basic.tmp_A_matrix[j,j2]
    self.assertAlmostEqual(A_complex_energy_einsum, A_complex_energy_loop)


if __name__ == '__main__':
  unittest.main()

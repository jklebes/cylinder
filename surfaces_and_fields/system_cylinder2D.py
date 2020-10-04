import math
import scipy.integrate as integrate
import numpy as np
import surfaces_and_fields.system_cylinder1D as system_cylinder1D

class Cylinder2D(system_cylinder1D.Cylinder1D):
  """
  System: Cylider with 2D Fourier decomposed field
  subclass of Cylinder (with no field), adds 2D field
  A physics system containing functions that return an energy for given real , complex parameter values
  Representing a section of sinusoidally perturbed cylindrical surface / surface of revolution r(z) = r(a) (1+a sin(kz))
  plus a complex-valued field over the two-dimensional surface, fourier decomposed into an array of modes indexed (j, beta)
  """
  def __init__(self, wavenumber, radius, alpha, C, u, n, kappa, gamma, intrinsic_curvature=0, num_field_coeffs=(0,0)):
    super().__init__(wavenumber=wavenumber, radius=radius, alpha=alpha, C=C, u=u, n=n, 
                    kappa=kappa, gamma=gamma, intrinsic_curvature=intrinsic_curvature)
    self.num_field_coeffs_z, self.num_field_coeffs_theta = num_field_coeffs
    self.len_arrays_z = 2*self.num_field_coeffs_z+1
    self.len_arrays_theta = 2*self.num_field_coeffs_theta+1
    ############ change to 2D : A and D matrices ########################
    # even though A matrix (->D matrix) is now 4D A_{jj'beta beta'}, still only calculate and save matrix A_{jj'}.  We don't need to store all the values
    # we get for different combinations of beta, beta' in a matrix because they are mostly zero, 
    #  A_{jj'beta beta'} = (2 pi A_{j j'} if beta == beta' else 0)
    #  instead implement this selection rule later
    ############## change to 2D : B matrices ###########################
    # fill the 4D object B_{j j beta beta'}

    self.tmp_A_integrals = np.zeros(4*self.len_arrays_z-3, dtype=complex) #initialize 1D np array of complex 
    #A_integrals list is longer than others because it covers range of possible differences in index between 2 or 4 field coeffs 
    self.tmp_B_matrix = np.zeros((self.len_arrays_z, self.len_arrays_z, self.len_arrays_theta, self.len_arrays_theta), dtype=complex) # 4D array j, j', beta, beta'
    # A and D matrices theoretically should have +2, +4 theta dimensions, but since the values are 0 or same according to selection rule its unnecessary to save these
    self.tmp_A_matrix = np.zeros((self.len_arrays_z, self.len_arrays_z), dtype=complex) #initialize np complex array nxn
    self.tmp_D_matrix = np.zeros((self.len_arrays_z, self.len_arrays_z, self.len_arrays_z, self.len_arrays_z),dtype=complex) # same 4D
    self.tmp_A_integrals_0 = 0
    # location to save long-term copies of the same
    self.B_matrix = np.zeros((self.len_arrays_z, self.len_arrays_z, self.len_arrays_theta, self.len_arrays_theta), dtype=complex) # 4D array j, j', beta, beta'
    self.A_matrix = np.zeros((self.len_arrays_z, self.len_arrays_z), dtype=complex) #initialize np complex array nxn
    self.D_matrix = np.zeros((self.len_arrays_z, self.len_arrays_z, self.len_arrays_z, self.len_arrays_z),dtype=complex) 
    
    #for quartic cross term selection rule:  generate list of allowed (nonorthogonal) theta-coefficient combinations
    # beta + beta2 = beta' + beta2'
    # (index in range 0..len_arrays) = (index beta in -range num_field_coeffs..num_field_coeffs) + const
    # selection rule eqaulvalent under either indexing scheme
    # (beta+const) + (beta2 + const) = (beta'+const) + (beta2' + const)
    self.quartic_selection_rule = self.generate_quartic_combinations(self.len_arrays_theta)

  def generate_quartic_combinations(self,len_arrays):
    # only done once, doesnt matter if slow
    allowed_combos = []
    for index1 in range(len_arrays):
      for index2 in range(len_arrays):
        for index3 in range(len_arrays):
          for index4 in range(len_arrays):
            if index1+index2 == index3+index4:
              allowed_combos.append((index1, index2, index3, index4))
    return allowed_combos
 

  ######## common terms in integrals ###########
  # in parent classes

  ############# B integrands different in 2D field case ############
  
  def B_integrand_real_part(self, i, j, beta, amplitude, z):
    if False and amplitude == 0: #TODO fix, returns wrong answers
      # TODO : check this
      z_bending = (i * j * self.wavenumber ** 2 * math.cos((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.radius)  # sqrt(g_theta theta) = radius
      theta_bending = (beta**2 * #math.cos(beta-beta') =1
                      1/self.radius) #to match index raising below, should be 1/r - now correct?  
      return (z_bending+theta_bending)
    else:
      simplified_integrand  = math.cos((i-j)*self.wavenumber*z)*(
                              # |d_z Psi |^2 part:
                              self.wavenumber**2 * i * j * self.sqrt_g_z(amplitude,z) * self.sqrt_g_theta(amplitude, z)+
                              # the following are all theta parts, with index raise eta^{theta theta} combined with sqrt(g)
                              # the squared part expands to three terms: beta beta', -2beta n A_theta, and n**2 A_theta**2
                              # * math.cos(beta-beta') =1
                              (beta - self.n*self.A_theta(amplitude, z))**2 * self.sqrt_g_z(amplitude, z) / self.sqrt_g_theta(amplitude, z))
                              
      return simplified_integrand
 
  
  def B_integrand_img_part(self, i, j, beta, amplitude, z):
    if False and amplitude == 0:
      # TODO : check this
      z_bending = (i * j * self.wavenumber ** 2 * math.sin((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.radius)  # sqrt(g_theta theta) = radius
      theta_bending = 0#(beta**2 * #math.cos(beta-beta') =0
                      #1/self.radius)  # now correct? 
      return (z_bending+theta_bending)
    else:
      simplified_integrand  = math.sin((i-j)*self.wavenumber*z)*( #sin: img part of e^i...
                              # |d_z Psi |^2 part:
                              self.wavenumber**2 * i * j * self.sqrt_g_z(amplitude,z) * self.sqrt_g_theta(amplitude, z) +
                              # these are all theta parts, with index raise eta^{theta theta} combined with sqrt(g)
                              # the squared part expands to three terms: beta beta', -2beta n A_theta, and n**2 A_theta**2
                              (beta - self.n*self.A_theta(amplitude, z))**2 * self.sqrt_g_z(amplitude, z) / self.sqrt_g_theta(amplitude, z))
                              
      return simplified_integrand
 

  def evaluate_A_integrals(self, amplitude):
    for diff in range(-4*self.num_field_coeffs_z,  4*self.num_field_coeffs_z+1):
      img_part, error = integrate.quad(lambda z: self.A_integrand_img_part(diff, amplitude, z),
                                       0, 2 * math.pi / self.wavenumber)
      real_part, error = integrate.quad(lambda z: self.A_integrand_real_part(diff, amplitude, z),
                                        0, 2 * math.pi / self.wavenumber)
      self.tmp_A_integrals[diff+4*self.num_field_coeffs_z] = complex(real_part, img_part) # in A_integrals list results are stored by diff between coefficient indices from -n to n
                                                            # places 0 to _ in list correspond to differences -2n+1 to 2n-1 (incl)
    ## fill matrix A[i,j] from list A[diff], where i j k are matrix indices from 0 to 2n (corresponding to ordered list of coefficient's indices from -n to n)
    ## fill matrix D[i,j,k,l] where D[i+j-l-k]=A[diff]
    l = self.len_arrays_z
    n_coeffs=self.num_field_coeffs_z
    # save to tmp location when first calcualted with new amplitude - gets transferred to permanent matrices accesed by calc_field_energy when amplitude is accepted
    for i in range(l): 
      self.tmp_A_matrix[i,::-1] = self.tmp_A_integrals[i+2*n_coeffs:i+4*n_coeffs+1]
      for j in range(l):
        for k in range(l):
          self.tmp_D_matrix[i,j,k,::-1] =  self.tmp_A_integrals[i+j-k+2*n_coeffs:i+j-k+4*n_coeffs+1] 
    self.tmp_A_integrals_0 = self.tmp_A_integrals[0]


  def evaluate_A_integral_0(self, amplitude):
    #useful in no-field simulations
    img_part, error = integrate.quad(lambda z: self.A_integrand_img_part(0, amplitude, z),
                                     0, 2 * math.pi / self.wavenumber)
    assert (math.isclose(img_part, 0, abs_tol=1e-9))
    real_part, error = integrate.quad(lambda z: self.A_integrand_real_part(0, amplitude, z),
                                      0, 2 * math.pi / self.wavenumber)
    # this is usually done for surface area - no need to fill into A_matrix
    self.tmp_A_integrals_0 = complex(real_part, img_part)
  
  def evaluate_B_integrals(self, amplitude):
    # start with all 0s
    self.tmp_B_matrix = np.zeros((self.len_arrays_z, self.len_arrays_z, self.len_arrays_theta, self.len_arrays_theta), dtype=complex) 
    # fill with value where beta = beta'
    for i in range(-self.num_field_coeffs_z, self.num_field_coeffs_z+1):
      for j in range(-self.num_field_coeffs_z, self.num_field_coeffs_z+1):
        for beta in range(-self.num_field_coeffs_theta, self.num_field_coeffs_theta+1):
          #fill element i,j,beta,beta'=beta  (elements beta' != beta stay 0)
          img_part, error = integrate.quad(lambda z: self.B_integrand_img_part(i, j, beta, amplitude, z),
                                         0, 2 * math.pi / self.wavenumber)
          real_part, error = integrate.quad(lambda z: self.B_integrand_real_part(i, j, beta, amplitude, z),
                                          0, 2 * math.pi / self.wavenumber) # integrands demand coefficient's indices in range -n to n
          B_matrix_element = complex(real_part, img_part)
          self.tmp_B_matrix[i+self.num_field_coeffs_z,j+self.num_field_coeffs_z,beta+self.num_field_coeffs_theta,beta+self.num_field_coeffs_theta]=B_matrix_element
 

  ############# calc energy ################
  def calc_field_energy(self, field_coeffs):
    """
    when amplitude doesn't chage: look at stored values self.X_matrix
    """
    A_complex_energy = 0+0j
    D_complex_energy = 0+0j
    # hybrid einsum and loop for Acc* and Dccc*c* sums
    for beta in range(self.len_arrays_theta):
    	A_complex_energy += np.einsum("ij, i, j -> ", self.A_matrix, field_coeffs[beta], field_coeffs[beta].conjugate())
    for (index1, index2, index3, index4) in self.quartic_selection_rule:
      D_complex_energy +=  np.einsum("ijkl, i, j, k, l -> ", self.D_matrix, field_coeffs[index1], field_coeffs[index2], field_coeffs[index3].conjugate(), field_coeffs[index4].conjugate())
    # 4D einsum for B integrals: energy term =  B_{j j' beta beta'}c_{beta j}c*_{beta' j'} (einstein summation convention)
    B_complex_energy = np.einsum("ijab, ai, bj -> ", self.B_matrix, field_coeffs, field_coeffs.conjugate()) #why the backwards list of indices? array field coeffs is jiven in transpose (beta,j)?
    assert (math.isclose(A_complex_energy.imag, 0, abs_tol=1e-7))
    assert (math.isclose(B_complex_energy.imag, 0, abs_tol=1e-7))
    assert (math.isclose(D_complex_energy.imag, 0, abs_tol=1e-7))
    return  self.alpha * A_complex_energy.real + self.C * B_complex_energy.real + 0.5 * self.u * D_complex_energy.real

  def save_temporary_matrices(self):
    #if accepted, save results of evaluation with proposed amplitude to more permanent location
    self.A_matrix = self.tmp_A_matrix
    self.D_matrix = self.tmp_D_matrix
    self.B_matrix = self.tmp_B_matrix

 
  def calc_field_energy_amplitude_change(self, amplitude, field_coeffs):
    """
    to be used on steps where amplitude change is proposed,
    to get resulting proposed energy of (same) field on proposed surface shape
    """
    self.evaluate_A_integrals(amplitude)
    self.evaluate_B_integrals(amplitude) #only evaluates with new amplitude -> tmp storage location
    A_complex_energy = 0+0j
    D_complex_energy = 0+0j
    # hybrid einsum and loop for Acc* and Dccc*c* sums
    for beta in range(self.len_arrays_theta):
    	A_complex_energy += np.einsum("ij, i, j -> ", self.tmp_A_matrix, field_coeffs[beta], field_coeffs[beta].conjugate())
    for (index1, index2, index3, index4) in self.quartic_selection_rule:
      D_complex_energy +=  np.einsum("ijkl, i, j, k, l -> ", self.tmp_D_matrix, field_coeffs[index1], field_coeffs[index2], field_coeffs[index3].conjugate(), field_coeffs[index4].conjugate())
    # 4D einsum for B integrals: energy term =  B_{j j' beta beta'}c_{beta j}c*_{beta' j'} (einstein summation convention)
    B_complex_energy = np.einsum("ijab, ai, bj -> ", self.tmp_B_matrix, field_coeffs, field_coeffs.conjugate()) 
    assert (math.isclose(A_complex_energy.imag, 0, abs_tol=1e-7))
    #print("B complex energy", B_complex_energy)
    #print("from einsum", self.tmp_B_matrix, field_coeffs, field_coeffs.conjugate())
    assert (math.isclose(B_complex_energy.imag, 0, abs_tol=1e-7))
    #print("D_complex_energy", D_complex_energy, field_coeffs)
    assert (math.isclose(D_complex_energy.imag, 0, abs_tol=1e-7)) 
    return  self.alpha * A_complex_energy.real + self.C * B_complex_energy.real + 0.5 * self.u * D_complex_energy.real

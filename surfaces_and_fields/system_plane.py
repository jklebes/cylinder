import math
import scipy.integrate as integrate
import numpy as np

class SystemPlane():

  def __init__(self, wavenumber, radius, alpha, C, u, n, kappa, gamma, num_field_coeffs):
    assert (all(map(lambda x: x >= 0, [wavenumber, radius, C, u, kappa, gamma, num_field_coeffs])))
    assert (alpha <= 0)
    self.wavenumber = wavenumber
    self.width = radius # not used, everything would be multiplied by width.  Left out like 2pi in cylinder.
    self.alpha = alpha
    self.C = C
    self.u = u
    self.n = n
    self.kappa = kappa
    self.gamma = gamma
    self.num_field_coeffs = num_field_coeffs
    self.len_arrays = 2*self.num_field_coeffs+1
    self.A_integrals = np.zeros(4*self.len_arrays-3, dtype=complex) #initialize 1D np array of complex 
    self.B_integrals = np.zeros((self.len_arrays, self.len_arrays), dtype=complex) # 2D np array of complex
    self.A_matrix = np.zeros((self.len_arrays, self.len_arrays), dtype=complex) #initialize np complex array nxn
    self.D_matrix = np.zeros((self.len_arrays, self.len_arrays, self.len_arrays, self.len_arrays),dtype=complex) # same 4D

  ######## common terms in integrals ###########

  # sqrt_g_thth is 1 - no distortion of area element in theta direction
  # A_theta = 0 
  # no radius/width rescale

  def sqrt_g_z(self, amplitude, z):
    # stretching of length of area element in z direction
    return math.sqrt(1 + ( amplitude * self.wavenumber * math.cos(self.wavenumber * z)) ** 2)


  ########## evaluates integrals ##########
  def A_integrand_img_part(self, diff, amplitude, z):
    if amplitude == 0:
      return math.sin(diff * self.wavenumber * z)   # img part of e^(i ...)
    else:
      return (math.sin(diff * self.wavenumber * z) *  # real part of e^(i ...)
              self.sqrt_g_z(amplitude, z))

  def A_integrand_real_part(self, diff, amplitude, z):
    """
    :param diff: coefficient difference (i-j) or (i+i'-j-j') in e^i(i-j)kz
    :param amplitude: a
    :return: float
    """
    if amplitude == 0:
      return (math.cos(diff * self.wavenumber * z)  # real part of e^(i ...)
             )  # sqrt(g_theta theta) = radius
    else:
      return (math.cos(diff * self.wavenumber * z) *  # real part of e^(i ...)
              self.sqrt_g_z(amplitude, z))

  def B_integrand_img_part(self, i, j, amplitude, z):
    if amplitude == 0:
      z_part = i * j * self.wavenumber ** 2 * math.sin((i - j) * self.wavenumber * z)   # |d e^... |^2
      return z_part
    else:
      z_part = (i * j * self.wavenumber ** 2 * math.sin((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.sqrt_g_z(amplitude, z))
      return z_part

  def B_integrand_real_part(self, i, j, amplitude, z):
    if amplitude == 0:
      z_part = i * j * self.wavenumber ** 2 * math.cos((i - j) * self.wavenumber * z)   # |d e^... |^2
      return z_part
    else:
      z_part = (i * j * self.wavenumber ** 2 * math.cos((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.sqrt_g_z(amplitude, z) )
      return z_part
 
  def Kzz_integrand(self, amplitude, z):
    #redo
    return (amplitude*self.wavenumber**2*math.sin(self.wavenumber*z))**2 /((self.sqrt_g_z(amplitude,z))**5)
  

  def evaluate_A_integrals(self, amplitude, num_field_coeffs):
    for diff in range(-4*self.num_field_coeffs,  4*self.num_field_coeffs+1):
      img_part, error = integrate.quad(lambda z: self.A_integrand_img_part(diff, amplitude, z),
                                       0, 2 * math.pi / self.wavenumber)
      real_part, error = integrate.quad(lambda z: self.A_integrand_real_part(diff, amplitude, z),
                                        0, 2 * math.pi / self.wavenumber)
      self.A_integrals[diff+4*self.num_field_coeffs] = complex(real_part, img_part) # in A_integrals list results are stored by diff between coefficient indices from -n to n
                                                            # places 0 to _ in list correspond to differences -2n+1 to 2n-1 (incl)
    ## fill matrix A[i,j] from list A[diff], where i j k are matrix indices from 0 to 2n (corresponding to ordered list of coefficient's indices from -n to n)
    ## fill matrix D[i,j,k,l] where D[i+j-l-k]=A[diff]
    l = self.len_arrays
    for i in range(l): 
      self.A_matrix[i] = self.A_integrals[2*l-i-2:3*l-2-i]
      for j in range(l):
        for k in range(l):
          self.D_matrix[i,j,k] =  self.A_integrals[k-i-j+2*l-2:k-i-j+3*l-2] 


  def evaluate_A_integral_0(self, amplitude):
    img_part, error = integrate.quad(lambda z: self.A_integrand_img_part(0, amplitude, z),
                                     0, 2 * math.pi / self.wavenumber)
    assert (math.isclose(img_part, 0, abs_tol=1e-9))
    real_part, error = integrate.quad(lambda z: self.A_integrand_real_part(0, amplitude, z),
                                      0, 2 * math.pi / self.wavenumber)
    self.A_integrals[0] = complex(real_part, img_part) # this is usually done for surface area - no need to fill into A_matrix

  def evaluate_B_integrals(self, amplitude, num_field_coeffs):
    for i in range(-num_field_coeffs, num_field_coeffs + 1):
      for j in range(-num_field_coeffs, num_field_coeffs + 1):
        img_part, error = integrate.quad(lambda z: self.B_integrand_img_part(i, j, amplitude, z),
                                         0, 2 * math.pi / self.wavenumber)
        real_part, error = integrate.quad(lambda z: self.B_integrand_real_part(i, j, amplitude, z),
                                          0, 2 * math.pi / self.wavenumber) # integrands demand coefficient's indices in range -n to n
        self.B_integrals[i+self.num_field_coeffs, j+self.num_field_coeffs] = complex(real_part, img_part) # while results are stored in order in array with indices 0 to 2n

  ############# calc energy ################
  def calc_field_energy(self, state, amplitude_change=False):
    """ einsum is (at least 10x) faster than loops! even with constructing matrix A from list
    same for D part even though 4d matrix is constructed from list every time: 10x faster at 3(-1to1) coeffs
    much more for longer set of coeffs
    :param field_coeffs: list of complex values as np array from lowest to highest index, i. e. [c_-n, ..., c_0 , ... c_n] value at index 0 is c_-n
    A, B, D matrices are orderd on the same scheme - because np array is faster than dict explcitly stating indices -n .. 0 .. n for matrix muktiplication
    """
    amplitude = state[0].real
    assert(state[0].imag == 0)
    field_coeffs = state[1:]
    if amplitude_change: #TODO: optimize further -
                          # don't check truth value every time
                          # separate out fct to recacl matrices and call externally
      print("if you mean to reevaluate integrals on amplitude change, call evaluate_integrals externally")
    #Matrix products of the form c_i A_ij c*_j
    A_complex_energy = np.einsum("ji, i, j -> ", self.A_matrix, field_coeffs, field_coeffs.conjugate()) # watch out for how A,D are transpose of expected
                                                                              # because its the faster way to construct them
    B_complex_energy = np.einsum("ij, i, j -> ", self.B_integrals, field_coeffs.conjugate(), field_coeffs) # B is filled directly with outcomes of B_integrals, not transposed
    #print(field_coeffs)
    D_complex_energy =  np.einsum("klij, i, j, k, l -> ", self.D_matrix, field_coeffs, field_coeffs, field_coeffs.conjugate(), field_coeffs.conjugate())
    assert (math.isclose(A_complex_energy.imag, 0, abs_tol=1e-7))
    assert (math.isclose(B_complex_energy.imag, 0, abs_tol=1e-7))
    assert (math.isclose(D_complex_energy.imag, 0, abs_tol=1e-7)) # this means either D matrix calculation is wrong OR proposed field coefficeitns got very big
    return self.alpha * A_complex_energy.real + self.C * B_complex_energy.real + 0.5 * self.u * D_complex_energy.real

  def calc_bending_energy(self, amplitude):
    """
    calculate bending as (K_i^i)**2.  Gaussian curvature and cross term 2 K_th^th K_z^z are omitted due to gauss-bonnet theorem.
    """
    if amplitude == 0:
      return 0 
    else:
      Kzz_integral, error = integrate.quad(lambda z: self.Kzz_integrand(amplitude, z), 0, 2 * math.pi / self.wavenumber)
      return (Kzz_integral)

  def evaluate_integrals(self, amplitude):
    self.evaluate_A_integrals(amplitude, self.num_field_coeffs)
    self.evaluate_B_integrals(amplitude, self.num_field_coeffs)

  def calc_surface_energy(self, amplitude, amplitude_change=True):
    """
    energy from surface tension * surface area, + bending rigidity constant * mean curvature squared
    """
    if amplitude_change:
      print("if you mean to reevaluate integrals on amplitude change, call evaluate_integrals")
    #print("A integral 0", self.A_integrals[0].real)
    # A_integrals[0] is just surface area
    return self.gamma * self.A_integrals[0].real + self.kappa * self.calc_bending_energy(amplitude)

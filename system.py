import math
import scipy.integrate as integrate
import numpy as np

class System():

  def __init__(self, wavenumber, radius, alpha, C, u, n, kappa, gamma, num_field_coeffs):
    assert (all(map(lambda x: x >= 0, [wavenumber, radius, C, u, kappa, gamma, num_field_coeffs])))
    assert (alpha <= 0)
    self.wavenumber = wavenumber
    self.radius = radius
    self.alpha = alpha
    self.C = C
    self.u = u
    self.n = n
    self.kappa = kappa
    self.gamma = gamma
    self.num_field_coeffs = num_field_coeffs
    self.len_arrays = 2*self.num_field_coeffs+1
    # memoization of integration results
    self.tmp_A_integrals = np.zeros(4*self.len_arrays-3, dtype=complex) #initialize 1D np array of complex 
    #A_integrals list is longer than others because it covers range of possible differences in index between 2 or 4 field coeffs 
    self.B_integrals = np.zeros((self.len_arrays, self.len_arrays), dtype=complex) # 2D np array of complex
    self.A_matrix = np.zeros((self.len_arrays, self.len_arrays), dtype=complex) #initialize np complex array nxn
    self.D_matrix = np.zeros((self.len_arrays, self.len_arrays, self.len_arrays, self.len_arrays),dtype=complex) # same 4D
    self.tmp_B_integrals = np.zeros((self.len_arrays, self.len_arrays), dtype=complex) # 2D np array of complex
    self.tmp_A_matrix = np.zeros((self.len_arrays, self.len_arrays), dtype=complex) #initialize np complex array nxn
    self.tmp_D_matrix = np.zeros((self.len_arrays, self.len_arrays, self.len_arrays, self.len_arrays),dtype=complex) # same 4D
    self.tmp_A_integrals_0 = 0

  ######## common terms in integrals ###########

  def sqrt_g_theta(self, amplitude, z):
    return self.radius_rescaled(amplitude) * (1 + amplitude * math.sin(self.wavenumber * z))

  def sqrt_g_z(self, amplitude, z):
    return math.sqrt(1 + (self.radius_rescaled(amplitude) * amplitude * self.wavenumber * math.cos(self.wavenumber * z)) ** 2)

  def radius_rescaled(self, amplitude):
    return self.radius / math.sqrt(1 + amplitude ** 2 / 2.0)

  def n_A_theta_squared(self, amplitude, z):
    return (self.n * self.wavenumber * self.radius_rescaled(amplitude) * amplitude * math.cos(self.wavenumber * z) /self.sqrt_g_z(amplitude, z)) **2 

  ########## evaluates integrals ##########
  def A_integrand_img_part(self, diff, amplitude, z):
    if amplitude == 0:
      return (math.sin(diff * self.wavenumber * z) *  # img part of e^(i ...)
              self.radius)  # sqrt(g_theta theta) = radius
      # radius rescale factor = 1
      # and sqrt(g_zz) =1
    else:
      return (math.sin(diff * self.wavenumber * z) *  # real part of e^(i ...)
              self.sqrt_g_theta(amplitude, z) *
              self.sqrt_g_z(amplitude, z))

  def A_integrand_real_part(self, diff, amplitude, z):
    """
    :param diff: coefficient difference (i-j) or (i+i'-j-j') in e^i(i-j)kz
    :param amplitude: a
    :return: float
    """
    if amplitude == 0:
      return (math.cos(diff * self.wavenumber * z) *  # real part of e^(i ...)
              self.radius)  # sqrt(g_theta theta) = radius
      # and sqrt(g_zz) =0
    else:
      assert (self.sqrt_g_z(amplitude, z) >= 1)
      return (math.cos(diff * self.wavenumber * z) *  # real part of e^(i ...)
              self.sqrt_g_theta(amplitude, z) *
              self.sqrt_g_z(amplitude, z))

  def B_integrand_img_part(self, i, j, amplitude, z):
    if amplitude == 0:
      z_part = (i * j * self.wavenumber ** 2 * math.sin((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.radius)  # sqrt(g_theta theta) = radius
      # and 1/sqrt(g_zz) =1
      return (z_part)
    else:
      z_part = (i * j * self.wavenumber ** 2 * math.sin((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.sqrt_g_theta(amplitude, z) *
                self.sqrt_g_z(amplitude, z))
      theta_part = (self.n_A_theta_squared(amplitude, z) *  # part of n_A_theta^2
                    self.sqrt_g_z(amplitude, z) * 
                    1.0/self.sqrt_g_theta(amplitude, z)*  # sqrt(g_thth) and g^thth
                    math.sin((i-j)*self.wavenumber*z))
      return (z_part+theta_part)

  def B_integrand_real_part(self, i, j, amplitude, z):
    if amplitude == 0:
      z_part = (i * j * self.wavenumber ** 2 * math.cos((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.radius)  # sqrt(g_theta theta) = radius
      return (z_part)
    else:
      z_part = (i * j * self.wavenumber ** 2 * math.cos((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.sqrt_g_theta(amplitude, z) *
                ## don't index raise in z direction
                # 1/self.sqrt_g_z(radius, amplitude, wavenumber, z))  # and 1/sqrt(g_zz) from metric ddeterminant, index raising g^zz
                self.sqrt_g_z(amplitude, z))
      theta_part = (self.n_A_theta_squared(amplitude, z) *  # part of n_A_theta^2
                    self.sqrt_g_z(amplitude, z) *  # sqrt(g_zz) from metric
                    1.0/self.sqrt_g_theta(amplitude, z)*  # sqrt(g_thth) from metric and g^thth = 1/g_thth index raising
                    math.cos((i-j)*self.wavenumber*z))
      return (z_part + theta_part)
  
  def Kthth_integrand(self, amplitude, z):
    return (1/(self.sqrt_g_theta(amplitude,z) *  self.sqrt_g_z(amplitude, z))) #*  # part of K_th^th^2
  #1 / self.sqrt_g_z(amplitude, z) *  # part of K_th^th^2*sqrt_g_zz
  # self.sqrt_g_theta(amplitude, z))  # one radius in sqrt_g_theta and -2 in Kthth
 
  def Kzz_integrand(self, amplitude, z):
    return ((self.radius_rescaled(amplitude)*amplitude*self.wavenumber**2*math.sin(self.wavenumber*z))**2 *
      self.sqrt_g_theta(amplitude, z) /
      (self.sqrt_g_z(amplitude,z))**5)
  

  def evaluate_A_integrals(self, amplitude):
    for diff in range(-4*self.num_field_coeffs,  4*self.num_field_coeffs+1):
      img_part, error = integrate.quad(lambda z: self.A_integrand_img_part(diff, amplitude, z),
                                       0, 2 * math.pi / self.wavenumber)
      real_part, error = integrate.quad(lambda z: self.A_integrand_real_part(diff, amplitude, z),
                                        0, 2 * math.pi / self.wavenumber)
      self.tmp_A_integrals[diff+4*self.num_field_coeffs] = complex(real_part, img_part) # in A_integrals list results are stored by diff between coefficient indices from -n to n
                                                            # places 0 to _ in list correspond to differences -2n+1 to 2n-1 (incl)
    ## fill matrix A[i,j] from list A[diff], where i j k are matrix indices from 0 to 2n (corresponding to ordered list of coefficient's indices from -n to n)
    ## fill matrix D[i,j,k,l] where D[i+j-l-k]=A[diff]
    l = self.len_arrays
    # save to tmp location when first calcualted with new amplitude - gets transferred to permanent matrices accesed by calc_field_energy when amplitude is accepted
    for i in range(l): 
      self.tmp_A_matrix[i] = self.tmp_A_integrals[2*l-i-2:3*l-2-i]
      #print(i,self.A_integrals[-i+0+4*self.num_field_coeffs], "filled", self.A_matrix[i,0])
      for j in range(l):
        for k in range(l):
          self.tmp_D_matrix[i,j,k] =  self.tmp_A_integrals[k-i-j+2*l-2:k-i-j+3*l-2] 
          #print(i,j,k,self.A_integrals[-i-j+k+0+4*self.num_field_coeffs], "filled", self.D_matrix[i,j,k,0])
    self.tmp_A_integrals_0 = self.tmp_A_integrals[0]

  """
  def evaluate_A_integral_0(self, amplitude):
    #useful in no-field simulations
    img_part, error = integrate.quad(lambda z: self.A_integrand_img_part(0, amplitude, z),
                                     0, 2 * math.pi / self.wavenumber)
    assert (math.isclose(img_part, 0, abs_tol=1e-9))
    real_part, error = integrate.quad(lambda z: self.A_integrand_real_part(0, amplitude, z),
                                      0, 2 * math.pi / self.wavenumber)
    self.A_integrals[0] = complex(real_part, img_part) # this is usually done for surface area - no need to fill into A_matrix
    self.tmp_A_integrals_ 0 = complex(real_part, img_part)
  """
  def evaluate_B_integrals(self, amplitude):
    for i in range(-self.num_field_coeffs, self.num_field_coeffs + 1):
      for j in range(-self.num_field_coeffs, self.num_field_coeffs + 1):
        img_part, error = integrate.quad(lambda z: self.B_integrand_img_part(i, j, amplitude, z),
                                         0, 2 * math.pi / self.wavenumber)
        real_part, error = integrate.quad(lambda z: self.B_integrand_real_part(i, j, amplitude, z),
                                          0, 2 * math.pi / self.wavenumber) # integrands demand coefficient's indices in range -n to n
        #print("evaluated with i ", i , "j ", j , "and put it at Bintegrals ", i+self.num_field_coeffs, j+self.num_field_coeffs)
        self.tmp_B_integrals[i+self.num_field_coeffs, j+self.num_field_coeffs] = complex(real_part, img_part) # while results are stored in order in array with indices 0 to 2n

  ############# calc energy ################
  def calc_field_energy(self, field_coeffs):
    """ einsum is (at least 10x) faster than loops! even with constructing matrix A from list
    same for D part even though 4d matrix is constructed from list every time: 10x faster at 3(-1to1) coeffs
    much more for longer set of coeffs
    :param field_coeffs: list of complex values as np array from lowest to highest index, i. e. [c_-n, ..., c_0 , ... c_n] value at index 0 is c_-n
    A, B, D matrices are orderd on the same scheme - because np array is faster than dict explcitly stating indices -n .. 0 .. n for matrix muktiplication
    """
    A_complex_energy = np.einsum("ji, i, j -> ", self.A_matrix, field_coeffs, field_coeffs.conjugate()) # watch out for how A,D are transpose of expected
                                                                              # because its the faster way to construct them
    B_complex_energy = np.einsum("ij, i, j -> ", self.B_integrals, field_coeffs.conjugate(), field_coeffs) # B is filled directly with outcomes of B_integrals, not transposed
    #print(field_coeffs)
    D_complex_energy =  np.einsum("klij, i, j, k, l -> ", self.D_matrix, field_coeffs, field_coeffs, field_coeffs.conjugate(), field_coeffs.conjugate())
    #D_complex_energy2= 0+0j
    #for (i,ci) in enumerate(field_coeffs.conjugate()):
    #  for (j,cj) in enumerate(field_coeffs.conjugate()):
    #    for (k,ck) in enumerate(field_coeffs):
    #      for (l, cl) in enumerate(field_coeffs):
    #        #print(self.A_integrals[i+j-k-l], self.D_matrix[k,l,i,j])
    #        D_complex_energy2 += ci*cj*ck*cl*self.A_integrals[k+l-i-j+4*self.num_field_coeffs]
    #print("remember to comment this out later!")
    #assert (math.isclose(D_complex_energy.real, D_complex_energy2.real))
    assert (math.isclose(A_complex_energy.imag, 0, abs_tol=1e-7))
    assert (math.isclose(B_complex_energy.imag, 0, abs_tol=1e-7))
    #print(field_coeffs, field_coeffs.conjugate())
    assert (math.isclose(D_complex_energy.imag, 0, abs_tol=1e-7)) # this means either D matrix calculation is wrong OR proposed field coefficeitns got very big
    # print("total")
    #print("B part", self.C, B_complex_energy.real)
    #print("A part", self.alpha, A_complex_energy.real)
    #print("D part", self.u, D_complex_energy.real)
    #print(self.A_matrix, self.D_matrix[1,1,:,:])
    # print(self.alpha * A_complex_energy.real + self.C * B_complex_energy.real + 0.5 * self.u * D_complex_energy.real)
    #debug
    """ 
    l=[]
    for start in np.arange(0, 2*math.pi, 0.1):
      img_part, error = integrate.quad(lambda z: self.B_integrand_img_part(1, -1, amplitude, z),
                                         0, 2 * math.pi / self.wavenumber)
      real_part, error = integrate.quad(lambda z: self.B_integrand_real_part(1, -1, amplitude, z),
                                          start, start+1) # integrands demand coefficient's indices in range -n to n
      l.append(complex(real_part, img_part)*field_coeffs[0]*field_coeffs[-1].conjugate())
    self.energies.append(l)
    
    self.coeffs.append(field_coeffs)
    """
    return self.alpha * A_complex_energy.real + self.C * B_complex_energy.real + 0.5 * self.u * D_complex_energy.real

  def calc_field_energy_loop(self, field_coeffs, amplitude, amplitude_change=False):
    """
    :param amplitude_change: True by default as this function is often called in simultaneous update of field and amplitude
    old version of cacl_field_energy with loop, dict lookups
    """
    # some memoization: use previously calculated A integrals (=D integrals) and B integrals
    # reevaluate only on  amplitude change
    # TODO : initialize on System object creation, instead of checking if empty every time
    if amplitude_change:
      print("if you mean to reevaluate integrals on amplitude change, call evaluate_integrals")
      num_field_coeffs = max(field_coeffs)
      self.evaluate_A_integrals(amplitude, num_field_coeffs)
      self.evaluate_B_integrals(amplitude, num_field_coeffs)

    A_complex_energy = 0 + 0j
    B_complex_energy = 0 + 0j
    D_complex_energy = 0 + 0j  # identity of complex sum
    for i1 in field_coeffs:
      for j1 in field_coeffs:
        A_complex_energy += field_coeffs[i1] * field_coeffs[j1].conjugate() * self.A_integrals[i1 - j1]
        B_complex_energy += field_coeffs[i1] * field_coeffs[j1].conjugate() * self.B_integrals[(i1, j1)]
        for i2 in field_coeffs:
          for j2 in field_coeffs:
            D_complex_energy += (field_coeffs[i1] * field_coeffs[i2] *
                                 field_coeffs[j1].conjugate() * field_coeffs[j2].conjugate() *
                                 self.A_integrals[(i1 + i2 - j1 - j2)])
    assert (math.isclose(A_complex_energy.imag, 0, abs_tol=1e-7))
    assert (math.isclose(B_complex_energy.imag, 0, abs_tol=1e-7))
    #print(field_coeffs)
    #print(D_complex_energy)
    assert (math.isclose(D_complex_energy.imag, 0, abs_tol=1e-7))
    # print("total")
    #print(self.C, B_complex_energy.real)
    # print(self.alpha, A_complex_energy.real)
    # print(self.alpha * A_complex_energy.real + self.C * B_complex_energy.real + 0.5 * self.u * D_complex_energy.real)
    return self.alpha * A_complex_energy.real + self.C * B_complex_energy.real + 0.5 * self.u * D_complex_energy.real


  def calc_field_energy_diff_ABpart(self, index, new_field_coeff, old_field_coeffs):
    old_field_coeff = old_field_coeffs[index]
    diff = new_field_coeff - old_field_coeff
    A_complex_energy_diff = 0 + 0j
    B_complex_energy_diff = 0 + 0j
    for i in old_field_coeffs:
      A_complex_energy_diff += diff * old_field_coeffs[i].conjugate() * self.A_integrals[index - i]
      A_complex_energy_diff += old_field_coeffs[i] * diff.conjugate() * self.A_integrals[i - index]
      B_complex_energy_diff += diff * old_field_coeffs[i].conjugate() * self.B_integrals[index, i]
      B_complex_energy_diff += old_field_coeffs[i] * diff.conjugate() * self.B_integrals[i, index]
    # undoing the change we did (with partially old value) where i=index above ,
    # + changing the point (index, i=index) correctly works out to this addition
    A_complex_energy_diff += diff * diff.conjugate() * self.A_integrals[0]
    B_complex_energy_diff += diff * diff.conjugate() * self.B_integrals[index, index]
    assert (math.isclose(B_complex_energy_diff.imag, 0, abs_tol=1e-7))
    #print("diff A B parts")
    #print("B integral 0", self.B_integrals[(0,0)])
    #print("A:", A_complex_energy_diff.real, "B:", B_complex_energy_diff.real)
    return A_complex_energy_diff.real, B_complex_energy_diff.real
  
  def calc_field_energy_diff_Dpart(self, index, new_field_coeff, old_field_coeffs):
    old_field_coeff = old_field_coeffs[index]
    diff = new_field_coeff - old_field_coeff
    D_complex_energy_diff = 0 + 0j
    for i in old_field_coeffs:
      if i != index:
        for j in old_field_coeffs:
          if j!= index:
            for k in old_field_coeffs:
              if k!= index:
                #replacing parts of the sum where c_index or c_index^* occured once
                D_complex_energy_diff += 2*diff * old_field_coeffs[i] * old_field_coeffs[j].conjugate() * old_field_coeffs[k].conjugate() * self.A_integrals[index + i - j-k ]
                D_complex_energy_diff += 2*old_field_coeffs[i] *old_field_coeffs[j] * diff.conjugate() * old_field_coeffs[k].conjugate() * self.A_integrals[i+j-index-k]
                #print("changed", "c_index", " * coeff[", i, j, k,"]")
                #print("diff", D_complex_energy_diff)
            #twice
            D_complex_energy_diff -= old_field_coeff**2 * old_field_coeffs[j].conjugate() * old_field_coeffs[i].conjugate() *self.A_integrals[index+index-i-j]
            D_complex_energy_diff += new_field_coeff**2 * old_field_coeffs[j].conjugate() * old_field_coeffs[i].conjugate() *self.A_integrals[index+index-i-j]
            D_complex_energy_diff -= old_field_coeff.conjugate()**2 * old_field_coeffs[j] * old_field_coeffs[i] *self.A_integrals[i+j-index-index]
            D_complex_energy_diff += new_field_coeff.conjugate()**2 * old_field_coeffs[j] * old_field_coeffs[i] *self.A_integrals[i+j-index-index]
            D_complex_energy_diff -= 4*old_field_coeff*old_field_coeff.conjugate() * old_field_coeffs[i]* old_field_coeffs[j].conjugate() *self.A_integrals[index+i-index-j]
            D_complex_energy_diff += 4*new_field_coeff*new_field_coeff.conjugate() * old_field_coeffs[i]* old_field_coeffs[j].conjugate() *self.A_integrals[index+i-index-j]
        #three times
        D_complex_energy_diff -=  2*old_field_coeff**2 * old_field_coeff.conjugate() * old_field_coeffs[i].conjugate()*self.A_integrals[index-i]
        D_complex_energy_diff +=  2*new_field_coeff**2 * new_field_coeff.conjugate() * old_field_coeffs[i].conjugate()*self.A_integrals[index-i]
        D_complex_energy_diff -= 2*old_field_coeff *old_field_coeffs[i]* old_field_coeff.conjugate()**2 * self.A_integrals[i-index]
        D_complex_energy_diff += 2*new_field_coeff *old_field_coeffs[i]* new_field_coeff.conjugate()**2 * self.A_integrals[i-index]
    #replacing the point where c_index occured 4 times
    D_complex_energy_diff -= old_field_coeff**2 * old_field_coeff.conjugate()**2*self.A_integrals[0]
    D_complex_energy_diff += new_field_coeff**2 * new_field_coeff.conjugate()**2*self.A_integrals[0]
    #try:
    assert (math.isclose(D_complex_energy_diff.imag, 0, abs_tol=1e-7))
    #except AssertionError:
      #print("failed assertion imag part of D_diff = 0.  D_complex_energy_diff: ", D_complex_energy_diff)
    #print("diff D parts")
    #print("D:", D_complex_energy_diff.real)
    return D_complex_energy_diff.real

  def calc_field_energy_diff(self, index, new_field_coeff, old_field_coeffs, amplitude, amplitude_change=False):
    if amplitude_change:
      
      print("if you mean to reevaluate integrals on amplitude change, call evaluate_integrals")
      num_field_coeffs = max(old_field_coeffs)
      self.evaluate_A_integrals(amplitude, num_field_coeffs)
      self.evaluate_B_integrals(amplitude, num_field_coeffs)
    A_energy_diff, B_energy_diff = self.calc_field_energy_diff_ABpart(index, new_field_coeff, old_field_coeffs)
    D_energy_diff = self.calc_field_energy_diff_Dpart(index, new_field_coeff, old_field_coeffs)
    return self.alpha * A_energy_diff + self.C * B_energy_diff + 0.5 * self.u * D_energy_diff

  def calc_bending_energy(self, amplitude):
    """
    calculate bending as (K_i^i)**2.  Gaussian curvature and cross term 2 K_th^th K_z^z are omitted due to gauss-bonnet theorem.
    """
    if amplitude == 0:
      Kthth_integral, error = integrate.quad(lambda z: 1.0 / self.radius ** 2, 0, 2 * math.pi / self.wavenumber)
      return Kthth_integral
    else:
      Kzz_integral, error = integrate.quad(lambda z: self.Kzz_integrand(amplitude, z), 0, 2 * math.pi / self.wavenumber)
      Kthth_integral, error = integrate.quad(lambda z: self.Kthth_integrand(amplitude, z),  0, 2 * math.pi / self.wavenumber)
      return (Kzz_integral + Kthth_integral)

  def save_temporary_matrices(self):
    #if accepted, save results of evaluation with proposed amplitude to more permanent location
    self.A_matrix = self.tmp_A_matrix
    self.D_matrix = self.tmp_D_matrix
    self.B_integrals = self.tmp_B_integrals

  def calc_surface_energy(self, amplitude, field_coeffs):
    """
    energy from surface tension * surface area, + bending rigidity constant * mean curvature squared
    """
    self.evaluate_A_integrals(amplitude)
    self.evaluate_B_integrals(amplitude)
    surface_energy =  self.gamma * self.tmp_A_integrals_0.real + self.kappa * self.calc_bending_energy(amplitude)
    #calculate effect on current field with temporary(!) matrix elements from proposed(!) amplitude
    A_complex_energy = np.einsum("ji, i, j -> ", self.tmp_A_matrix, field_coeffs, field_coeffs.conjugate()) # watch out for how A,D are transpose of expected
    B_complex_energy = np.einsum("ij, i, j -> ", self.tmp_B_integrals, field_coeffs.conjugate(), field_coeffs) 
    D_complex_energy =  np.einsum("klij, i, j, k, l -> ", self.tmp_D_matrix, field_coeffs, field_coeffs, field_coeffs.conjugate(), field_coeffs.conjugate())
    assert (math.isclose(A_complex_energy.imag, 0, abs_tol=1e-7))
    assert (math.isclose(B_complex_energy.imag, 0, abs_tol=1e-7))
    assert (math.isclose(D_complex_energy.imag, 0, abs_tol=1e-7)) 
    field_energy =  self.alpha * A_complex_energy.real + self.C * B_complex_energy.real + 0.5 * self.u * D_complex_energy.real
    return surface_energy + field_energy

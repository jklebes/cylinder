import math
import scipy.integrate as integrate

class System():

  def __init__(self, wavenumber, radius, alpha, C, u, n, kappa, gamma):
    assert(all(map(lambda x: x>=0, [wavenumber, radius, C, u, kappa, gamma] )))
    assert(alpha <=0)
    self.wavenumber = wavenumber
    self.radius = radius
    self.alpha = alpha
    self.C = C
    self.u = u
    self.n = n
    self.kappa= kappa
    self.gamma = gamma
    #memoization of integration results:
    #empty dicts before first calculation
    self.A_integrals = dict()
    self.B_integrals = dict()

  ######## common terms in integrals ###########

  def sqrt_g_theta(self, amplitude, z):
    return self.radius_rescaled(amplitude) * (1 + amplitude * math.sin(self.wavenumber * z))

  def sqrt_g_z(self,  amplitude, z):
    return math.sqrt(1 + (self.radius_rescaled(amplitude) * amplitude
                          * self.wavenumber * math.cos(self.wavenumber * z)) ** 2)

  def radius_rescaled(self, amplitude):
    return self.radius / math.sqrt(1 + amplitude ** 2 / 2.0)

  def n_A_theta_squared(self, amplitude, z):
    return (self.n * self.wavenumber * self.radius_rescaled(amplitude) * amplitude * math.cos(
      self.wavenumber * z)) ** 2  # without: /self.sqrt_g_z(radius,amplitude, wavenumber, z)

  ########## evaluates integrals ##########
  def A_integrand_img_part(self, diff, amplitude, z):
    if amplitude == 0:
      return (math.sin(diff * self.wavenumber * z) *  # img part of e^(i ...)
              self.radius)                            # sqrt(g_theta theta) = radius
                                                      # radius rescale factor = 1
                                                      # and sqrt(g_zz) =1
    else:
      return (math.sin(diff * self.wavenumber * z) *                     # real part of e^(i ...)
              self.sqrt_g_theta(amplitude, z) *  
              self.sqrt_g_z(amplitude, z))

  def A_integrand_real_part(self, diff, amplitude, z):
    """
    :param diff: coefficient difference (i-j) or (i+i'-j-j') in e^i(i-j)kz
    :param amplitude: a
    :return: float
    """
    if amplitude == 0:
      return (math.cos(diff * self.wavenumber * z) * # real part of e^(i ...)
              self.radius)                           # sqrt(g_theta theta) = radius
                                                     # and sqrt(g_zz) =0
    else:
      assert(self.sqrt_g_z(amplitude, z) >= 1)
      return (math.cos(diff * self.wavenumber * z) *  # real part of e^(i ...)
              self.sqrt_g_theta(amplitude, z) * 
              self.sqrt_g_z(amplitude, z))

  def B_integrand_img_part(self, i, j, amplitude, z):
    if amplitude == 0:
      z_part = (i * j * self.wavenumber ** 2 * math.sin((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.radius)                                                         # sqrt(g_theta theta) = radius
                                                                                     # and 1/sqrt(g_zz) =1
      return (z_part)
    else:
      z_part = (i * j * self.wavenumber ** 2 * math.sin((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.sqrt_g_theta( amplitude,z) * 
                self.sqrt_g_z(amplitude,z))
      return (z_part)

  def B_integrand_real_part(self, i, j, amplitude, z):
    if amplitude == 0:
      z_part = (i * j * self.wavenumber ** 2 * math.cos((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.radius)                                                               # sqrt(g_theta theta) = radius
                                                                                      # and 1/sqrt(g_zz) =1
      return (z_part)
    else:
      z_part = (i * j * self.wavenumber ** 2 * math.cos((i - j) * self.wavenumber * z) *  # |d e^... |^2
                self.sqrt_g_theta(amplitude, z) * 
                ## don't index raise in z direction
                #1/self.sqrt_g_z(radius, amplitude, wavenumber, z))  # and 1/sqrt(g_zz) from metric ddeterminant, index raising g^zz
                self.sqrt_g_z( amplitude, z))
      theta_part = (self.n_A_theta_squared( amplitude, z) *  # part of n_A_theta^2
                    1.0 / self.sqrt_g_z( amplitude, z) *  # sqrt(g_zz) annd with 1/g_zz normalizing A_theta^2
                    ## index raise in theta direction?
                    1.0 / self.sqrt_g_theta( amplitude, z))  # sqrt(g_thth) and g^thth
                    #self.sqrt_g_theta(radius, amplitude, wavenumber, z))
      return (z_part + theta_part)

  def Kzz_integrand(self,amplitude, z):
    # TODO: redo with functions
    return ((amplitude * self.wavenumber ** 2 * math.sin(self.wavenumber * z)) ** 2 *
            self.radius_rescaled(amplitude) ** 3 * (1 + amplitude * math.sin(self.wavenumber * z)) *
            (1 + amplitude ** 2 / 2) ** (-3 / 2.0) *
            1 / (1 + (amplitude * self.wavenumber * math.cos(self.wavenumber * z)) ** 2))

  def Kthth_integrand(self, amplitude, z):
    return (1 / ((self.radius) ** 2) *  # part of K_th^th^2
            1 / self.sqrt_g_z(amplitude, z) *  # part of K_th^th^2*sqrt_g_zz
            self.sqrt_g_theta(amplitude, z))  # one radius in sqrt_g_theta and -2 in Kthth

  def evaluate_A_integrals(self, amplitude, field_coeffs):
    num_field_coeffs = max(field_coeffs)
    for diff in range(-4 * num_field_coeffs, 4 * num_field_coeffs + 1):
      img_part, error = integrate.quad(lambda z: self.A_integrand_img_part(diff, amplitude, z),
                                       0, 2 * math.pi / self.wavenumber)
      real_part, error = integrate.quad(lambda z: self.A_integrand_real_part(diff, amplitude, z),
                                        0, 2 * math.pi / self.wavenumber)
      self.A_integrals[diff] = complex(real_part, img_part)

  def evaluate_A_integral_0(self, amplitude):
    img_part, error = integrate.quad(lambda z: self.A_integrand_img_part(0, amplitude, z),
                                     0, 2 * math.pi / self.wavenumber)
    assert( math.isclose(img_part, 0, abs_tol=1e-9))
    real_part, error = integrate.quad(lambda z: self.A_integrand_real_part(0, amplitude, z),
                                      0, 2 * math.pi / self.wavenumber)
    self.A_integrals[0] = complex(real_part, img_part)

  def evaluate_B_integrals(self, amplitude, field_coeffs): 
    for i in field_coeffs:
      for j in field_coeffs:
        img_part, error = integrate.quad(lambda z: self.B_integrand_img_part(i, j, amplitude, z),
                                          0, 2 * math.pi / self.wavenumber)
        real_part, error = integrate.quad(lambda z: self.B_integrand_real_part(i, j, amplitude, z) ,
          0, 2 * math.pi / self.wavenumber) 
        self.B_integrals[(i, j)] = complex(real_part, img_part)

  ############# calc energy ################

  def calc_field_energy(self, field_coeffs, amplitude, amplitude_change=True):
    # some memoization: use previously calculated A integrals (=D integrals) and B integrals
    # reevaluate only on  amplitude change
    # TODO : initialize on System object creation, instead of checking if empty every time
    if amplitude_change or (not self.A_integrals) :
      self.evaluate_A_integrals(amplitude, field_coeffs=field_coeffs)
    if amplitude_change or (not self.B_integrals) :
      self.evaluate_B_integrals(amplitude, field_coeffs=field_coeffs)

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
    assert (math.isclose(D_complex_energy.imag, 0, abs_tol=1e-7))
    return self.alpha * A_complex_energy.real + self.C * B_complex_energy.real + 0.5 * self.u * D_complex_energy.real

  def calc_field_energy_diff(self, index, new_field_coeff, old_field_coeffs, amplitude, amplitude_change=False):
    if amplitude_change or (not self.A_integrals) :
      self.evaluate_A_integrals(amplitude, field_coeffs=old_field_coeffs)
    if amplitude_change or (not self.B_integrals) :
      self.evaluate_B_integrals(amplitude, field_coeffs=old_field_coeffs)
    old_field_coeff=old_field_coeffs[index]
    diff = new_field_coeff - old_field_coeff
    A_complex_energy = 0 + 0j
    B_complex_energy = 0 + 0j
    for i in old_field_coeffs:
      A_complex_energy += diff * old_field_coeffs[i].conjugate() * self.A_integrals[index - i]
      A_complex_energy += old_field_coeffs[i] * diff.conjugate() * self.A_integrals[i - index]
      B_complex_energy += diff * old_field_coeffs[i].conjugate() * self.B_integrals[ index, i]
      B_complex_energy += old_field_coeffs[i] * diff.conjugate() * self.B_integrals[i, index]
    # undoing the change we did (with partially old value) where i=index above ,
    # + changing the point (index, i=index) correctly works out to this addition
    A_complex_energy += diff*diff.conjugate()*self.A_integrals[0]
    B_complex_energy += diff*diff.conjugate()*self.B_integrals[index, index]
    assert (math.isclose(B_complex_energy.imag, 0, abs_tol=1e-7))

    # todo: test this logic
    D_complex_energy = 0 + 0j  # identity of complex sum
    for i in old_field_coeffs:
      for j in old_field_coeffs:
        for k in old_field_coeffs:
          D_complex_energy+=2* self.A_integrals[index+i-j-k]*diff * old_field_coeffs[i] * old_field_coeffs[j].conjugate() * old_field_coeffs[k].conjugate()
          #3-variable sums with i,j, k in varyng roles
          D_complex_energy+= 2* self.A_integrals[i+j-index-k]*old_field_coeffs[i]*old_field_coeffs[j]*diff.conjugate()* old_field_coeffs[k].conjugate()
        D_complex_energy += 4* self.A_integrals[i+index-j-index] *old_field_coeffs[i]*old_field_coeffs[j].conjugate()* diff* diff.conjugate()
        #2-variable correction to previous and addition, condensed
        D_complex_energy+= self.A_integrals[index+index-i-j]*old_field_coeffs[i].conjugate()*old_field_coeffs[j].conjugate()*diff*diff
        D_complex_energy+= self.A_integrals[i+j-index-index]*old_field_coeffs[i]*old_field_coeffs[j]*diff.conjugate()*diff.conjugate()
      #1-variable sum parts
      D_complex_energy += self.A_integrals[index+index-index-i]*old_field_coeffs[i].conjugate()*(new_field_coeff*new_field_coeff.conjugate()*(2*new_field_coeff-4*old_field_coeff)+old_field_coeff.conjugate()*(4*old_field_coeff*old_field_coeff-2*new_field_coeff*new_field_coeff))
      D_complex_energy+= self.A_integrals[index+i-index-index]*old_field_coeffs[i]*(new_field_coeff*new_field_coeff.conjugate()*(3*new_field_coeff.conjugate()-4*old_field_coeff.conjugate())+new_field_coeff*(4*old_field_coeff.conjugate()*old_field_coeff.conjugate()-2*new_field_coeff.conjugate()*new_field_coeff.conjugate()))
    #the point (index,index,index,index) 
    #corrected from previous + the all-updated point
    D_complex_energy+=self.A_integrals[0]*(3*old_field_coeff**2*old_field_coeff.conjugate()**2 + new_field_coeff*new_field_coeff.conjugate()*(-2*new_field_coeff.conjugate()*old_field_coeff -2* new_field_coeff*old_field_coeff.conjugate()+ new_field_coeff*new_field_coeff.conjugate()))
    try:
      
      assert (math.isclose(D_complex_energy.imag, 0, abs_tol=1e-7))
    except AssertionError:
      print(D_complex_energy.imag)
      raise(AssertionError)
    return self.alpha * A_complex_energy.real + self.C * B_complex_energy.real + 0.5 * self.u * D_complex_energy.real

  def calc_bending_energy(self, amplitude):
    """
    calculate bending as (K_i^i)**2.  Gaussian curvature and cross term 2 K_th^th K_z^z are omitted due to gauss-bonnet theorem.
    """
    if amplitude == 0:
      Kthth_integral, error = integrate.quad(lambda z: 1.0 / self.radius ** 2,
                                             0, 2 * math.pi / self.wavenumber)
      return Kthth_integral
    else:
      Kzz_integral, error = integrate.quad(lambda z: self.Kzz_integrand(amplitude, z),
                                           0, 2 * math.pi / self.wavenumber)
      Kthth_integral, error = integrate.quad(lambda z: self.Kthth_integrand(amplitude, z),
        0, 2 * math.pi / self.wavenumber)
      return (Kzz_integral + Kthth_integral)

  def calc_surface_energy(self, amplitude, amplitude_change=True):
    """
    energy from surface tension * surface area, + bending rigidity constant * mean curvature squared
    """
    if (not self.A_integrals) or amplitude_change:
      self.evaluate_A_integral_0(amplitude)
    #A_integrals[0] is just surface area
    return self.gamma * self.A_integrals[0].real + self.kappa * self.calc_bending_energy(amplitude)

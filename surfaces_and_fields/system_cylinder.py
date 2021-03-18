import math
import scipy.integrate as integrate
import numpy as np

class Cylinder():
  """
  System: Cylider
  This class contains basic cylinder with mechanical properties: surface tension and bending rigidity only
  Extended by subclasses which add field in various schemes
  A physics system containing functions that return an energy for given real , complex parameter values
  Representing a section of sinusoidally perturbed cylindrical surface / surface of revolution r(z) = r(a) (1+a sin(kz))
  plus a complex-valued field over the two-dimensional surface, fourier decomposed into an array of modes indexed (j, beta)
  """
  def __init__(self, wavenumber, radius, kappa, gamma=1, intrinsic_curvature=0):
    assert (all(map(lambda x: x >= 0, [wavenumber, radius, kappa, gamma])))
    self.wavenumber = wavenumber
    self.radius = radius
    self.kappa = kappa
    self.intrinsic_curvature = intrinsic_curvature
    #effective surface tension including H_0^2 constant
    self.effective_gamma = gamma 

  ######## common terms in integrals ###########
  def g_theta(self, amplitude, z):
    return self.radius_rescaled(amplitude)**2 * (1 + amplitude * math.sin(self.wavenumber * z))**2

  def sqrt_g_theta(self, amplitude, z):
    return self.radius_rescaled(amplitude) * (1 + amplitude * math.sin(self.wavenumber * z))

  def g_z(self, amplitude, z):
    return 1 + (self.radius_rescaled(amplitude) * amplitude * self.wavenumber * math.cos(self.wavenumber * z)) ** 2

  def sqrt_g_z(self, amplitude, z):
    return math.sqrt(1 + (self.radius_rescaled(amplitude) * amplitude * self.wavenumber * math.cos(self.wavenumber * z)) ** 2)

  def radius_rescaled(self, amplitude):
    return self.radius / math.sqrt(1 + amplitude ** 2 / 2.0)

  def A_theta(self, amplitude, z):
    return  self.wavenumber * self.radius_rescaled(amplitude) * amplitude * math.cos(self.wavenumber * z) /self.sqrt_g_z(amplitude, z)


  ########## evaluates integrals ##########

  def Kthth_integrand(self, amplitude, z):
    """
    Kthth^2 * sqrt(g) = sqrt(g) /sqrt(g)^2 = 1/sqrt(g)
    """
    return (1/(self.sqrt_g_theta(amplitude,z) *  self.sqrt_g_z(amplitude, z)))
 
  def Kzz_integrand(self, amplitude, z):
    return ((self.radius_rescaled(amplitude)*amplitude*self.wavenumber**2*math.sin(self.wavenumber*z))**2 *
      self.sqrt_g_theta(amplitude, z) /
      (self.sqrt_g_z(amplitude,z))**5)


  def Kthth_linear_integrand(self, amplitude, z):
    """
    integrand related to cross term -4KththH_0 
    Kzz sqrt(g) =-1
    """
    return -1

  def Kzz_linear_integrand(self, amplitude, z):
    """
    integrand related to cross term -4KzzH_0 
    Kzz sqrt(g) = R'' sqrt(gzz) / gthth
    """
    return ( self.radius_rescaled(amplitude)*amplitude * self.wavenumber**2 * math.sin(self.wavenumber*z) *#minus omitted, cancelled wth (-) from -2KzzH0 later
             self.sqrt_g_theta(amplitude, z) / self.g_z(amplitude, z))

  def A_integrand_real_part(self, diff, amplitude, z):
    """
    In this base class, only diff=0 is evre used, so cos(diff*...) part =1
    :param diff: coefficient difference (i-j) or (i+i'-j-j') in e^i(i-j)kz
    :param amplitude: a
    :return: float
    """
    if amplitude == 0:
      return (self.radius) 
    else:
      # real part of e^(i ...) cos(diff=0*...) =1
      return ( self.sqrt_g_theta(amplitude, z) *
              self.sqrt_g_z(amplitude, z))

  def A_th_squared_integrand(self, amplitude, z):
    return self.A_theta(amplitude, z)**2 * self.sqrt_g_z(amplitude,z)/self.sqrt_g_theta(amplitude,z) #sqrt{g}*index raising 1/g_thth


  def evaluate_A_integral_0(self, amplitude):
    #useful in no-field simulations
    #not doing img part, whcih should be 0, every times
    #img_part, error = integrate.quad(lambda z: self.A_integrand_img_part(0, amplitude, z),
    #                                 0, 2 * math.pi / self.wavenumber)
    #assert (math.isclose(img_part, 0, abs_tol=1e-9))
    real_part, error = integrate.quad(lambda z: self.A_integrand_real_part(0, amplitude, z),
                                      0, 2 * math.pi / self.wavenumber)
    # this is usually done for surface area - no need to fill into A_matrix
    return real_part
  

  ############# calc energy ################

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
      #for interaction with intrinsic mean curvature H0 
      Kzz_linear_integral, error = integrate.quad(lambda z: self.Kzz_linear_integrand(amplitude, z),  0, 2 * math.pi / self.wavenumber)
      Kthth_linear_integral, error = integrate.quad(lambda z: self.Kthth_linear_integrand(amplitude, z),  0, 2 * math.pi / self.wavenumber)
      return (Kzz_integral + Kthth_integral - 4*self.intrinsic_curvature*(Kzz_linear_integral+Kthth_linear_integral))

  def calc_surface_energy(self, amplitude):
    """
    energy from surface tension * surface area, + bending rigidity constant * mean curvature squared
    """
    return  (self.effective_gamma * self.evaluate_A_integral_0(amplitude) + 
             .5*self.kappa * self.calc_bending_energy(amplitude))

  def calc_field_bending_energy(self, amplitude, magnitudesquared, Cnsquared):
    """
    utility that lets us compare what the energy landscape would look like with
    bending energy' from coupling an
    ideally ordered field |Psi|^2=-alpha/u univformaly everywhere
    """
    A_th_squared_integral, error = integrate.quad(lambda z: self.A_th_squared_integrand(amplitude, z), 0, 2 * math.pi / self.wavenumber)
    return Cnsquared*magnitudesquared*A_th_squared_integral
                

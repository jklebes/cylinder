import math
import scipy.integrate as integrate


######## common terms in integrals ###########

def sqrt_g_theta(radius, amplitude, wavenumber, z):
  return radius * (1+ amplitude*math.sin(wavenumber*z))

def sqrt_g_z(radius, amplitude, wavenumber, z):
  return math.sqrt(1+ (radius * amplitude * wavenumber * math.cos(wavenumber*z))**2)

def radius_rescale_factor(amplitude):
  return (1+amplitude**2/2)**(-0.5)

########## evaluates integrals ##########

A_integrals = dict()
B_integrals = dict()

def A_integrand_img_part(diff, amplitude, z,wavenumber):
  """
  :param diff: coefficient difference (i-j) or (i+i'-j-j') in e^i(i-j)kz
  :param amplitude: a
  :return: float
  """
  if amplitude==0:
    return (math.sin(diff * wavenumber* z) *#img part of e^(i ...)
                          radius ) #sqrt(g_theta theta) = radius
                          # radius rescale factor = 1
                          # and sqrt(g_zz) =1
  else:
    return (math.sin(diff * wavenumber * z) *  # real part of e^(i ...)
            sqrt_g_theta(radius, amplitude, wavenumber, z) *  # sqrt(g_theta theta)
            radius_rescale_factor(amplitude)*  # r(a) adjustment for volume conservation
            sqrt_g_z(radius, amplitude, wavenumber, z)) # and sqrt(g_zz)

def A_integrand_real_part(diff, amplitude, z, wavenumber):
  """
  :param diff: coefficient difference (i-j) or (i+i'-j-j') in e^i(i-j)kz
  :param amplitude: a
  :return: float
  """
  if amplitude==0:
    return (math.cos( diff * wavenumber* z) *#real part of e^(i ...)
                          radius ) #sqrt(g_theta theta) = radius
                          # r(a) due to volume conservation = r
                          # and sqrt(g_zz) =0
  else:
    return (math.cos(  diff * wavenumber* z) *#real part of e^(i ...)
            sqrt_g_theta(radius, amplitude, wavenumber, z) * #sqrt(g_theta theta)
            radius_rescale_factor(amplitude)* # r(a) adjustment for volume conservation
            sqrt_g_z(radius, amplitude, wavenumber, z))# and sqrt(g_zz)


def B_integrand_img_part(i,j, amplitude, z, wavenumber):
  if amplitude == 0:
    z_part = (i * j * wavenumber ** 2 * math.sin((i - j) * wavenumber * z) *  # |d e^... |^2
              radius)  # sqrt(g_theta theta) = radius
    # r(a) rescale due to volume conservation = *1
    # and 1/sqrt(g_zz) =1
    return (z_part)
  else:
    z_part = (i * j * wavenumber ** 2 * math.sin((i - j) * wavenumber * z) *  # |d e^... |^2
              sqrt_g_theta(radius, amplitude, wavenumber, z) *  # sqrt(g_theta theta)
              radius_rescale_factor(amplitude) *  # r(a) due to volume conservation
              sqrt_g_z(radius, amplitude, wavenumber, z))  # and 1/sqrt(g_zz)
    return (z_part)

def B_integrand_real_part(i, j , amplitude, z, wavenumber):
  if amplitude == 0:
    z_part = (i * j * wavenumber ** 2 * math.cos((i - j) * wavenumber * z) *  # |d e^... |^2
              radius)  # sqrt(g_theta theta) = radius
              # r(a) due to volume conservation
              # and 1/sqrt(g_zz) =1
    return(z_part)
  else:
    z_part = (i * j * wavenumber ** 2 * math.cos((i - j) * wavenumber * z) *  # |d e^... |^2
              sqrt_g_theta(radius, amplitude, wavenumber, z) *  # sqrt(g_theta theta)
              radius_rescale_factor(amplitude) *  # r(a) due to volume conservation
              1/sqrt_g_z(radius, amplitude, wavenumber, z))  # and 1/sqrt(g_zz) from metric ddeterminant, index raising g^zz
    theta_part = ((n*wavenumber*radius*amplitude*math.cos(wavenumber*z))**2 * #part of A_theta^2
                  1.0/sqrt_g_z(radius, amplitude, wavenumber, z) * #sqrt(g_zz) annd with 1/g_zz normalizing A_theta^2
                  1.0/sqrt_g_theta(radius, amplitude, wavenumber, z)* #sqrt(g_thth) and g^thth
                  radius_rescale_factor(amplitude)) # two from r^2 in A_th, -2 from g^thth, 1 from sqrt_g_thth
    return (z_part + theta_part)

def Kzz_integrand(amplitude, z, wavenumber):
  return( (amplitude * wavenumber**2 * math.sin(wavenumber*z))**2 *
          radius**3 * (1+amplitude * math.sin(wavenumber*z)) *
          (1+amplitude**2/2)**(-3/2.0)*
          1/(1+(amplitude*wavenumber*math.cos(wavenumber*z))**2))

def Kthth_integrand(amplitude, z, wavenumber):
  return( 1/((radius )**2)* #part of K_th^th^2
          1/sqrt_g_z(radius, amplitude, wavenumber, z)*# part of K_th^th^2*sqrt_g_zz
          sqrt_g_theta(radius, amplitude, wavenumber, z)*
          1/radius_rescale_factor(amplitude)) # one radius in sqrt_g_theta and -2 in Kthth

def evaluate_A_integrals(amplitude, wavenumber, num_field_coeffs):
  global A_integrals
  for diff in range(-4*num_field_coeffs, 4*num_field_coeffs+1):
    img_part, error = integrate.quad(lambda z: A_integrand_img_part(diff, amplitude, z, wavenumber=wavenumber),
                              0, 2 *math.pi / wavenumber)
    real_part, error = integrate.quad(lambda z: A_integrand_real_part(diff, amplitude, z, wavenumber=wavenumber),
                               0, 2 *math.pi / wavenumber)
    A_integrals[diff]= complex(real_part, img_part)

def evaluate_A_integral_0(amplitude, wavenumber, num_field_coeffs):
  global A_integrals
  img_part, error = integrate.quad(lambda z: A_integrand_img_part(0, amplitude, z, wavenumber=wavenumber),
                              0, 2 *math.pi / wavenumber)
  if not math.isclose(img_part, 0, abs_tol=1e-9):
    print("surface area = A[diff=0] with imaginary component", img_part)
  real_part, error = integrate.quad(lambda z: A_integrand_real_part(0, amplitude, z, wavenumber=wavenumber),
                               0, 2 *math.pi / wavenumber)
  A_integrals[0]= complex(real_part, img_part)

def evaluate_B_integrals(amplitude, wavenumber):
  global B_integrals
  B_integrals = dict()
  for i in range(-1*num_field_coeffs, num_field_coeffs+1):
    for j in range(-1 * num_field_coeffs, num_field_coeffs + 1):
      img_part, error = integrate.quad(lambda z: B_integrand_img_part(i, j , amplitude, z, wavenumber=wavenumber),
                                       0, 2 * math.pi / wavenumber)
      real_part, error = integrate.quad(lambda z: B_integrand_real_part(i, j , amplitude, z, wavenumber=wavenumber),
                                        0, 2 * math.pi / wavenumber) #throws integrationwarning for amplitude >.9
      B_integrals[(i,j)] = complex(real_part, img_part)

############# calc energy ################

def calc_field_energy(field_coeffs, amplitude, wavenumber, amplitude_change=True):
  #some memoization: repurpose A integrals as D integrals
  # and don't reevaluate until amplitude change
  num_field_coeffs = len(field_coeffs)
  if not A_integrals or amplitude_change:
    #print("amplitude", amplitude)
    evaluate_A_integrals(amplitude, wavenumber=wavenumber)
    evaluate_B_integrals(amplitude, wavenumber=wavenumber)
    #print("A integrals", A_integrals)
    #print("B integrals", B_integrals)
  A_complex_energy = 0+0j
  for i in range(-1 * num_field_coeffs, num_field_coeffs + 1):
    for j in range(-1 * num_field_coeffs, num_field_coeffs + 1):
      A_complex_energy+= (field_coeffs[i]*field_coeffs[j].conjugate()*A_integrals[i-j])
  if not math.isclose(A_complex_energy.imag, 0, abs_tol=1e-9):
    print("nonzero imaginary energy component in A", A_complex_energy)
  B_complex_energy = 0 + 0j
  for i in range(-1 * num_field_coeffs, num_field_coeffs + 1):
      for j in range(-1 * num_field_coeffs, num_field_coeffs + 1):
        B_complex_energy += (field_coeffs[i] * field_coeffs[j].conjugate() * B_integrals[(i,j)])
  if not math.isclose(B_complex_energy.imag, 0, abs_tol=1e-9):
    print("nonzero imaginary energy component in B", B_complex_energy)
  #same reduce / list comprehension as loops this time because there are 4 variables
  D_complex_energy = 0+0j #identity of complex sum
  for i1 in range(-1 * num_field_coeffs, num_field_coeffs + 1):
    for i2 in range(-1 * num_field_coeffs, num_field_coeffs + 1):
      for j1 in range(-1 * num_field_coeffs, num_field_coeffs + 1):
        for j2 in range(-1 * num_field_coeffs, num_field_coeffs + 1):
          D_complex_energy += (field_coeffs[i1]*field_coeffs[i2]*
                                field_coeffs[j1].conjugate()*field_coeffs[j2].conjugate()*
                                                A_integrals[(i1+i2-j1-j2)])
  if not math.isclose(D_complex_energy.imag, 0, abs_tol=1e-8):
    print("nonzero imaginary energy component in D", D_complex_energy)
  return alpha*A_complex_energy.real + C * B_complex_energy.real + 0.5*u*D_complex_energy.real


def calc_bending_energy(amplitude, wavenumber):
  if amplitude==0:
    Kthth_integral, error= integrate.quad(lambda z: 1.0/radius**2,
                                       0, 2 * math.pi / wavenumber)
    return Kthth_integral
  else:
    Kzz_integral, error = integrate.quad(lambda z: Kzz_integrand(amplitude, z, wavenumber=wavenumber),
                                       0, 2 * math.pi / wavenumber)
    Kthth_integral, error= integrate.quad(lambda z: Kthth_integrand( amplitude, z, wavenumber=wavenumber),
                                       0, 2 * math.pi / wavenumber)
    return(Kzz_integral+Kthth_integral)

def calc_surface_energy(amplitude, wavenumber, kappa, amplitude_change=True):
  if (not A_integrals) or amplitude_change:
    evaluate_A_integral_0(amplitude)
  #print('surface area', A_integrals[0].real, "a", amplitude)
  return gamma* A_integrals[0].real + kappa*calc_bending_energy(amplitude, wavenumber)
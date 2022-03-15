import numpy as np
import random
import math
import cmath
import os
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from bisect import bisect
try:
  import surfaces.system_cylinder as system_cylinder
except ModuleNotFoundError:
  #differnt if this file is run directly ( __name__=="__main__" )
  import sys
  import os
  parent = os.getcwd()
  sys.path.append(parent)
  #print(sys.path)
  import surfaces.system_cylinder as system_cylinder
import metropolis

class Lattice():
  """
  The base class scalar order parameter+director scheme 
  Works together with a surface shape and a metropolis engine.
  """

  def __init__(self, aspect_ratio, D, n, dims, wavenumber, radius=1, shape=None, n_substeps=None):
    #material parameters
    self.aspect_ratio = aspect_ratio
    self.n=n
    self.D=D
    self.L=self.aspect_ratio*self.D
    self.particlearea= self.L*self.D+math.pi/4*self.D**2
    self.particleperimeter=2*self.L+math.pi*self.D

    #lattice characteristics
    self.wavenumber=wavenumber
    self.radius=radius
    self.shape=shape
    #don't use literally the z-direction number of lattice points provided, but scale with wavenumber
    cylinder_len_z = 2*math.pi / self.wavenumber # = wavelength lambda
    cylinder_radius_th = 2*math.pi*self.radius # circumference - in len units, not radians
    # so that z-direction pixel length is the same and results are comparable with different wavenumber
    self.z_len, self.th_len = dims
    self.z_pixel_len = cylinder_len_z /self.z_len # length divided py number of pixels
    #assert(math.isclose(self.z_pixel_len,2*math.pi/float(dims[0]))) #should be same as for wavenumber 1, length 2pi, dims[0] pixels
    self.th_pixel_len = cylinder_radius_th /self.th_len

    #run settings
    if n_substeps is None:
      self.n_substeps =self.z_len*self.th_len
    else:
      self.n_substeps = n_substeps
    self.n_dims =dims[0]*dims[1]
    self.n_sweep = self.n_dims*n_substeps

    #the lattices
    self.alpha2 = np.zeros((self.z_len, self.th_len))
    self.etas = np.zeros((self.z_len, self.th_len))
    self.K1s = np.zeros((self.z_len, self.th_len))
    self.K3s  = np.zeros((self.z_len, self.th_len))
    self.nth = np.zeros((self.z_len, self.th_len))
    self.nz = np.zeros((self.z_len, self.th_len))
    self.na = np.zeros((self.z_len, self.th_len))
    self.nb = np.zeros((self.z_len, self.th_len))
    self.divsquared = np.zeros((self.z_len, self.th_len))
    self.curlsquared = np.zeros((self.z_len, self.th_len))
    self.diagdivsquared = np.zeros((self.z_len, self.th_len))
    self.diagcurlsquared = np.zeros((self.z_len, self.th_len))
    self.gradientenergies = np.zeros((self.z_len, self.th_len))
    self.densityenergies = np.zeros((self.z_len, self.th_len))
    #angle of director
    self.director = np.zeros((self.z_len, self.th_len))

    #locals running avgs #TODO keep?
    self.avg_lattice = np.zeros((self.z_len, self.th_len), dtype=complex)
    self.avg_amplitude_profile=np.zeros((self.z_len))
    #self.avg_amplitude_history()


    self.fill_alpha2_lookup(self.aspect_ratio)
    self.random_initialize(shape)
    print("initialized")

  def fill_alpha2_lookup(self, aspect_ratio):
    #we can only do eta(alpha2)
    #make a list (alpha2, eta)
    alpha=0
    a2s=[alpha**2]
    etas=[self.eta(alpha**2, aspect_ratio)]
    while self.eta(alpha**2,aspect_ratio)==0:
      alpha+=.01 #don't fill table while eta=0
    for a in np.arange(alpha, 50, .01):
      a2s.append(a**2)
      etas.append(self.eta(a**2, aspect_ratio))
    self.alpha2lookup=(etas, a2s)
    #plt.plot(list(self.alpha2lookup.keys()), list(self.alpha2lookup.values()))
    #plt.show()

  def random_initialize(self, shape):
    #local fields  
    for z_index in range(self.z_len):
      for th_index in range(self.th_len):
        eta=.6#random.uniform(.6,.8)
        alpha2=self.get_alpha2(eta)
        self.etas[z_index, th_index] =eta
        self.alpha2[z_index, th_index] = alpha2
        self.K1s[z_index, th_index] = self.K1(alpha2)
        self.K3s[z_index, th_index] = self.K3(alpha2)
        self.densityenergies[z_index, th_index] = self.densityenergy(eta=eta, alpha2=alpha2)
        self.director[z_index, th_index] = random.uniform(0,2*math.pi)
    #fill derivatives
    for z_index in range(self.z_len):
      z_loc = z_index * self.z_pixel_len
      Ath = cy.A_theta(z=z_loc, amplitude = cy.amplitude)
      sqrtgzz = cy.sqrt_g_z(z=z_loc, amplitude = cy.amplitude)
      sqrtgthth = cy.sqrt_g_theta(z=z_loc, amplitude = cy.amplitude)
      for th_index in range(self.th_len):    
        nz = cmath.rect(1, self.director[z_index, th_index]).real  
        nth= cmath.rect(1, self.director[z_index, th_index]).imag  
        na = cmath.rect(1, self.director[z_index, th_index]-math.pi/4).real
        nb = cmath.rect(1, self.director[z_index, th_index]-math.pi/4).imag
        leftnz = cmath.rect(1, self.director[z_index-1, th_index]).real  
        leftnth= cmath.rect(1, self.director[z_index-1, th_index]).imag  
        downnz= cmath.rect(1, self.director[z_index, th_index-1]).real  
        downnth= cmath.rect(1, self.director[z_index, th_index-1]).imag  
        downleftna = cmath.rect(1, self.director[z_index-1, th_index-1]-math.pi/4).real
        downleftnb = cmath.rect(1, self.director[z_index-1, th_index-1]-math.pi/4).imag
        try:
          downrightna = cmath.rect(1, self.director[z_index+1, th_index-1]-math.pi/4).real
          downrightnb = cmath.rect(1, self.director[z_index+1, th_index-1]-math.pi/4).imag
        except IndexError:
          downrightna = cmath.rect(1, self.director[z_index+1-self.z_len, th_index-1]-math.pi/4).real
          downrightnb = cmath.rect(1, self.director[z_index+1-self.z_len, th_index-1]-math.pi/4).imag
        self.nz[z_index, th_index]=nz
        self.nth[z_index, th_index]=nth
        self.na[z_index, th_index]=na
        self.nb[z_index, th_index]=nb
        #TODO line element is wrong
        z_dist = self.z_pixel_len*sqrtgzz
        th_dist = self.th_pixel_len*sqrtgthth
        diag_pixel_len = math.sqrt(z_dist**2+th_dist**2)
        diagdivsquared= self.get_diagdivsquared(na=na, nb=nb, downleftna=downleftna, downrightnb=downrightnb, left_diag_dist=diag_pixel_len, right_diag_dist=diag_pixel_len, Ath=Ath)
        divsquared = self.get_orthogonaldivsquared(nth=nth, nz=nz, leftnz=leftnz, downnth=downnth, z_dist=z_dist, th_dist=th_dist, Ath=Ath)
        self.divsquared[z_index, th_index] = divsquared
        self.diagdivsquared[z_index, th_index] = diagdivsquared
        diagcurlsquared= self.get_diagcurlsquared(na=na, nb=nb, downleftna=downleftna, downrightnb=downrightnb, downrightna=downrightna, downleftnb=downleftnb,  z_dist = z_dist, th_dist = th_dist, left_diag_dist=diag_pixel_len, right_diag_dist=diag_pixel_len, Ath=Ath)
        curlsquared= self.get_orthogonalcurlsquared(nth=nth, nz=nz, leftnth=leftnth, downnz=downnz, z_dist=z_dist, th_dist=th_dist, Ath=Ath)
        self.curlsquared[z_index, th_index] = curlsquared
        self.diagcurlsquared[z_index, th_index] =diagcurlsquared
        self.gradientenergies[z_index, th_index] =self.gradientenergy(self.K1s[z_index, th_index], self.K3s[z_index, th_index], divsquared, diagdivsquared, curlsquared, diagcurlsquared)

  def S(self, alpha2):
    return scipy.special.iv(1,alpha2)/scipy.special.iv(0,alpha2)

  def P(self, alpha2):
    return scipy.special.iv(2,alpha2)/scipy.special.iv(0,alpha2)

  def eta(self, alpha2, aspect_ratio):
    S = self.S(alpha2)
    inveta = 1 + (8*aspect_ratio**2 *2 *S)/(3 * math.pi * (4*aspect_ratio+math.pi) * alpha2)
    return 1/inveta #TODO sometimes returns nan, not 0 

  def get_alpha2(self, eta):
    #print(self.alpha2lookup)
    index=bisect(self.alpha2lookup[0], eta)
    try:
      alpha2=self.alpha2lookup[1][index]
    except IndexError:
      #eta too big, located after end of list of etas generated
      # > return alpha2 corresponding to last eta generated
      alpha2=self.alpha2lookup[1][index-1]
    return alpha2

  def K1(self, alpha2):
    l=self.aspect_ratio
    eta = self.eta(alpha2, self.aspect_ratio)
    S = self.S(alpha2)
    P= self.P(alpha2)
    ans = 128*math.pi*eta**2*l**2*S/(9*(1-eta)*(4*l+math.pi)**2)
    ans*= (1+l**2)*S + (1-l**2)*P
    return ans

  def K3(self,alpha2):
    l=self.aspect_ratio
    eta = self.eta(alpha2, self.aspect_ratio)
    S = self.S(alpha2)
    P= self.P(alpha2)
    ans = 128*math.pi*eta**2*l*S/(9*(1-eta)*(4*l+math.pi)**2)
    ans*= (1+l**2)*l*S + .5*(18*l - 2*l**3 + 3*math.pi)*P
    return ans

  def get_orthogonalcurlsquared(self,nz, nth, leftnth, downnz, z_dist, th_dist, Ath):
    #TODO unit test these
    curl = (nz - downnz)/th_dist -(nth-leftnth)/z_dist
    return (curl+self.n*Ath*nth)**2 #there is also an index raising 1/sqrtg in there - all of gradient term has that so its put in later

  def get_diagcurlsquared(self,na, nb, downleftna, downleftnb, downrightna, downrightnb, z_dist, th_dist, left_diag_dist, right_diag_dist, Ath):
    #TODO test, unit test this
    #curl in skew basis where angle between a,b is >90degress by phi
    phi = math.pi/4 - math.atan2(th_dist, z_dist) #angle off from orthogonal: actual angle a,b = 90degrees+phi
    volumeelement = math.cos(phi)
    curl = (na - downrightna)/right_diag_dist -(nb-downleftnb)/left_diag_dist
    curl += math.sin(phi)*((na - downleftna)/left_diag_dist -(nb-downrightnb)/right_diag_dist)
    curl /= volumeelement
    #print(phi, math.cos(phi), math.sin(phi))
    #TODO how dows Ath fit in?
    # not exactly right
    Aa = math.sqrt(2)/2*Ath
    Ab=Aa
    return (curl+self.n*(Ab*nb-Aa*na))**2 

  def get_orthogonaldivsquared(self,nz, nth, leftnz, downnth, z_dist, th_dist, Ath):
    div = (nz-leftnz)/z_dist + (nth - downnth)/th_dist
    return (div+self.n*Ath*nz)**2

  def get_diagdivsquared(self,na, nb, downleftna, downrightnb, left_diag_dist, right_diag_dist, Ath):
    #the same as in orthonormal coordinates?
    div = (na-downleftna)/left_diag_dist + (nb - downrightnb)/right_diag_dist
    Aa = math.sqrt(2)/2*Ath
    Ab=Aa
    return (div+self.n*(Ab*na-Aa*nb))**2

  def gradientenergy(self,K1, K3, divsquared, diagdivsquared, curlsquared, diagcurlsquared):
    return K1/2*(divsquared+diagdivsquared)+K3/2*(curlsquared+diagcurlsquared)

  def densityenergy(self, eta, alpha2):
    #TODO examine and test
    numberdensity=eta/self.particlearea
    energy =  0 #id part?  constant for whole object?
    S = scipy.special.iv(1, alpha2)/scipy.special.iv(0, alpha2)
    n1=numberdensity*self.particleperimeter
    tensorn11= self.L*(1+S) + math.pi/2 *self.D
    tensorn22= self.L*(1-S) + math.pi/2 *self.D
    N = 5/(6*math.pi)*n1**2-2/(3*math.pi)*numberdensity**2*(tensorn11**2+tensorn22**2)
    energy += -numberdensity *math.log(1-eta) + N/(2*(1-eta))
    return energy

  """
  def gradientenergy(self, nz, nth, gradn, delcrossn, K1, K3, Ath, index_raise):
    Ath*= index_raise
    return K1*(gradn+ self.n*Ath*nz)**2 + K3*(delcrossn+self.n*Ath*nth)**2 

  def diaggradientenergy(self, na, nb, diagdivsquared, diagdelcrossn, K1, K3, Ath, index_raise):
    Ab = math.sqrt(2)/2 *(Ath*index_raise)
    Aa= Ab
    return K1*(diagdivsquared+ self.n*(Ab*na-Aa*nb))**2 + K3*(diagdelcrossn+self.n*(Ab*nb-Aa*na))**2 
  """
  def total_field_energy(self, shape):
    """calculates energy on proposed amplitude change or whole lattice change"""
    energy=0
    for z_index in range(0,self.z_len):
      z_loc = z_index * self.z_pixel_len
      z_loc_interstitial = (z_index-.5) * self.z_pixel_len
      z_spacing = shape.sqrt_g_z(z=z_loc_interstitial, amplitude=shape.amplitude)
      th_spacing = shape.sqrt_g_theta(z=z_loc, amplitude=shape.amplitude)
      col_sqrtg = z_spacing*th_spacing
      col_index_raise_and_sqrtg = shape.sqrt_g_z(z=z_loc, amplitude=shape.amplitude)/shape.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)
      col_A_th = shape.A_theta(z=z_loc_interstitial, amplitude=shape.amplitude)
      psi_col = self.psi[z_index]
      psi_squared_column = self.psi_squared[z_index]
      #TODO could do *sqrtg at the end to whole column, if covariant laplacian returned value/sqrt_g
      energy_col = (self.alpha*sum(psi_squared_column)+self.u/2*sum(psi_squared_column**2)) *col_sqrtg
      dz_col = self.dz[z_index]/z_spacing #dz does contain /z_pixel_len, but not sqrtgz adjustment
      #TODO likelyt problem part
      energy_col += self.C*sum(self.squared(dz_col))*col_sqrtg #TODO check this squares elementwise, then sums
      dth_col = self.dth[z_index]#/th_spacing <- this is the index raise?
      #C|dth Psi(x)|^2 part of energy density
      #TODO both gradient parts causing problems
      energy_col += self.C*sum(self.squared(dth_col))*col_index_raise_and_sqrtg
      #-iCn(A_th* Psi* dth Psi(x)) and c.c. part of energy density
      energy_col += self.C*self.n*2*sum((col_A_th.conjugate()*psi_col.conjugate()*dth_col).imag)*col_index_raise_and_sqrtg
      # Cn^2|A_th Psi(x)|^2  part of energy density
      energy_col += self.Cnsquared*sum(self.squared(psi_col))*self.squared(col_A_th)*col_index_raise_and_sqrtg
      energy += energy_col
    #print("amplitude ", amplitude, "would get field energy", energy*self.z_pixel_len*self.th_pixel_len)
    return energy*self.z_pixel_len*self.th_pixel_len


  def update_rescale_params(self, amplitude):
    """
    hook for subclasses to update characteristics of each cell (background energy per area, 1st order corrections)
    that change with cell size- must be updated and remembered when amplitude changes
    """
    pass

  def step_director(self, shape, sampling_width, me):
    """
    A single location step - dims*n_substeps of these make up a sweep
    """
    #choose a location
    index_z, index_th = (random.randrange(-1,(self.z_len-1)), random.randrange(-1,(self.th_len-1)))
    self.step_director_loc(index_z, index_th, shape, sampling_width, me)
    #TODO output energy, make metropolis decision here

  def step_director_loc(self, index_z, index_th, shape, sampling_width, me):
    """
    Main energy calc of single location change
    """
    z_loc = self.z_pixel_len * index_z 
    z_loc_neighbor = self.z_pixel_len * (index_z +1)
    z_loc_neighbor_left = self.z_pixel_len * (index_z -1)
    #properties of the surface at this point
    A_th= shape.A_theta(z=z_loc, amplitude=shape.amplitude)
    index_raise = 1/shape.g_theta(z=z_loc, amplitude=shape.amplitude) 
    z_pixel_len = self.z_pixel_len*shape.sqrt_g_z(z=z_loc, amplitude=shape.amplitude)
    th_pixel_len = self.th_pixel_len*shape.sqrt_g_theta(z=z_loc, amplitude=shape.amplitude)
    z_pixel_len_neighbor = self.z_pixel_len*shape.sqrt_g_z(z=z_loc_neighbor, amplitude=shape.amplitude)
    th_pixel_len_neighbor = self.th_pixel_len*shape.sqrt_g_theta(z=z_loc_neighbor, amplitude=shape.amplitude)
    z_pixel_len_left_neighbor = self.z_pixel_len*shape.sqrt_g_z(z=z_loc_neighbor_left, amplitude=shape.amplitude)
    th_pixel_len_left_neighbor = self.th_pixel_len*shape.sqrt_g_theta(z=z_loc_neighbor_left, amplitude=shape.amplitude)
    A_th_neighbor= shape.A_theta(z=z_loc_neighbor, amplitude=shape.amplitude) 
    A_th_left_neighbor= shape.A_theta(z=z_loc_neighbor_left, amplitude=shape.amplitude)  
    neighbor_index_raise = 1/shape.g_theta(z=z_loc_neighbor, amplitude=shape.amplitude)
    left_neighbor_index_raise = 1/shape.g_theta(z=z_loc_neighbor_left, amplitude=shape.amplitude)
    sqrt_g = (shape.sqrt_g_theta(z=z_loc, amplitude=shape.amplitude)*shape.sqrt_g_z(z=z_loc, amplitude=shape.amplitude))
    
    old_director = self.director[index_z, index_th]
    old_nz = self.nz[index_z, index_th]
    old_nth = self.nth[index_z, index_th]
    old_na = self.na[index_z, index_th]
    old_nb = self.nb[index_z, index_th]
    #get K1, K3
    K1 = self.K1s[index_z, index_th]
    K3 = self.K3s[index_z, index_th]
    old_energy = self.gradientenergies[index_z, index_th]

    stepsize=sampling_width#choose something order of magnitude of predicted width of distribution
    #stepsize *= sqrt_g
    new_director = random.gauss(old_director,stepsize)
    #new_director += random.choice([0, 0, 0, math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, math.pi])

    new_nz = cmath.rect(1, new_director).real
    new_nth= cmath.rect(1, new_director).imag
    new_na = cmath.rect(1, new_director-math.pi/4).real 
    new_nb= cmath.rect(1, new_director-math.pi/4).imag
    leftnz = self.nz[index_z-1, index_th]
    leftnth= self.nth[index_z-1, index_th]
    downnz= self.nz[index_z, index_th-1]
    downnth= self.nth[index_z, index_th-1]
    downleftna= self.na[index_z-1, index_th-1]
    downleftnb= self.nb[index_z-1, index_th-1]
    downrightna= self.na[index_z+1, index_th-1]
    downrightnb= self.nb[index_z+1, index_th-1]
    #TODO line element is wrong
    diag_pixel_len = math.sqrt(z_pixel_len**2+th_pixel_len**2)
    diag_pixel_len_neighbor = math.sqrt(z_pixel_len_neighbor**2+th_pixel_len_neighbor**2)
    diag_pixel_len_left_neighbor = math.sqrt(z_pixel_len_left_neighbor**2+th_pixel_len_left_neighbor**2)
    new_divsquared = self.get_orthogonaldivsquared(nth=new_nth, nz=new_nz, leftnz=leftnz, downnth=downnth, z_dist=z_pixel_len, th_dist=th_pixel_len, Ath=A_th)
    #assert(math.isclose(new_divsquared,self.divsquared[index_z, index_th]))
    new_diagdivsquared =  self.get_diagdivsquared(na=new_na, nb=new_nb, downleftna=downleftna, downrightnb=downrightnb, left_diag_dist=diag_pixel_len, right_diag_dist=diag_pixel_len, Ath=A_th)
    #assert(math.isclose(new_diagdivsquared,self.diagdivsquared[index_z, index_th]))
    new_curlsquared =  self.get_orthogonalcurlsquared(nth=new_nth, nz=new_nz, leftnth=leftnth, downnz=downnz, z_dist=z_pixel_len, th_dist=th_pixel_len, Ath=A_th)
    #assert(math.isclose(new_curlsquared,self.curlsquared[index_z, index_th]))
    new_diagcurlsquared=self.get_diagcurlsquared(na=new_na, nb=new_nb, downleftna=downleftna, downrightnb=downrightnb, downrightna=downrightna, downleftnb=downleftnb,  z_dist = z_pixel_len, th_dist = th_pixel_len, left_diag_dist=diag_pixel_len, right_diag_dist=diag_pixel_len, Ath=A_th)
    #assert(math.isclose(new_diagdelcrossn,self.diagcurlsquared[index_z, index_th]))
    new_energy = self.gradientenergy(K1, K3, new_divsquared, new_diagdivsquared, new_curlsquared, new_diagcurlsquared)
    diff_energy = new_energy - old_energy
    #assert(math.isclose(new_energy,old_energy))

    new_rightdivsquared = self.get_orthogonaldivsquared(nth=self.nth[index_z+1, index_th], nz=self.nz[index_z+1, index_th], leftnz=new_nz, downnth=self.nth[index_z+1, index_th-1], z_dist=z_pixel_len_neighbor, th_dist=th_pixel_len_neighbor, Ath=A_th_neighbor)
    new_rightcurlsquared = self.get_orthogonalcurlsquared(nth=self.nth[index_z+1, index_th], nz=self.nz[index_z+1, index_th], downnz=self.nz[index_z+1, index_th-1], leftnth=new_nth, z_dist=z_pixel_len_neighbor, th_dist=th_pixel_len_neighbor, Ath=A_th_neighbor)
    new_rightenergy = self.gradientenergy(self.K1s[index_z+1, index_th], self.K3s[index_z+1, index_th], new_rightdivsquared, self.diagdivsquared[index_z+1, index_th], new_rightcurlsquared, self.diagcurlsquared[index_z+1, index_th])
    old_rightenergy = self.gradientenergies[index_z+1, index_th]
    #assert(math.isclose(new_rightenergy,old_rightenergy))
    diff_energy += (new_rightenergy - old_rightenergy)

    new_updivsquared = self.get_orthogonaldivsquared(nth=self.nth[index_z, index_th+1], nz=self.nz[index_z, index_th+1], leftnz=self.nz[index_z-1, index_th+1], downnth=new_nth, z_dist=z_pixel_len, th_dist=th_pixel_len, Ath=A_th)
    new_upcurlsquared = self.get_orthogonalcurlsquared(nth=self.nth[index_z, index_th+1], nz=self.nz[index_z, index_th+1], leftnth=self.nth[index_z-1, index_th+1], downnz=new_nz, z_dist=z_pixel_len, th_dist=th_pixel_len, Ath=A_th)
    new_upenergy = self.gradientenergy(self.K1s[index_z, index_th+1], self.K3s[index_z, index_th+1], new_updivsquared, self.diagdivsquared[index_z, index_th+1], new_upcurlsquared, self.diagcurlsquared[index_z, index_th+1])
    old_upenergy = self.gradientenergies[index_z, index_th+1]
    #assert(math.isclose(new_upenergy, old_upenergy))
    diff_energy += (new_upenergy - old_upenergy)

    uprightna = self.na[index_z+1, index_th+1]
    uprightnb = self.nb[index_z+1, index_th+1]
    try:
      new_uprightdivsquared = self.get_diagdivsquared(uprightna, uprightnb, downleftna=new_na, downrightnb=self.nb[index_z+2, index_th], left_diag_dist=diag_pixel_len_neighbor, right_diag_dist=diag_pixel_len_neighbor, Ath=A_th_neighbor)
      new_uprightcurlsquared = self.get_diagcurlsquared(uprightna, uprightnb, downleftna=new_na, downrightnb=self.nb[index_z+2, index_th],downleftnb=new_nb, downrightna=self.na[index_z+2, index_th], z_dist=z_pixel_len_neighbor, th_dist=th_pixel_len_neighbor, left_diag_dist=diag_pixel_len_neighbor, right_diag_dist=diag_pixel_len_neighbor, Ath=A_th_neighbor)
    except IndexError:
      new_uprightdivsquared = self.get_diagdivsquared(uprightna, uprightnb, downleftna=new_na, downrightnb=self.nb[index_z+2-self.z_len, index_th], left_diag_dist=diag_pixel_len_neighbor, right_diag_dist=diag_pixel_len_neighbor, Ath=A_th_neighbor)
      new_uprightcurlsquared = self.get_diagcurlsquared(uprightna, uprightnb, downleftna=new_na, downrightnb=self.nb[index_z+2-self.z_len, index_th],downleftnb=new_nb, downrightna=self.na[index_z+2-self.z_len, index_th], z_dist=z_pixel_len_neighbor, th_dist=th_pixel_len_neighbor, left_diag_dist=diag_pixel_len_neighbor, right_diag_dist=diag_pixel_len_neighbor, Ath=A_th_neighbor)
    #assert(math.isclose(new_uprightcurlsquared,self.diagcurlsquared[index_z+1, index_th+1]))
    new_uprightenergy = self.gradientenergy(self.K1s[index_z+1, index_th+1], self.K3s[index_z+1, index_th+1], self.divsquared[index_z+1, index_th+1], new_uprightdivsquared , self.curlsquared[index_z+1, index_th+1], new_uprightcurlsquared)
    old_uprightenergy = self.gradientenergies[index_z+1, index_th+1]
    #assert(math.isclose(new_uprightenergy, old_uprightenergy))
    diff_energy += (new_uprightenergy - old_uprightenergy)

    upleftna = self.na[index_z-1, index_th+1]
    upleftnb = self.nb[index_z-1, index_th+1]
    new_upleftdivsquared = self.get_diagdivsquared(upleftna, upleftnb, downleftna=self.na[index_z-2, index_th], downrightnb=new_nb, left_diag_dist=diag_pixel_len_left_neighbor, right_diag_dist=diag_pixel_len_left_neighbor, Ath=A_th_left_neighbor)
    new_upleftcurlsquared = self.get_diagcurlsquared(upleftna, upleftnb, downleftna=self.na[index_z-2, index_th], downrightna=new_na,downleftnb=self.nb[index_z-2, index_th], downrightnb=new_nb,th_dist=th_pixel_len_left_neighbor, z_dist=z_pixel_len_left_neighbor, left_diag_dist=diag_pixel_len_left_neighbor, right_diag_dist=diag_pixel_len_left_neighbor, Ath=A_th_left_neighbor)
    new_upleftenergy = self.gradientenergy(self.K1s[index_z-1, index_th+1], self.K3s[index_z-1, index_th+1], self.divsquared[index_z-1, index_th+1], new_upleftdivsquared , self.curlsquared[index_z-1, index_th+1], new_upleftcurlsquared)
    old_upleftenergy = self.gradientenergies[index_z-1, index_th+1]
    #assert(math.isclose(new_upleftenergy, old_upleftenergy))
    diff_energy += (new_upleftenergy - old_upleftenergy)

    diff_energy*=sqrt_g
    diff_energy*=self.z_pixel_len*self.th_pixel_len #- included in renormalizing coefficients
    accept = me.metropolis_decision(0,diff_energy)
    #print("director", old_director, "to ", new_director)
    #print("energy diff", diff_energy, "accept", accept)
    if accept:
      #print("accept")
      #change stored value of pixel
      self.director[index_z,index_th] = new_director
      self.nz[index_z, index_th]  = new_nz
      self.nth[index_z, index_th]  = new_nth
      self.na[index_z, index_th]  = new_na
      self.nb[index_z, index_th]  = new_nb

      self.divsquared[index_z, index_th] = new_divsquared
      self.curlsquared[index_z, index_th] = new_curlsquared
      self.diagdivsquared[index_z, index_th] = new_diagdivsquared
      self.diagcurlsquared[index_z, index_th] = new_diagcurlsquared
      self.gradientenergies[index_z, index_th] = new_energy

      self.divsquared[index_z+1, index_th] = new_rightdivsquared
      self.curlsquared[index_z+1, index_th] = new_rightcurlsquared
      self.gradientenergies[index_z+1, index_th] = new_rightenergy

      self.divsquared[index_z, index_th+1] = new_updivsquared
      self.curlsquared[index_z, index_th+1] = new_upcurlsquared
      self.gradientenergies[index_z, index_th+1] = new_upenergy

      self.diagdivsquared[index_z+1, index_th+1] = new_uprightdivsquared
      self.diagcurlsquared[index_z+1, index_th+1] = new_uprightcurlsquared
      self.gradientenergies[index_z+1, index_th+1] = new_uprightenergy

      self.diagdivsquared[index_z-1, index_th+1] = new_upleftdivsquared
      self.diagcurlsquared[index_z-1, index_th+1] = new_upleftcurlsquared
      self.gradientenergies[index_z-1, index_th+1] = new_upleftenergy
    me.update_sigma_max2pi(accept, name="director")

  def step_alpha2_single(self, shape, sampling_width, me):
    """
    A single location step - dims*n_substeps of these make up a sweep
    """
    #choose a location
    index_z, index_th = (random.randrange(-1,(self.z_len-1)), random.randrange(-1,(self.th_len-1)))
    self.step_alpha2_loc(index_z, index_th, shape, sampling_width, me)
    #TODO output energy, make metropolis decision here

  def step_alpha2_loc_single(self, index_z, index_th, shape, sampling_width, me):
    """
    Main energy calc of single location change
    """
    z_loc = self.z_pixel_len * index_z 
    sqrt_g = (shape.sqrt_g_theta(z=z_loc, amplitude=shape.amplitude)*shape.sqrt_g_z(z=z_loc, amplitude=shape.amplitude))

    old_alpha2 = self.alpha2[index_z, index_th]
    K1 = self.K1s[index_z, index_th]
    K3 = self.K3s[index_z, index_th]
    old_energy = self.gradientenergies[index_z, index_th]

    stepsize=sampling_width#choose something order of magnitude of predicted width of distribution
    #stepsize *= sqrt_g
    new_alpha2 = random.gauss(old_alpha2,stepsize)
    #new_director += random.choice([0, 0, 0, math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, math.pi])
    new_K1 = self.K1(new_alpha2)
    new_K3 = self.K3(new_alpha2)
    new_energy = self.gradientenergy(new_K1, new_K3, self.divsquared[index_z, index_th], self.diagdivsquared[index_z, index_th-1], self.curlsquared[index_z, index_th-1], self.diagcurlsquared[index_z, index_th-1])

    diff_energy = new_energy - old_energy
    diff_energy*=sqrt_g
    diff_energy*=self.z_pixel_len*self.th_pixel_len
    accept  = me.metropolis_decision(0, diff_energy)
    if accept:
      #print("accept")
      #change stored value of pixel
      self.alpha2[index_z,index_th] = new_alpha2
    me.update_sigma(accept, name="alpha2")

  def step_alpha2(self, shape, sampling_width, me):
    """
    Switch step
    """
    #choose a location
    index_z_1, index_th_1 = (random.randrange(-1,(self.z_len-1)), random.randrange(-1,(self.th_len-1)))
    index_z_2, index_th_2 = (random.randrange(-1,(self.z_len-1)), random.randrange(-1,(self.th_len-1)))
    self.step_alpha2_loc(index_z_1, index_th_1,  index_z_2, index_th_2, shape, sampling_width, me)
    #TODO output energy, make metropolis decision here

  def step_alpha2_loc(self, index_z_1, index_th_1, index_z_2, index_th_2, shape, sampling_width, me):
    """
    Main energy calc of single location change
    """
    z_loc_1 = self.z_pixel_len * index_z_1 
    sqrt_g_1 = (shape.sqrt_g_theta(z=z_loc_1, amplitude=shape.amplitude)*shape.sqrt_g_z(z=z_loc_1, amplitude=shape.amplitude))
    z_loc_2 = self.z_pixel_len * index_z_2 
    sqrt_g_2 = (shape.sqrt_g_theta(z=z_loc_2, amplitude=shape.amplitude)*shape.sqrt_g_z(z=z_loc_2, amplitude=shape.amplitude))

    old_alpha2_1 = self.alpha2[index_z_1, index_th_1]
    old_energy_1 = self.gradientenergies[index_z_1, index_th_1]
    old_alpha2_2 = self.alpha2[index_z_2, index_th_2]
    old_energy_2 = self.gradientenergies[index_z_2, index_th_2]

    stepsize=sampling_width#choose something order of magnitude of predicted width of distribution
    #stepsize *= sqrt_g
    new_alpha2_1 = random.gauss(old_alpha2_1,stepsize)
    #new_director += random.choice([0, 0, 0, math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, math.pi])
    new_K1_1 = self.K1(new_alpha2_1)
    new_K3_1 = self.K3(new_alpha2_1)
    new_energy_1 = self.gradientenergy(new_K1_1, new_K3_1, self.divsquared[index_z_1, index_th_1], self.diagdivsquared[index_z_1, index_th_1-1], self.curlsquared[index_z_1, index_th_1-1], self.diagcurlsquared[index_z_1, index_th_1-1])
    diff_energy_1 = new_energy_1 - old_energy_1
    diff_energy_1*=sqrt_g_1

    #balance material exchange
    old_density_1 = self.eta(old_alpha2_1, self.aspect_ratio)
    new_density_1 = self.eta(new_alpha2_1, self.aspect_ratio)
    density_diff_1 = new_density_1 - old_density_1
    density_diff_2 = density_diff_1/sqrt_g_1*sqrt_g_2
    print(density_diff_1, sqrt_g_1,density_diff_2,sqrt_g_2)
    old_density_2 = self.eta(old_alpha2_2, self.aspect_ratio)
    target_density_2 =old_density_2-density_diff_2

    new_alpha2_2 = self.get_alpha2(target_density_2)
    #new_director += random.choice([0, 0, 0, math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, math.pi])
    new_K1_2 = self.K1(new_alpha2_2)
    new_K3_2 = self.K3(new_alpha2_2)
    new_energy_2 = self.gradientenergy(new_K1_2, new_K3_2, self.divsquared[index_z_2, index_th_2], self.diagdivsquared[index_z_2, index_th_2-1], self.curlsquared[index_z_2, index_th_2-1], self.diagcurlsquared[index_z_2, index_th_2-1])
    diff_energy_2 = new_energy_2 - old_energy_2
    diff_energy_2*=sqrt_g_2

    diff_energy=diff_energy_1+diff_energy_2
    diff_energy*=self.z_pixel_len*self.th_pixel_len
    accept  = me.metropolis_decision(0, diff_energy)
    print(diff_energy, accept)
    if accept:
      #print("accept")
      #change stored value of pixel
      self.alpha2[index_z_1,index_th_1] = new_alpha2_1s
      self.etas[index_z_1,index_th_1] = new_density_1
      self.K1s[index_z_1,index_th_1] = new_K1_1
      self.K3s[index_z_1,index_th_1] = new_K1_1

      self.alpha2[index_z_2,index_th_2] = new_alpha2_2
      self.etas[index_z_1,index_th_1] = new_density_2
      self.K1s[index_z_2,index_th_2] = new_K1_2
      self.K3s[index_z_2,index_th_2] = new_K1_2
    me.update_sigma(accept, name="alpha2")
    
  def step_eta(self, shape, sampling_width, me):
    """
    Switch step
    """
    #choose a location
    index_z_1, index_th_1 = (random.randrange(-1,(self.z_len-1)), random.randrange(-1,(self.th_len-1)))
    index_z_2, index_th_2 = (random.randrange(-1,(self.z_len-1)), random.randrange(-1,(self.th_len-1)))
    self.step_eta_loc(index_z_1, index_th_1,  index_z_2, index_th_2, shape, sampling_width, me)
    #TODO output energy, make metropolis decision here

  def step_eta_loc(self, index_z_1, index_th_1, index_z_2, index_th_2, shape, sampling_width, me):
    """
    Main energy calc of single location change
    """
    z_loc_1 = self.z_pixel_len * index_z_1 
    sqrt_g_1 = (shape.sqrt_g_theta(z=z_loc_1, amplitude=shape.amplitude)*shape.sqrt_g_z(z=z_loc_1, amplitude=shape.amplitude))
    z_loc_2 = self.z_pixel_len * index_z_2 
    sqrt_g_2 = (shape.sqrt_g_theta(z=z_loc_2, amplitude=shape.amplitude)*shape.sqrt_g_z(z=z_loc_2, amplitude=shape.amplitude))

    old_eta_1 = self.etas[index_z_1, index_th_1]
    old_energy_1 = self.gradientenergies[index_z_1, index_th_1]+self.densityenergies[index_z_1, index_th_1]
    old_eta_2 = self.etas[index_z_2, index_th_2]
    old_energy_2 = self.gradientenergies[index_z_2, index_th_2]+self.densityenergies[index_z_2, index_th_2]

    stepsize=sampling_width#choose something order of magnitude of predicted width of distribution
    #stepsize *= sqrt_g
    diff = random.gauss(0,stepsize)
    new_eta_1 = old_eta_1 + diff
    #balance material exchange
    scaled_diff = diff/sqrt_g_2*sqrt_g_1
    #print(diff, sqrt_g_1,scaled_diff,sqrt_g_2)
    new_eta_2 =old_eta_2-scaled_diff
    #TODO could transfer this rule to metropolis module
    #try again while generating invalid volume fractions
    while new_eta_1<0 or new_eta_2<0 or new_eta_1>1 or new_eta_2>1:
      me.update_sigma(False, name="eta")
      diff = random.gauss(0,stepsize)
      new_eta_1 = old_eta_1 + diff
      #balance material exchange
      scaled_diff = diff/sqrt_g_2*sqrt_g_1
      #print(diff, sqrt_g_1,scaled_diff,sqrt_g_2)
      new_eta_2 =old_eta_2-scaled_diff

    #new_director += random.choice([0, 0, 0, math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, math.pi])
    new_alpha2_1 = self.get_alpha2(new_eta_1)
    new_K1_1 = self.K1(new_alpha2_1)
    new_K3_1 = self.K3(new_alpha2_1)
    new_energy_1 = self.gradientenergy(new_K1_1, new_K3_1, self.divsquared[index_z_1, index_th_1], self.diagdivsquared[index_z_1, index_th_1-1], self.curlsquared[index_z_1, index_th_1-1], self.diagcurlsquared[index_z_1, index_th_1-1])
    new_energy_1+=self.densityenergy(eta=new_eta_1, alpha2=new_alpha2_1)
    diff_energy_1 = new_energy_1 - old_energy_1
    diff_energy_1*=sqrt_g_1

    new_alpha2_2 = self.get_alpha2(new_eta_2)
    #new_director += random.choice([0, 0, 0, math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, math.pi])
    new_K1_2 = self.K1(new_alpha2_2)
    new_K3_2 = self.K3(new_alpha2_2)
    new_energy_2 = self.gradientenergy(new_K1_2, new_K3_2, self.divsquared[index_z_2, index_th_2], self.diagdivsquared[index_z_2, index_th_2-1], self.curlsquared[index_z_2, index_th_2-1], self.diagcurlsquared[index_z_2, index_th_2-1])
    new_energy_2+=self.densityenergy(eta=new_eta_2, alpha2=new_alpha2_2)
    diff_energy_2 = new_energy_2 - old_energy_2
    diff_energy_2*=sqrt_g_2

    diff_energy=diff_energy_1+diff_energy_2
    diff_energy*=self.z_pixel_len*self.th_pixel_len
    accept  = me.metropolis_decision(0, diff_energy)
    #print(diff_energy, accept)
    if accept:
      #print("accept")
      #change stored value of pixel
      self.alpha2[index_z_1,index_th_1] = new_alpha2_1
      self.etas[index_z_1,index_th_1] = new_eta_1
      self.K1s[index_z_1,index_th_1] = new_K1_1
      self.K3s[index_z_1,index_th_1] = new_K1_1

      self.alpha2[index_z_2,index_th_2] = new_alpha2_2
      self.etas[index_z_2,index_th_2] = new_eta_2
      self.K1s[index_z_2,index_th_2] = new_K1_2
      self.K3s[index_z_2,index_th_2] = new_K1_2
    me.update_sigma(accept, name="eta")
      
  def step_lattice_all(self, shape, sampling_width, me, old_energy):
    addition = random.gauss(0, sampling_width)+random.gauss(0, sampling_width)*1j 
    lattice_addition = np.full((self.z_len, self.th_len), addition)
    new_lattice=self.psi+lattice_addition
    new_psi_squared = abs(np.multiply(new_lattice, new_lattice.conjugate())) #np.square is elementwise square of complex numbers, z^2 not zz*
    new_energy=self.total_field_energy(shape) #dz,dth differnces nuchanged by uniform addition
    accept= me.metropolis_decision(old_energy, new_energy)#TODO check order
    if accept:
       self.psi=new_lattice
       self.psi_squared=new_psi_squared
    me.update_sigma(accept, name="field")

  
if __name__ == "__main__":
  """
  main for test
  """
  import matplotlib.pyplot as plt

  n_substeps=10
  base_dims = (50, 50)
  wavenumber=.9
  radius=1
  #makes a shape object
  gamma=1
  kappa=0
  amplitude=.7
  cy = system_cylinder.Cylinder(gamma=gamma, kappa=kappa, wavenumber=wavenumber, radius=radius, amplitude = amplitude)


  dims = (int(math.floor(base_dims[0]/wavenumber)), base_dims[1])
  lattice = Lattice(aspect_ratio=5, n=2, D=.0001, dims=dims, wavenumber=wavenumber, radius=radius, shape=cy, n_substeps=n_substeps)
  
  
  #test random initialize
  m=  lambda x: (x+2*math.pi)%(2*math.pi)
  #plt.plot(lattice.alpha2[16], label="a2", linestyle=":")
  plt.plot(lattice.etas[16], label="eta", linestyle=":")
  #plt.plot(lattice.K1s[16], label="alpha2", linestyle=":")
  #plt.plot(lattice.gradientenergies[16], label="energy", linestyle=":")

  #makes a metropolis object
  temperature=1
  sigmas_initial = {"director":.01, "eta":.01}
  me = metropolis.Metropolis(temperature=temperature, sigmas_init=sigmas_initial, target_acceptance=.3)

  #mock data collector
  field_energy_history= []

  #run - test director stepping

  n_steps =500
  
  for i in range(20):
    for j in range(n_substeps):
      for ii in range(lattice.n_dims):
        lattice.step_director(shape=cy, sampling_width=me.sigmas["director"], me=me)
    print(me.sigmas)
    print(i)
  for i in range(20):
    for j in range(n_substeps):
      for ii in range(lattice.n_dims):
        lattice.step_eta(shape=cy, sampling_width=me.sigmas["eta"], me=me)
    print(me.sigmas)
    print(i)
  for i in range(n_steps):
    for j in range(n_substeps):
      for ii in range(lattice.n_dims):
        lattice.step_eta(shape=cy, sampling_width=me.sigmas["eta"], me=me)
        lattice.step_director(shape=cy, sampling_width=me.sigmas["director"], me=me)
    #field_energy = lattice.total_field_energy(shape=cy)
    #print(field_energy)
    print(me.sigmas)
    #field_energy_history.append(field_energy)
    #lattice.step_lattice_all(shape=cy, sampling_width=me.sigmas["field"], me=me, old_energy=field_energy)
    print(i)

  #mock output
  #plt.plot(lattice.alpha2[16], label="a2")
  plt.plot(lattice.etas[16], label="eta")
  #plt.plot(lattice.K1s[16], label="alpha2")
  #plt.plot(lattice.gradientenergies[16], label="energy")
  plt.legend()
  plt.show()
  
  m=  lambda x: (x+2*math.pi)%(2*math.pi)
  plt.imshow(m(lattice.director), cmap='hsv') 
  plt.show()
  plt.imshow(abs(lattice.etas))
  plt.show()
  plt.imshow(abs(lattice.alpha2))
  plt.show()
  plt.imshow(abs(lattice.gradientenergies))
  plt.show()
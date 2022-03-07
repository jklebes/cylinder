import numpy as np
import random
import math
import cmath
import os
import pandas as pd
import scipy
import matplotlib.pyplot as plt
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

  def __init__(self, aspect_ratio, n, dims, wavenumber, radius=1, shape=None, n_substeps=None):
    #material parameters
    self.aspect_ratio = aspect_ratio
    self.n=n

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
    self.gradn = np.zeros((self.z_len, self.th_len))
    self.delcrossn = np.zeros((self.z_len, self.th_len))
    self.gradientenergies= np.zeros((self.z_len, self.th_len))
    #angle of director
    self.director = np.zeros((self.z_len, self.th_len))

    #locals running avgs #TODO keep?
    self.avg_lattice = np.zeros((self.z_len, self.th_len), dtype=complex)
    self.avg_amplitude_profile=np.zeros((self.z_len))
    #self.avg_amplitude_history()

    self.random_initialize(shape)
    print("initialized")

  def random_initialize(self, shape):
    #local fields  
    for z_index in range(self.z_len):
      for th_index in range(self.th_len):
        alpha2=  1#random.uniform(0,1)
        self.alpha2[z_index, th_index] = alpha2
        self.etas[z_index, th_index] = self.eta(alpha2)
        self.K1s[z_index, th_index] = self.K1(alpha2)
        self.K3s[z_index, th_index] = self.K3(alpha2)
        self.director[z_index, th_index] =  random.uniform(0,2*math.pi)
    #fill derivatives
    for z_index in range(self.z_len):
      z_loc = z_index * self.z_pixel_len
      Ath = cy.A_theta(z=z_loc, amplitude = cy.amplitude)
      for th_index in range(self.th_len):    
        nz = cmath.rect(1, self.director[z_index, th_index]).real  
        nth= cmath.rect(1, self.director[z_index, th_index]).imag  
        leftnz = cmath.rect(1, self.director[z_index-1, th_index]).real  
        leftnth= cmath.rect(1, self.director[z_index-1, th_index]).imag  
        downnz= cmath.rect(1, self.director[z_index, th_index-1]).real  
        downnth= cmath.rect(1, self.director[z_index, th_index-1]).imag  
        self.nz[z_index, th_index]=nz
        self.nth[z_index, th_index]=nth
        #TODO line element is wrong
        gradn=(nz-leftnz)/self.z_pixel_len + (nth - downnth)/self.th_pixel_len
        self.gradn[z_index, th_index] = gradn
        delcrossn = (nz - downnz)/self.th_pixel_len -(nth-leftnth)/self.z_pixel_len
        self.delcrossn[z_index, th_index] =delcrossn

        self.gradientenergies[z_index, th_index] = self.gradientenergy(nz, nth, gradn, delcrossn, self.K1s[z_index, th_index], self.K3s[z_index, th_index], Ath)

  def S(self, alpha2):
    return scipy.special.iv(1,alpha2)/scipy.special.iv(0,alpha2)

  def P(self, alpha2):
    return scipy.special.iv(2,alpha2)/scipy.special.iv(0,alpha2)

  def eta(self, alpha2):
    l=self.aspect_ratio
    S = self.S(alpha2)
    inveta = 1 + (8*l**2 *2 *S)/(3 * math.pi * (4*l+math.pi) * alpha2)
    return 1/inveta

  def K1(self, alpha2):
    l=self.aspect_ratio
    eta = self.eta(alpha2)
    S = self.S(alpha2)
    P= self.P(alpha2)
    ans = 128*math.pi*eta**2*l**2*S/(9*(1-eta)*(4*l+math.pi)**2)
    ans*= (1+l**2)*S + (1-l**2)*P
    return ans

  def K3(self,alpha2):
    l=self.aspect_ratio
    eta = self.eta(alpha2)
    S = self.S(alpha2)
    P= self.P(alpha2)
    ans = 128*math.pi*eta**2*l*S/(9*(1-eta)*(4*l+math.pi)**2)
    ans*= (1+l**2)*l*S + .5*(18*l - 2*l**3 + 3*math.pi)*P
    return ans

  def gradientenergy(self, nz, nth, gradn, delcrossn, K1, K3, Ath):
    return K1*(gradn- self.n*Ath*nz)**2 + K3*(delcrossn- self.n*Ath*nth)**2 

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
    z_loc_interstitial = self.z_pixel_len * (index_z -.5)
    z_loc_neighbor_interstitial = self.z_pixel_len * (index_z +.5)
    #properties of the surface at this point
    A_th= shape.A_theta(z=z_loc_interstitial, amplitude=shape.amplitude)  #only needed at i-1/2 s
    index_raise = 1/shape.g_theta(z=z_loc_interstitial, amplitude=shape.amplitude)  #only needed at i-1/2 s
    A_th_neighbor= shape.A_theta(z=z_loc_neighbor_interstitial, amplitude=shape.amplitude)  #only needed at i-1/2 s
    neighbor_index_raise = 1/shape.g_theta(z=z_loc_neighbor_interstitial, amplitude=shape.amplitude)  #only needed at i-1/2 s 
    sqrt_g = (shape.sqrt_g_theta(z=z_loc, amplitude=shape.amplitude)*shape.sqrt_g_z(z=z_loc, amplitude=shape.amplitude))
    
    old_director = self.director[index_z, index_th]
    old_nz = self.nz[index_z, index_th]
    old_nth = self.nth[index_z, index_th]
    #get K1, K3
    K1 = self.K1s[index_z, index_th]
    K3 = self.K3s[index_z, index_th]
    old_energy = self.gradientenergies[index_z, index_th]

    stepsize=sampling_width#choose something order of magnitude of predicted width of distribution
    stepsize *= sqrt_g
    new_director = random.gauss(old_director, stepsize)

    new_nz = cmath.rect(1, new_director).real  
    new_nth= cmath.rect(1, new_director).imag  
    leftnz = self.nz[index_z-1, index_th]
    leftnth= self.nth[index_z-1, index_th]
    downnz= self.nz[index_z, index_th-1]
    downnth= self.nth[index_z, index_th-1]
    #TODO line element is wrong
    new_gradn = (new_nz-leftnz)/self.z_pixel_len + (new_nth - downnth)/self.th_pixel_len
    new_delcrossn =  (new_nz - downnz)/self.th_pixel_len - (new_nth-leftnth)/self.z_pixel_len
    new_energy = self.gradientenergy(new_nz, new_nth, new_gradn, new_delcrossn, K1, K3, A_th)
    diff_energy = new_energy - old_energy
    #TODO up and right neighbors gradient energy!

    new_rightgradn = self.gradn[index_z+1, index_th] - (self.nz[index_z+1, index_th]-old_nz)/self.z_pixel_len + (self.nz[index_z+1, index_th]-new_nz)/self.z_pixel_len
    new_rightdelcrossn = self.delcrossn[index_z+1, index_th] +(self.nth[index_z+1, index_th]-old_nth)/self.z_pixel_len - (self.nth[index_z+1, index_th]-new_nth)/self.z_pixel_len 
    new_rightenergy = self.gradientenergy(self.nz[index_z+1, index_th], self.nth[index_z+1, index_th], new_rightgradn, new_rightdelcrossn,
                                         self.K1s[index_z+1, index_th], self.K3s[index_z+1, index_th], A_th_neighbor)
    old_rightenergy = self.gradientenergies[index_z+1, index_th]
    diff_energy += (new_rightenergy - old_rightenergy)

    new_upgradn = self.gradn[index_z, index_th+1] - (self.nth[index_z, index_th+1]-old_nth)/self.th_pixel_len + (self.nth[index_z, index_th+1]-new_nth)/self.th_pixel_len
    new_updelcrossn = self.delcrossn[index_z, index_th+1] -(self.nz[index_z, index_th+1]-old_nz)/self.th_pixel_len  +(self.nz[index_z, index_th+1]-new_nz)/self.th_pixel_len 
    new_upenergy = self.gradientenergy(self.nz[index_z, index_th+1], self.nth[index_z, index_th+1], new_upgradn, new_updelcrossn,
                                         self.K1s[index_z, index_th+1], self.K3s[index_z, index_th+1], A_th)
    old_upenergy = self.gradientenergies[index_z, index_th+1]
    diff_energy += (new_upenergy - old_upenergy)

    diff_energy*=sqrt_g
    diff_energy*=self.z_pixel_len*self.th_pixel_len #- included in renormalizing coefficients
    accept = me.metropolis_decision(0,diff_energy)
    #print("director", old_director, "to ", new_director)
    #print("energy diff", diff_energy, "accept", accept)
    if accept:
      #change stored value of pixel
      self.director[index_z,index_th] = cmath.phase(complex(new_nz, new_nth))
      self.nz[index_z, index_th]  = new_nz
      self.nth[index_z, index_th]  = new_nth

      self.gradn[index_z, index_th] = new_gradn
      self.delcrossn[index_z, index_th] = new_delcrossn
      self.gradientenergies[index_z, index_th] = new_energy

      self.gradn[index_z+1, index_th] = new_rightgradn
      self.delcrossn[index_z+1, index_th] = new_rightdelcrossn
      self.gradientenergies[index_z+1, index_th] = new_rightenergy

      self.gradn[index_z, index_th+1] = new_upgradn
      self.delcrossn[index_z, index_th+1] = new_updelcrossn
      self.gradientenergies[index_z, index_th+1] = new_upenergy
    me.update_sigma(accept, name="field")
      
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
  amplitude=0
  cy = system_cylinder.Cylinder(gamma=gamma, kappa=kappa, wavenumber=wavenumber, radius=radius, amplitude = amplitude)


  dims = (int(math.floor(base_dims[0]/wavenumber)), base_dims[1])
  lattice = Lattice(aspect_ratio=5, n=1, dims=dims, wavenumber=wavenumber, radius=radius, shape=cy, n_substeps=n_substeps)
  
  
  #test random initialize
  plt.plot(lattice.alpha2[16], label="a2", linestyle=":")
  plt.plot(lattice.etas[16], label="eta", linestyle=":")
  #plt.plot(lattice.nth[16], label="nth", linestyle=":")
  #plt.plot(lattice.nz[16], label="nz", linestyle=":")
  plt.plot(lattice.gradn[16], label="grad", linestyle=":")
  plt.plot(lattice.delcrossn[16], label="cross", linestyle=":")
  #plt.plot(lattice.K1s[16], label="K1", linestyle=":")
  #plt.plot(lattice.K3s[16], label="K3", linestyle=":")

  #makes a metropolis object
  temperature=.001
  sigmas_initial = {"field":.1}
  me = metropolis.Metropolis(temperature=temperature, sigmas_init=sigmas_initial)

  #mock data collector
  field_energy_history= []

  #run - test director stepping

  n_steps = 60
  
  for i in range(n_steps):
    for j in range(n_substeps):
      for ii in range(lattice.n_dims):
        lattice.step_director(shape=cy, sampling_width=me.sigmas["field"], me=me)
    #field_energy = lattice.total_field_energy(shape=cy)
    #print(field_energy)
    print(me.sigmas)
    #field_energy_history.append(field_energy)
    #lattice.step_lattice_all(shape=cy, sampling_width=me.sigmas["field"], me=me, old_energy=field_energy)
    print(i)

  #mock output
  plt.plot(lattice.alpha2[16], label="a2")
  plt.plot(lattice.etas[16], label="eta")
  #plt.plot(lattice.nth[16], label="nth")
  #plt.plot(lattice.nz[16], label="nz")
  plt.plot(lattice.gradn[16], label="grad")
  plt.plot(lattice.delcrossn[16], label="cross")
  #plt.plot(lattice.K1s[16], label="K1")
  #plt.plot(lattice.K3s[16], label="K3")
  plt.legend()
  plt.show()
  
  m=  lambda x: (x+2*math.pi)%math.pi
  plt.imshow(m(lattice.director), cmap='hsv') 
  plt.show()
  plt.imshow(abs(lattice.nz))
  plt.show()
  plt.imshow(abs(lattice.nth))
  plt.show()
  plt.imshow(abs(lattice.gradn))
  plt.show()
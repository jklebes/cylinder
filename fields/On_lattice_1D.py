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
  import fields.On_lattice_simple as lattice
except ModuleNotFoundError:
  #problems if this file is run directly ( __name__=="__main__" )
  import sys
  import os
  parent = os.getcwd()
  sys.path.append(parent)
  import surfaces.system_cylinder as system_cylinder
  import fields.On_lattice as lattice
import metropolis

class Lattice1D():
  """
  Even simpler 1D version of On_lattice_simple
  - for Type 1 defect free states
  """
  def __init__(self, alpha, u, C, n, z_dim, wavenumber, radius=1, n_substeps=None):
    #material parameters
    self.alpha = alpha
    self.u = u
    self.C= C
    self.n= n
    self.Cnsquared = self.C*self.n**2

    #lattice characteristics
    self.wavenumber=wavenumber
    self.radius=radius
    #don't use literally the z-direction number of lattice points provided, but scale with wavenumber
    cylinder_len_z = 2*math.pi / self.wavenumber # = wavelength lambda
    cylinder_circumference = 2*math.pi*self.radius # circumference - in len units, not radians
    # so that z-direction pixel length is the same and results are comparable with different wavenumber
    self.z_len = z_dim
    self.z_pixel_len = cylinder_len_z /self.z_len # length divided py number of pixels
    #assert(math.isclose(self.z_pixel_len,2*math.pi/float(dims[0]))) #should be same as for wavenumber 1, length 2pi, dims[0] pixels

    #run settings
    if n_substeps is None:
      self.n_substeps =self.z_len
    else:
      self.n_substeps = n_substeps
    self.n_dims =z_dim
    self.n_sweep = self.n_dims*n_substeps

    #the lattices
    self.psi = np.zeros(self.z_len)
    #values also saved for convenience
    self.psi_squared = np.zeros(self.z_len) 
    #self.dz_squared = np.zeros((self.z_len, self.th_len))
    self.dz = np.zeros(self.z_len)
    # note: these are left derivatives:
    # derivative stored at i refers to difference between i-1, i positions in corrseponding 
    # grid of lattice values

    #locals running avgs #TODO keep?
    self.avg_lattice = np.zeros(self.z_len)
    #self.avg_amplitude_history()

    self.random_initialize()
    print("initialized")

  def random_initialize(self):
    for z_index in range(self.z_len):
      value = random.uniform(0,.1)
      self.psi[z_index] = value
      #fill stored energy density, 
      self.psi_squared[z_index] = value**2
    #fill derivatives
    for z_index in range(self.z_len):   
      #dz
      value= self.psi[z_index]
      left_value_z = self.psi[z_index-1]
      #just a left (backwards) derivative
      self.dz[z_index]  = value-left_value_z
    self.dz/= self.z_pixel_len
  

  def total_field_energy(self, shape):
    """calculates energy on proposed amplitude change or whole lattice change"""
    energy=0
    for z_index in range(0,self.z_len):
      z_loc = z_index * self.z_pixel_len
      z_loc_interstitial = (z_index-.5) * self.z_pixel_len
      z_spacing = shape.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude)
      psi_squared=self.psi_squared[z_index]

      energy_col = (self.alpha*psi_squared+self.u/2*psi_squared**2) *z_spacing
      dz_col = self.dz[z_index]/z_spacing #dz does contain /z_pixel_len, but not sqrtgz adjustment
      #TODO likelyt problem part
      energy_col += self.C*dz_col**2*z_spacing #TODO check this squares elementwise, then sums
      
      #print("amplitude ", amplitude, "would get field energy", energy*self.z_pixel_len*self.th_pixel_len)
      energy += energy_col
    return energy*self.z_pixel_len

  def update_rescale_params(self, amplitude):
    """
    hook for subclasses to update characteristics of each cell (background energy per area, 1st order corrections)
    that change with cell size- must be updated and remembered when amplitude changes
    """
    pass
  
  def step_lattice(self, shape, sampling_width, me):
    """
    A single location step - dims*n_substeps of these make up a sweep
    """
    #choose a location
    index_z= random.randrange(-1,(self.z_len-1))
    self.step_lattice_loc(index_z, shape, sampling_width, me)
    #TODO output energy, make metropolis decision here

  def step_lattice_loc(self, index_z, shape, sampling_width, me):
    """
    Main energy calc of single location change
    """
    z_loc = self.z_pixel_len * index_z 
    z_loc_interstitial = self.z_pixel_len * (index_z -.5)
    z_loc_neighbor_interstitial = self.z_pixel_len * (index_z +.5)
    
    sqrt_g = (shape.sqrt_g_z(z=z_loc, amplitude=amplitude))
    #TODO choose which one to use
    #sqrt_g_interstitial = (shape.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)*shape.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude))
    # random new value with magnitude similar to old value
    #TODO dynamic stepsize
    stepsize=sampling_width#choose something order of magnitude of predicted width of distribution
    #stepsize *= self.sqrt_g
    #print("sampling stepsize", stepsize)
    value=self.psi[index_z]
    new_value = value+random.gauss(0,stepsize)
    new_psi_squared = new_value**2
    #energy difference of switching the pixel:
    old_psi_squared = self.psi_squared[index_z]
    alpha_eff = self.alpha + self.Cnsquared*shape.A_theta(z=z_loc, amplitude=amplitude)**2
    old_pixel_energy_density = alpha_eff *old_psi_squared + self.u/2*old_psi_squared**2
    new_pixel_energy_density = alpha_eff * new_psi_squared + self.u/2*new_psi_squared**2
    diff_energy = new_pixel_energy_density -  old_pixel_energy_density 
    #techncally the diff is *sqrt(g), then
    #normailzed by sqrt(g) again in the end
    
    #effect on derivative at the location
    left_value_z = self.psi[index_z-1]
    new_derivative_z = (new_value-left_value_z)/self.z_pixel_len
    #term |d_i psi|^2 in energy
    diff_derivative_z = (new_derivative_z**2 - self.dz[index_z]**2)/shape.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude)**2
    
    diff_energy += self.C*( diff_derivative_z )

    #effect on derivative at neighboring locations i_z+1, i_th +1
    right_value_z = self.psi[index_z+1]
    new_neighbor_derivative_z = (right_value_z-new_value)/self.z_pixel_len
    diff_neighbor_derivative_z = (new_neighbor_derivative_z**2- self.dz[index_z+1]**2 )/shape.sqrt_g_z(z=z_loc_neighbor_interstitial, amplitude=amplitude)**2
    
    diff_energy += self.C*( diff_neighbor_derivative_z )

    #The naive or 0th order scaling: \alpha, u by cell area; c by cell area but was /lx^2, /ly^2 earlier
    diff_energy*=sqrt_g
    diff_energy*=self.z_pixel_len #- included in renormalizing coefficients
    accept = me.metropolis_decision(0,diff_energy)
    if accept:
      #change stored value of pixel
      self.psi[index_z] = new_value
      #change stored values of dth, dz across its boundaries
      #self.energy += diff_energy
      #fill stored energy density, 
      self.psi_squared[index_z]  = new_psi_squared
      #dz
      self.dz[index_z]  = new_derivative_z
      self.dz[index_z+1]  = new_neighbor_derivative_z
    me.update_sigma(accept, name="field")
      
  def step_lattice_all(self, shape, sampling_width, me, old_energy):
    addition = random.gauss(0, sampling_width)
    lattice_addition = np.full(self.z_len, addition)
    new_lattice=self.psi+lattice_addition
    new_psi_squared = np.square(new_lattice) #np.square is elementwise square of complex numbers, z^2 not zz*
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

  n_substeps=100
  base_dim = 50
  wavenumber=.9
  radius=1
  alpha=-1
  dim = int(math.floor(base_dim/wavenumber))
  lattice = Lattice1D(alpha=alpha, u=1, C=.1, n=6, z_dim=dim, wavenumber=wavenumber, radius=radius, n_substeps=n_substeps)
  
  #mock of run loop fixed_amplitude in main.py:
  #makes a shape object
  gamma=1
  kappa=0
  amplitude=.8
  cy = system_cylinder.Cylinder(gamma=gamma, kappa=kappa, wavenumber=wavenumber, radius=radius)

  #makes a metropolis object
  temperature=.001
  sigmas_initial = {"field":.05}
  me = metropolis.Metropolis(temperature=temperature, sigmas_init=sigmas_initial)

  #mock data collector
  field_energy_history= []

  #run
  n_steps = 100
  for i in range(n_steps):
    for j in range(n_substeps):
      for ii in range(lattice.n_dims):
        lattice.step_lattice(shape=cy, sampling_width=me.sigmas["field"], me=me)
    field_energy = lattice.total_field_energy(shape=cy)
    #print(field_energy)
    field_energy_history.append(field_energy)
    lattice.step_lattice_all(shape=cy, sampling_width=me.sigmas["field"], me=me, old_energy=field_energy)
    print(field_energy, me.sigmas)

  #mock output
  plt.plot(field_energy_history)
  plt.show()
  plt.plot(lattice.psi)
  #plt.ylim((0, 1.1*math.sqrt(abs(alpha))))
  plt.show()

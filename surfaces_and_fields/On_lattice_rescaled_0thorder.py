import numpy as np
import random
import math
import cmath
import os
import pandas as pd
import scipy
import matplotlib.pyplot as plt
try:
  import surfaces_and_fields.system_cylinder as system_cylinder
  import surfaces_and_fields.On_lattice_simple as lattice
except ModuleNotFoundError:
  #differnt if this file is run ( __name__=="__main__" )
  #then relatve import:
  import system_cylinder as system_cylinder
  import On_lattice_simple as lattice
import metropolisengine

class Lattice(lattice.Lattice):
  def __init__(self, amplitude, wavenumber, radius, gamma, kappa, intrinsic_curvature, alpha,
                u, C, n, temperature, temperature_final, dims = (100,50), n_substeps=None):
    # all needs to happen before random_initialize in parent init:
    # alpha given is for areas 1:
    # therefore use an alpha_0 roughly smallest area * alpha
    # remember alpha_0, u_0, (c=c_0 is const) for microscopic lengthscale, roughly
    #this value that is always subtracted depends only on short lengscale 
    #add more arrays to remember between steps where amplitude doesnt change:
    # decide a microscpic lengthscale cutoff
    self.z_len = int(round(dims[0]/wavenumber))
    self.short_lengthscale_th= 2*math.pi*10**(-4)
    self.alpha_0 = alpha*self.short_lengthscale_th**2
    self.background_energy = np.zeros((self.z_len)) #per area in a bigger cell
    # rescaling correction factor int G_0 dq for each location z
    self.energy_const=0
    super().__init__(amplitude, wavenumber, radius, gamma, kappa, intrinsic_curvature, alpha,
                     u, C, n, temperature, temperature_final, dims, n_substeps)
    biggest_celldims =  (self.z_pixel_len * self.surface.sqrt_g_z(z=math.pi/(2*self.wavenumber), amplitude=.99), self.th_pixel_len * self.surface.sqrt_g_theta(z=math.pi/(2*self.wavenumber), amplitude=.99))
    self.energy_const = -self.get_background_energy(biggest_celldims) 
    print("setting total average background energy per area", self.energy_const)
    if self.alpha<0:
      self.energy_const+=.5*self.alpha**2/self.u
    #else: self.energy_const -= 0
 
  def get_background_energy(self,cell_dims):
    l_max = max(cell_dims)/self.short_lengthscale_th
    l_min = min(cell_dims)/self.short_lengthscale_th #ls,cutoff_length in units of cutoff length
    area = cell_dims[0]*cell_dims[1]
    qmax = math.sqrt(l_max*l_min) #treat anisotropic cells like square cells of same area
    smaller_qmin=math.sqrt(l_min/l_max) #=cutoff_lenth=1
    larger_qmin=math.sqrt(l_max/l_min)
    try:
      ans = (qmax**2-1**2)
      ans *= self.temperature/2
    except ValueError:
      print("non-renormalizable : alpha_0 l^2 + c q^2 < 0 encountered with alpha_0=", self.alpha_0, "l^2=", area,
            "c=",c, "q range ",smaller_qmin, "to", qmax, "or c=0")
      raise ValueError
    return ans/area + self.energy_const

  def update_rescale_params(self, amplitude):
    """
    update remembered values background energy per area as a function of z
    (and in subclasses add others to 1st order)
    that change when ampltidue changes
    """
    for z_index in range(0, self.z_len):
      z_loc = z_index * self.z_pixel_len
      cell_dims = (self.z_pixel_len * self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude), self.th_pixel_len * self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude))
      self.background_energy[z_index] = self.get_background_energy(cell_dims)
    #print(self.background_energy)


  def random_initialize(self):
    super().random_initialize()
    #also fill background energy density for each cell, while amplitude=0
    # each cell dimesions are 
    cell_dims = (self.z_pixel_len, self.th_pixel_len)
    flat_background_energy = self.get_background_energy(cell_dims)
    self.background_energy=np.full((self.z_len), flat_background_energy)

  def surface_field_energy(self, amplitude):
    """calculates energy on proposed amplitude change"""
    energy=0
    for z_index in range(0,self.z_len):
      z_loc = z_index * self.z_pixel_len
      z_loc_interstitial = (z_index-.5) * self.z_pixel_len
      col_sqrtg = self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude)*self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude)
      col_index_raise_and_sqrtg = self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude)/self.surface.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)
      col_A_th = self.surface.A_theta(z=z_loc_interstitial, amplitude=amplitude)
      col_A_th = self.surface.A_theta(z=z_loc_interstitial, amplitude=amplitude)
      psi_col = self.lattice[z_index]
      psi_squared_column = self.psi_squared[z_index]
      #TODO could do *sqrtg at the end to whole column, if covariant laplacian returned value/sqrt_g
      energy_col = (self.alpha*sum(psi_squared_column)+self.u/2*sum(psi_squared_column**2)) *col_sqrtg
      dz_col = self.dz[z_index]
      energy_col += self.C*sum(self.squared(dz_col))*col_sqrtg #TODO check this squares elementwise, then sums
      dth_col = self.dth[z_index]
      #C|dth Psi(x)|^2 part of energy density
      energy_col += self.C*sum(self.squared(dth_col))*col_index_raise_and_sqrtg
      #-iCn(A_th* Psi* dth Psi(x)) and c.c. part of energy density
      energy_col += self.C*self.n*2*sum((col_A_th.conjugate()*psi_col.conjugate()*dth_col).imag)*col_index_raise_and_sqrtg
      # Cn^2|A_th Psi(x)|^2  part of energy density
      energy_col += self.Cnsquared*sum(self.squared(psi_col))*self.squared(col_A_th)*col_index_raise_and_sqrtg
      #0th order rescale:
      #also add bacground energy from implicit fluctuations for each col
      # that's number of cells in column*background energy (per area) of each, given its dimensions(*area of each cell, sqrtg part, base part done in return)
      cell_dims = (self.z_pixel_len * self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude), self.th_pixel_len * self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude))
      energy_col += self.th_len*self.get_background_energy(cell_dims)*col_sqrtg
      energy += energy_col
    #print("amplitude ", amplitude, "would get field energy", energy*self.z_pixel_len*self.th_pixel_len)
    return energy*self.z_pixel_len*self.th_pixel_len

if __name__ == "__main__":
  lattice = Lattice(amplitude=0, wavenumber=.2, radius=1, gamma=1, kappa=0, intrinsic_curvature=0,
                  alpha=-1, u=1, C=1, n=6, temperature=.01, temperature_lattice = .01,
                    dims=(50,25))
  n_steps=10000
  n_sub_steps=lattice.z_len*lattice.th_len
  lattice.run(n_steps, n_sub_steps)
  print(lattice.lattice)
  print(lattice.lattice_acceptance_counter)
  lattice.plot()

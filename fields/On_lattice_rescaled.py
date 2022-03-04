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
  import surfaces_and_field.On_lattice_simple as lattice
except ModuleNotFoundError:
  #differnt if this file is run ( __name__=="__main__" )
  #then relatve import:
  import system_cylinder as system_cylinder
  import On lattice_simple as lattice
import metropolisengine

class Lattice_Rescale(lattice.Lattice):
  def __init__(self, amplitude, wavenumber, radius, gamma, kappa, intrinsic_curvature, alpha,
                u, C, n, temperature, temperature_lattice, dims = (100,50), n_substeps=None,
                short_len_per_pixel = 200 ):
    # all needs to happen before random_initialize in parent init:
    # decide a microscpic lengthscale cutoff
    self.short_lengthscale_th=2*pi*10**(-4)
    # alpha given is for areas 1:
    # therefore use an alpha_0 roughly smallest area * alpha
    # remember alpha_0, u_0, (c=c_0 is const) for microscopic lengthscale, roughly
    self.alpha_0 = self.alpha*self.short_lengthscale**2
    self.alpha_loop_amplitude=40 #a combinatoric factor
    #this value that is always subtracted depends only on short lengscale 
    self.integral_lower_bound = 
    #add more arrays to remember between steps where amplitude doesnt change:
    self.rescaled_alpha=np.zeros((self.z_len)) #just to note this exists, really intializaed in initialize
    self.background_energy = np.zeros((self.z_len)) #per area in a bigger cell
    # rescaling correction factor int G_0 dq for each location z
    super().__init__(amplitude, wavenumber, radius, gamma, kappa, intrinsic_curvature, alpha,
                     u, C, n, temperature, temperature_lattice, dims, n_substeps)

  def correct_alpha_0(cell_dims):
    #implicit, not passed: lower length cutoff is self.short_lengthscale
    #estimate of integral over the central square
    G_integral =
    if qmax_x != qmax_y:
      #add estimate for integral over other bits of rectangle not approximated by centrla square
      G_integral += 2*side_width*side_integral
    return alpha_0 + self.alpha_loop_amplitude*self.u*G_integral
 
  def background_energy(cell_dims):
    l_max = max(cell_dims)/cutoff_length
    l_min = min(cell_dims)/cutoff_length #ls,cutoff_length in units of cutoff length
    area = l_min*l_max
    qmax = l_max
    smaller_qmin=1 #=cutoff_lenth=1
    larger_qmin=l_max/l_min
    alpha_rescaled = alpha_0*l_max**2
    try:
      ans =  c*(1+math.log(math.pi))*(qmax**2-smaller_qmin**2)+(alpha_rescaled+ c *qmax**2)*math.log(1/math.sqrt(alpha_rescaled+c*qmax**2
                ))-(alpha_rescaled + c *smaller_qmin**2)*math.log(1/math.sqrt(alpha_rescaled+c*smaller_qmin**2 ))
      ans *= -math.pi/(2*c)
    except ValueError:
      print("non-renormalizable : alpha_0 l^2 + c q^2 < 0 encountered with alpha_0=", self.alpha_0, "l^2=", area,
            "c=",c, "q range ",smaller_qmin, "to", qmax, "or c=0")
      raise ValueError
    if l_max != l_min:
      sidewidth=2*l_min
      k1=math.sqrt(alpha_rescaled/c)
      ans-=2*sidewidth*(k1*(math.atan(larger_qmin**2/k1)-math.atan(smaller_qmin**2/k1))+.5*(larger_qmin*(
            2+math.log(math.pi/(alpha_rescaled+c*larger_qmin**2)))-smaller_qmin*(
            2+math.log(math.pi/(alpha_rescaled+c*smaller_qmin**2)))))
    return ans/area 
  

  def random_initialize(self):
    super().random_initialize()
    #also initial fill of rescaled_alpha:
    # for amplitude==0 
    alpha_rescaled = self.correct_alpha_0(cell_dims)
    self.rescaled_alpha=np.full((self.z_len), alpha_rescaled)

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
      #first order rescale correction to alpha_0 for fluctuations (scale by area hppens later)
      col_rescaled_alpha = self.alpha*(1+self.alpha_loop_amplitude*self.one_loop[z_index])
      #TODO could do *sqrtg at the end to whole column, if covariant laplacian returned value/sqrt_g
      energy_col = (col_rescaled_alpha*sum(psi_squared_column)+self.u/2*sum(psi_squared_column**2)) *col_sqrtg
      dz_col = self.dz[z_index]
      energy_col += self.C*sum(self.squared(dz_col))*col_sqrtg #TODO check this squares elementwise, then sums
      dth_col = self.dth[z_index]
      #C|dth Psi(x)|^2 part of energy density
      energy_col += self.C*sum(self.squared(dth_col))*col_index_raise_and_sqrtg
      #-iCn(A_th* Psi* dth Psi(x)) and c.c. part of energy density
      energy_col += self.C*self.n*2*sum((col_A_th.conjugate()*psi_col.conjugate()*dth_col).imag)*col_index_raise_and_sqrtg
      # Cn^2|A_th Psi(x)|^2  part of energy density
      energy_col += self.Cnsquared*sum(self.squared(psi_col))*self.squared(col_A_th)*col_index_raise_and_sqrtg

      energy += energy_col
    #print("amplitude ", amplitude, "would get field energy", energy*self.z_pixel_len*self.th_pixel_len)
    return energy*self.z_pixel_len*self.th_pixel_len


  def step_lattice_loc(self, amplitude, index_z, index_th):
    z_loc = self.z_pixel_len * index_z 
    z_loc_interstitial = self.z_pixel_len * (index_z -.5)
    z_loc_neighbor_interstitial = self.z_pixel_len * (index_z +.5)
    #properties of the surface at this point
    A_th= self.surface.A_theta(z=z_loc_interstitial, amplitude=amplitude)  #only needed at i-1/2 s
    index_raise = 1/self.surface.g_theta(z=z_loc_interstitial, amplitude=amplitude)  #only needed at i-1/2 s
    A_th_neighbor= self.surface.A_theta(z=z_loc_neighbor_interstitial, amplitude=amplitude)  #only needed at i-1/2 s
    neighbor_index_raise = 1/self.surface.g_theta(z=z_loc_neighbor_interstitial, amplitude=amplitude)  #only needed at i-1/2 s
    
    sqrt_g = (self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude)*self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude))
    #TODO choose which one to use
    #sqrt_g_interstitial = (self.surface.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)*self.surface.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude))
    # random new value with magnitude similar to old value
    #TODO dynamic stepsize
    stepsize=self.sampling_width#choose something order of magnitude of predicted width of distribution
    stepsize *= sqrt_g
    #print("sampling stepsize", stepsize)
    value=self.lattice[index_z, index_th]
    new_value = complex(value.real+random.gauss(0,stepsize), value.imag+random.gauss(0,stepsize))
    new_psi_squared = self.squared(new_value)
    #energy difference of switching the pixel:
    old_psi_squared = self.psi_squared[index_z, index_th]
    old_pixel_energy_density = self.alpha *old_psi_squared + self.u/2*old_psi_squared**2
    new_pixel_energy_density = self.alpha * new_psi_squared + self.u/2*new_psi_squared**2
    diff_energy = new_pixel_energy_density -  old_pixel_energy_density 
    #techncally the diff is *sqrt(g), then
    #normailzed by sqrt(g) again in the end
    
    #effect on derivative at the location
    left_value_z = self.lattice[index_z-1, index_th]
    left_value_th = self.lattice[index_z, index_th-1]
    new_derivative_z = (new_value-left_value_z)/self.z_pixel_len
    new_derivative_th = (new_value -left_value_th)/self.th_pixel_len
    #term |d_i psi|^2 in energy
    diff_derivative_z = (self.squared(new_derivative_z) - self.squared(self.dz[index_z, index_th]))
    diff_derivative_th = (self.squared(new_derivative_th) - self.squared(self.dth[index_z, index_th]))*index_raise
    
    diff_energy += self.C*( diff_derivative_z + diff_derivative_th )

    #effect on derivative at neighboring locations i_z+1, i_th +1
    right_value_z = self.lattice[index_z+1, index_th]
    right_value_th = self.lattice[index_z, index_th+1]
    new_neighbor_derivative_z = (right_value_z-new_value)/self.z_pixel_len
    new_neighbor_derivative_th = (right_value_th-new_value)/self.th_pixel_len
    diff_neighbor_derivative_z = (self.squared(new_neighbor_derivative_z) - self.squared(self.dz[index_z+1, index_th]) )
    diff_neighbor_derivative_th = (self.squared(new_neighbor_derivative_th) - self.squared(self.dth[index_z, index_th+1]))*neighbor_index_raise
    
    diff_energy += self.C*( diff_neighbor_derivative_z + diff_neighbor_derivative_th )
    
    #diff in cross-term in(A_th* Psi* d_th Psi) and complex conjugate, at spatial location i_th-1/2
    # (stored at array location i)
    #i(A_th* Psi* d_th Psi)+c.c. = 2*Im(A_th*Psi* d_th Psi)
    old_cross_term = (A_th.conjugate()*self.lattice[index_z,index_th].conjugate()*
                      self.dth[index_z, index_th]).imag #except for *index raise,*2nC done later to both
    new_interstitial_psi =  (new_value)# +left_value_th)/2
    new_cross_term = (A_th.conjugate()*new_interstitial_psi.conjugate()*
                      new_derivative_th).imag
    diff_energy += self.C*2*self.n*index_raise*(new_cross_term - old_cross_term) 

    #diff in cross-term in(A_th* Psi* d_th Psi) and complex conjugate, for neightbor at i+1
    old_neighbor_cross_term = (A_th_neighbor.conjugate()*self.lattice[index_z,index_th+1].conjugate()*
                      self.dth[index_z, index_th+1]).imag #except for *index raise,*2nC done later to both
    new_neighbor_cross_term = (A_th_neighbor.conjugate()*self.lattice[index_z,index_th+1].conjugate()*
                      new_neighbor_derivative_th).imag
    diff_energy += self.C*2*self.n*neighbor_index_raise*(new_neighbor_cross_term - old_neighbor_cross_term)

    diff_energy += self.Cnsquared*index_raise*self.squared(A_th)*(self.squared(new_value)-
                   self.squared(self.lattice[index_z,index_th]))
    #The naive or 0th order scaling: \alpha, u by cell area; c by cell area but was /lx^2, /ly^2 earlier
    diff_energy*=sqrt_g
    diff_energy*=self.z_pixel_len*self.th_pixel_len #- included in renormalizing coefficients
    #leaving this out like scaling effective temperature everywhere equally, 
    # relative to temperature at which surface shape is sampled
    #instead do it explicitly by setting (lower) temperature of lattice step
    # by dividing energy by this extra temperature factor 
    diff_energy/= self.temperature_factor
    accept = self.me.metropolis_decision(0,diff_energy)
    if accept:
      #change stored value of pixel
      self.lattice[index_z,index_th] = new_value
      #change stored values of dth, dz across its boundaries
      #self.energy += diff_energy
      self.lattice_acceptance_counter+=1
      #fill stored energy density, 
      self.psi_squared[index_z, index_th]  = new_psi_squared
      #dz
      self.dz[index_z, index_th]  = new_derivative_z
      self.dz[index_z+1, index_th]  = new_neighbor_derivative_z
      #dth
      self.dth[index_z, index_th] = new_derivative_th 
      self.dth[index_z, index_th+1] = new_neighbor_derivative_th
      self.acceptance_counter+=1
    self.update_sigma(accept)
      

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

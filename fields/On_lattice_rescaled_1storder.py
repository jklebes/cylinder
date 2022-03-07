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
import metropolis

class Lattice(lattice.Lattice):
  def __init__(self, amplitude, wavenumber, radius, gamma, kappa, intrinsic_curvature, alpha,
                u, C, n, temperature, dims = (100,50), n_substeps=None):
    # all needs to happen before random_initialize in parent init:
    # alpha given is for areas 1:
    # therefore use an alpha_0 roughly smallest area * alpha
    # remember alpha_0, u_0, (c=c_0 is const) for microscopic lengthscale, roughly
    #this value that is always subtracted depends only on short lengscale 
    #add more arrays to remember between steps where amplitude doesnt change:
    # decide a microscpic lengthscale cutoff
    self.z_len = int(round(dims[0]/wavenumber))
    self.short_lengthscale_th= 2*math.pi*10**(-4)
    self.alpha = alpha #value in is per unit area
    self.u = u
    self.C= C
    self.area_factor = 2/math.sqrt(math.pi) #a number needed often in approximating square domains with equal-area circles
    self.background_energy = np.zeros((self.z_len)) #per area in a bigger cell
    # rescaling correction factor int G_0 dq for each location z
    self.energy_const=0
    #a Order(u) correction to background energy
    # integral of q/(alpha+cq^2) from 1/50 to 1
    upper_limit=1*self.area_factor
    lower_limit = (1/max(dims))*self.area_factor
    try:
      integral=math.pi/self.C *(math.log(self.C*upper_limit**2+self.alpha) - math.log(self.C*lower_limit**2+self.alpha)) 
    except ValueError:
      if self.C*upper_limit**2+self.alpha>0: #only second log is the problem
        print("tried taking the log of (c q**2 + alpha)=" , self.C*lower_limit**2+self.alpha, "in flcutuation order(u) correction,")
        lower_limit = math.sqrt(abs(self.alpha)/self.C)+.0000001
        print("dialing back to correlation up to length", 1/lower_limit, "only in <Psi^4> addition to background")
        integral=math.pi/self.C *(math.log(self.C*upper_limit**2+self.alpha) - math.log(self.C*lower_limit**2+self.alpha)) 
      else: 
        print("c < |alpha|, No sort of perturbative renormalization possible of fluctuations larger than cell size, skipping order(u) correction to background energy")
        integral =0
    self.correction = self.u*temperature**3*4 * integral**2 #multiplicity of this diagram:4 #1/2 in front of u, *2 for 2 fields
    #opposite sign as ans, and pertains to whole object's energy
    # per cell/degree of freedom:
    # has to be done in q-space integration because its about long-range interactions
    print("calculated order u correction: ", self.correction, "per surface")
    self.correction /= dims[0]*dims[1]
    print(self.correction, "per cell") 
    super().__init__(amplitude, wavenumber, radius, gamma, kappa, intrinsic_curvature, alpha,
                     u, C, n, temperature, temperature_final, dims, n_substeps)
    biggest_celldims =  (self.z_pixel_len * self.surface.sqrt_g_z(z=math.pi/(2*self.wavenumber), amplitude=.99), self.th_pixel_len * self.surface.sqrt_g_theta(z=math.pi/(2*self.wavenumber), amplitude=.99))
    #self.energy_const = -self.get_background_energy(biggest_celldims, ordered_reference=False) 
    print("setting total average background energy per area", self.energy_const)

  def get_background_energy(self,cell_dims, ordered_reference=True):
    l_max = max(cell_dims)/self.short_lengthscale_th
    l_min = min(cell_dims)/self.short_lengthscale_th #ls,cutoff_length in units of cutoff length
    area = cell_dims[0]*cell_dims[1]
    """
    qmax = math.sqrt(l_max*l_min) #treat anisotropic cells like square cells of same area
    smaller_qmin=math.sqrt(l_min/l_max) #=cutoff_lenth=1
    larger_qmin=math.sqrt(l_max/l_min)
    try:
      ans =4*(l_max*l_min-1) #area of cell in Lambda^2 = number of modes
      ans *= self.temperature #*1/2 in equipartion theorem, factor of 2 for degrees of freedom of complex field
    except ValueError:
      print("non-renormalizable : alpha_0 l^2 + c q^2 < 0 encountered with alpha=", self.alpha, 
            "c=",c, "q range ",smaller_qmin, "to", qmax, "or c=0")
      raise ValueError
    #also add to refernce state: energy of ordered reference system at this scale
    """
    # thermal energy of reference plane with same cell size =
    # 2* 1/2 kBT per cell
    # energy per unit area = kbT * number of cells per area = kbT / area per cell
    ans = -self.temperature
    #a Order(u) correction to background energy
    # integral of q/(alpha+cq^2) from 1/50 to 1
    upper_limit=1*self.area_factor
    lower_limit = (1/max((self.z_len, self.th_len)))*self.area_factor
    #alpha on this lattice spacing scales with area.  self.alpha realtes to unit area
    alpha_cell = self.alpha  * area
    #c is invariant, for isotropic lattice scaling
    #for anisotropc cells, this is probably where problems come from.  one side is short and gradient energy high.
    try:
      integral=math.pi/self.C *(math.log(self.C*upper_limit**2+alpha_cell) - math.log(self.C*lower_limit**2+alpha_cell)) 
    except ValueError:
      if self.C*upper_limit**2+alpha_cell>0: #only second log is the problem
        print("tried taking the log of (c q**2 + alpha)=" , self.C*lower_limit**2+self.alpha, "in flcutuation order(u) correction,")
        lower_limit = math.sqrt(abs(alpha_cell)/self.C)+.0000001
        print("dialing back to correlation up to length", 1/lower_limit, "only in <Psi^4> addition to background")
        integral=math.pi/self.C *(math.log(self.C*upper_limit**2+alpha_cell) - math.log(self.C*lower_limit**2+alpha_cell)) 
      else: 
        print("c < |alpha|, No sort of perturbative renormalization possible of fluctuations larger than cell size, skipping order(u) correction to background energy")
        integral =0
    correction = self.u*self.temperature**3*4 * integral**2 #multiplicity of this diagram:4 #1/2 in front of u, *2 for 2 fields
    correction /= self.z_len * self.th_len
    ans+= correction #correction is per cell
    alpha=self.alpha
    if alpha<0 and ordered_reference:
      energy_ordered_reference = .5 *alpha**2/self.u #this is an energy per unit area
    else:
      energy_ordered_reference=0
    return ans/area  + energy_ordered_reference #+ self.energy_const
    
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
    print("flat backgroud energt", self.background_energy[5])

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
    diff_energy = (new_pixel_energy_density -  old_pixel_energy_density)
    #print(diff_energy) 
    #effect on derivative at the location
    left_value_z = self.lattice[index_z-1, index_th]
    left_value_th = self.lattice[index_z, index_th-1]
    new_derivative_z = (new_value-left_value_z)/self.z_pixel_len
    new_derivative_th = (new_value -left_value_th)/self.th_pixel_len
    #term |d_i psi|^2 in energy
    diff_derivative_z = (self.squared(new_derivative_z) - self.squared(self.dz[index_z, index_th]))/(self.surface.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude))**2
    diff_derivative_th = (self.squared(new_derivative_th) - self.squared(self.dth[index_z, index_th]))*index_raise
    
    diff_energy += self.C*( diff_derivative_z + diff_derivative_th )

    #effect on derivative at neighboring locations i_z+1, i_th +1
    right_value_z = self.lattice[index_z+1, index_th]
    right_value_th = self.lattice[index_z, index_th+1]
    new_neighbor_derivative_z = (right_value_z-new_value)/self.z_pixel_len
    new_neighbor_derivative_th = (right_value_th-new_value)/self.th_pixel_len
    diff_neighbor_derivative_z = (self.squared(new_neighbor_derivative_z) - self.squared(self.dz[index_z+1, index_th]) )/self.surface.sqrt_g_z(z=z_loc_neighbor_interstitial, amplitude=amplitude)**2
    diff_neighbor_derivative_th = (self.squared(new_neighbor_derivative_th) - self.squared(self.dth[index_z, index_th+1]))*neighbor_index_raise
    
    diff_energy += self.C*( diff_neighbor_derivative_z + diff_neighbor_derivative_th )
    
    
    #diff in cross-term in(A_th* Psi* d_th Psi) and complex conjugate, at spatial location i_th-1/2
    # (stored at array location i)
    #i(A_th* Psi* d_th Psi)+c.c. = 2*Im(A_th*Psi* d_th Psi)
    old_cross_term = (A_th.conjugate()*self.lattice[index_z,index_th].conjugate()*
                      self.dth[index_z, index_th]/self.surface.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)).imag
                      #except for *index raise,*2nC done later to both
    new_interstitial_psi =  (new_value)# +left_value_th)/2
    new_cross_term = (A_th.conjugate()*new_interstitial_psi.conjugate()*
                      new_derivative_th/self.surface.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)).imag
    diff_energy += self.C*2*self.n*index_raise*(new_cross_term - old_cross_term) 

    #diff in cross-term in(A_th* Psi* d_th Psi) and complex conjugate, for neightbor at i+1
    old_neighbor_cross_term = (A_th_neighbor.conjugate()*self.lattice[index_z,index_th+1].conjugate()*
                      self.dth[index_z, index_th+1]/self.surface.sqrt_g_theta(z=z_loc_neighbor_interstitial, amplitude=amplitude)).imag
    new_neighbor_cross_term = (A_th_neighbor.conjugate()*self.lattice[index_z,index_th+1].conjugate()*
                      new_neighbor_derivative_th/self.surface.sqrt_g_theta(z=z_loc_neighbor_interstitial, amplitude=amplitude)).imag
    diff_energy += self.C*2*self.n*neighbor_index_raise*(new_neighbor_cross_term - old_neighbor_cross_term)

    diff_energy += self.Cnsquared*index_raise*self.squared(A_th)*(self.squared(new_value)-
                   self.squared(self.lattice[index_z,index_th]))
    
    #The scaling of alpha', u' by area
    diff_energy*=sqrt_g
    diff_energy*=self.z_pixel_len*self.th_pixel_len #- included in renormalizing coefficients
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
      

  def sublattice_field_energy(self, amplitude, z_index_start, lattice, psi_squared, dz, dth):
    energy=0
    z_dim, th_dim = lattice.shape
    #again passed end index because negative indeices out of range are better handled
    for i, z_index in enumerate(range(z_index_start,z_index_start+z_dim)): # not taking energy from first row & col, which are padding
      z_loc = z_index * self.z_pixel_len
      z_loc_interstitial = (z_index-.5) * self.z_pixel_len
      z_spacing = self.surface.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude)
      th_spacing = self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude)
      col_sqrtg = z_spacing*th_spacing
      col_index_raise_and_sqrtg = self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude)/self.surface.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)
      col_A_th = self.surface.A_theta(z=z_loc_interstitial, amplitude=amplitude)
      psi_col = lattice[i]
      psi_squared_column = psi_squared[i]
      #TODO could do *sqrtg at the end to whole column, if covariant laplacian returned value/sqrt_g
      energy_col = (self.alpha*sum(psi_squared_column)+self.u/2*sum(psi_squared_column**2)) *col_sqrtg
      dz_col = dz[i]/z_spacing #dz does contain /z_pixel_len, but not sqrtgz adjustment
      #TODO likelyt problem part
      energy_col += self.C*sum(self.squared(dz_col))*col_sqrtg #TODO check this squares elementwise, then sums
      dth_col = dth[i]#/th_spacing <- this is the index raise?
      #C|dth Psi(x)|^2 part of energy density
      #TODO both gradient parts causing problems
      energy_col += self.C*sum(self.squared(dth_col))*col_index_raise_and_sqrtg
      #-iCn(A_th* Psi* dth Psi(x)) and c.c. part of energy density
      energy_col += self.C*self.n*2*sum((col_A_th.conjugate()*psi_col.conjugate()*dth_col).imag)*col_index_raise_and_sqrtg
      # Cn^2|A_th Psi(x)|^2  part of energy density
      energy_col += self.Cnsquared*sum(self.squared(psi_col))*self.squared(col_A_th)*col_index_raise_and_sqrtg
      #different in this subclass:
      cell_dims = (self.z_pixel_len * self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude), self.th_pixel_len * self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude))
      energy_col += self.th_len*self.get_background_energy(cell_dims)*col_sqrtg
      energy += energy_col
    #print("amplitude ", amplitude, "would get field energy", energy*self.z_pixel_len*self.th_pixel_len)
    return energy*self.z_pixel_len*self.th_pixel_len
 

  def surface_field_energy(self, amplitude, reference_amplitude, lattice, psi_squared, dz, dth):
    """calculates energy on proposed amplitude change"""
    energy=0
    for z_index in range(0,self.z_len):
      z_loc = z_index * self.z_pixel_len
      z_loc_interstitial = (z_index-.5) * self.z_pixel_len
      z_spacing = self.surface.sqrt_g_z(z=z_loc, amplitude=reference_amplitude) 
      th_spacing = self.surface.sqrt_g_theta(z=z_loc, amplitude=reference_amplitude)
      col_sqrtg = self.surface.sqrt_g_z(z=z_loc, amplitude=reference_amplitude)*self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude)
      col_index_raise_and_sqrtg = col_sqrtg / th_spacing**2
      col_A_th = self.surface.A_theta(z=z_loc_interstitial, amplitude=amplitude)
      psi_col = lattice[z_index]
      psi_squared_column = psi_squared[z_index]
      #TODO could do *sqrtg at the end to whole column, if covariant laplacian returned value/sqrt_g
      cell_dims = (self.z_pixel_len * self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude), self.th_pixel_len * self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude))
      energy_col = (self.alpha*sum(psi_squared_column)+self.u/2*sum(psi_squared_column**2)) *col_sqrtg
      dz_col = dz[z_index]/z_spacing
      energy_col += self.C*sum(self.squared(dz_col))*col_sqrtg #TODO check this squares elementwise, then sums
      dth_col = dth[z_index]#/self.surface.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude) <- this is the index raise?
      #C|dth Psi(x)|^2 part of energy density
      energy_col += self.C*sum(self.squared(dth_col))*col_index_raise_and_sqrtg
      #-iCn(A_th* Psi* dth Psi(x)) and c.c. part of energy density
      energy_col += self.C*self.n*2*sum((col_A_th.conjugate()*psi_col.conjugate()*dth_col).imag)*col_index_raise_and_sqrtg
      # Cn^2|A_th Psi(x)|^2  part of energy density
      energy_col += self.Cnsquared*sum(self.squared(psi_col))*self.squared(col_A_th)*col_index_raise_and_sqrtg
      #0th order rescale:
      #also add bacground energy from implicit fluctuations for each col
      # that's number of cells in column*background energy (per area) of each, given its dimensions(*area of each cell, sqrtg part, base part done in return)
      energy_col += self.th_len*self.get_background_energy(cell_dims)*col_sqrtg
      energy += energy_col
    #print("amplitude ", amplitude, "would get field energy", energy*self.z_pixel_len*self.th_pixel_len)
    return energy*self.z_pixel_len*self.th_pixel_len

if __name__ == "__main__":
  lattice = Lattice(amplitude=0, wavenumber=.2, radius=1, gamma=1, kappa=0, intrinsic_curvature=0,
                  alpha=-1, u=1, C=1, n=6, temperature=.01, 
                    dims=(50,25))
  n_steps=10000
  n_sub_steps=lattice.z_len*lattice.th_len
  lattice.run(n_steps, n_sub_steps)
  print(lattice.lattice)
  print(lattice.lattice_acceptance_counter)
  lattice.plot()

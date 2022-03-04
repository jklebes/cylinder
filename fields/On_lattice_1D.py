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
except ModuleNotFoundError:
  #differnt if this file is run ( __name__=="__main__" )
  #then relatve import:
  import system_cylinder as system_cylinder
import metropolisengine

class Lattice1D():
  def __init__(self, amplitude, wavenumber, radius, gamma, kappa, intrinsic_curvature, alpha, N,
                u, C, n, temperature, temperature_final, dims = 100, n_substeps=None, fieldsteps_per_ampstep=1):
    #experiment variables
    self.wavenumber = wavenumber
    self.radius=radius
    self.gamma=gamma
    self.kappa=kappa
    self.initial_amplitude = amplitude
    self.intrinsic_curvature = intrinsic_curvature
    self.initial_amplitude = amplitude
    self.alpha = alpha
    self.N = N
    self.u = u
    self.C= C
    self.n= n
    self.Cnsquared = self.C*self.n**2
    self.temperature = temperature
    self.fieldsteps_per_ampstep = fieldsteps_per_ampstep
    #don't use literally the z-direction number of lattice points provided, but scale with wavenumber
    cylinder_len_z = 2*math.pi / self.wavenumber # = wavelength lambda
    # so that z-direction pixel length is the same and results are comparable with different wavenumber
    self.z_len = int(round(dims/self.wavenumber))
    if n_substeps is None:
      self.n_substeps =self.z_len
    else:
      self.n_substeps = n_substeps
    self.z_pixel_len = cylinder_len_z /self.z_len # length divided py number of pixels
    self.amplitude = self.initial_amplitude
    #set up metropolis engine coupled to bare cylinder system
    self.surface = system_cylinder.Cylinder(wavenumber=self.wavenumber, radius=self.radius, gamma=self.gamma, kappa=self.kappa, intrinsic_curvature=self.intrinsic_curvature)
    #Psi at each point 
    self.lattice = np.zeros(self.z_len)
    #running avg at each point
    self.avg_lattice = np.zeros(self.z_len)
    #values also saved for convenience
    self.psi_squared = np.zeros(self.z_len) 
    self.dz = np.zeros(self.z_len)
    #establish what fraction of each column is defect
    self.defect_area = np.zeros(self.z_len)
    fill_defect_area(self.N)
    print("defect area", self.defect_area)

    self.avg_amplitude_profile=np.zeros(self.z_len)
    #self.avg_amplitude_history()

    self.random_initialize()
    #print("initialized\n", self.lattice)

    #simple option with no influence from field energy to surface shape
    energy_fct_surface_term = lambda real_params, complex_params : self.surface.calc_surface_energy(*real_params) 
    # advanced option coupling surface fluctutations to the energy it would cause on field
    energy_fct_field_term = lambda real_params, complex_params : self.surface_field_energy(*real_params, self.amplitude, self.lattice, self.psi_squared, self.dz, self.dth)
    energy_fct_by_params_group = { "real": {"surface": energy_fct_surface_term, 
                                            "field": energy_fct_field_term}, 
                                  "all":{ "surface":energy_fct_surface_term,
                                        "field": energy_fct_field_term}}
    self.me = metropolisengine.MetropolisEngine(energy_functions = energy_fct_by_params_group,  initial_complex_params=None, initial_real_params = [float(self.initial_amplitude)], 
                                 covariance_matrix_complex=None, sampling_width=.005, temp=self.temperature
                                 ,complex_sample_method=None)
    self.me.set_reject_condition(lambda real_params, complex_params : abs(real_params[0])>.8 )  
    self.lattice_acceptance_counter = 0
    self.step_counter = 0
    self.amplitude_average= 0
    self.field_average = np.zeros((self.z_len))
    self.field_profile_history=[]
    self.field_abs_profile_history=[]

    #dynamic step size
    self.acceptance_counter=0
    self.step_counter=0
    self.bump_step_counter=0
    self.target_acceptance = .5 #TODO: hard code as fct of nuber of parameter space dims
    self.ppf_alpha = -1 * scipy.stats.norm.ppf(self.target_acceptance / 2)
    self.m = 1
    self.ratio = ( #  a constant - no need to recalculate 
        (1 - (1 / self.m)) * math.sqrt(2 * math.pi) * math.exp(self.ppf_alpha ** 2 / 2) / 2 * self.ppf_alpha + 1 / (
        self.m * self.target_acceptance * (1 - self.target_acceptance)))
    self.sampling_width=.005
  
  def fill_defect_area(self, N):
    """fills a 1D array that tells us how much area is taken up by a circular defect 
    on the widest point and one on the nattowest point
    from 1 defect - multiply by 2N to get area occupied by defects on N-rotating field"""
    if self.alpha <=0:
      return np.zeros(self.z_len)
    defect_radius = .5*self.sqrt(self.c/(2*abs(self.alpha)))#correlation length in ordered state, a guess
    wide_point = math.floor(self.z_len/4) 
    narrow_point = math.floor(3*self.z_len/4)#could be other way around at negative amplitude, doesnt matter
    start=narrow_point - defect_radius
    start2 = wide_point-defect_radius
    for z_index in (narrow_point-defect_radius, narrow_point+defect_radius+1):
      chord_location = abs(narrow_point-z_index)/defect_radius
      circle_chord = 2*math.sin(math.acos(chord_location))#as at circle_location from the center of a circle of radius 1
      assert(circle_chord >=0 and circle_chord <=2)
      self.defect_area[start+z_index] = circle_chord*defect_radius
      self.defect_area[start2+z_index] = circle_chord*defect_radius


  def measure_avgs(self):
    #update a running avg of |a|
    self.amplitude_average *= self.step_counter/float(self.step_counter+1)
    self.amplitude_average += abs(self.amplitude) / float(self.step_counter+1)
    assert(self.amplitude_average <1)
    divisor = float(self.step_counter+1)
    #update a running avg of |Psi|(z) (1D array; avgd in theta direction)
    #update each cell in avg_lattice with running time avg
    self.avg_lattice *= self.step_counter/divisor 
    self.avg_lattice += self.lattice / divisor
    #now collecting whole profile history in measure
    #only equlibrated part is averaged and recorded later
    self.step_counter+=1


  def random_initialize(self):
    #assuming that this is called on amplitude=0 cylinders
    for z_index in range(self.z_len):
      #fill lattice
      value = random.uniform(0,.1)
      self.lattice[z_index] = value
      #fill stored energy density, 
      self.psi_squared[z_index] = value**2
    #fill derivatives
    for z_index in range(self.z_len):
      value= self.lattice[z_index]
      left_value_z = self.lattice[z_index-1]
      self.dz[z_index]  = value-left_value_z
    self.dz/= self.z_pixel_len
  

  def surface_field_energy(self, amplitude, lattice, psi_squared, dz, dth):
    """calculates energy on proposed amplitude change or whole lattice change"""
    energy=0
    for z_index in range(0,self.z_len):
      z_loc = z_index * self.z_pixel_len
      z_loc_interstitial = (z_index-.5) * self.z_pixel_len
      sqrt_g_z = self.surface.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude)
      col_A_th = self.surface.A_theta(z=z_loc, amplitude=amplitude)
      radius_factor = self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude) 
      index_raise = 1/radius_factor**2 
      circumference = 2*math.pi*radius_factor*self.radius
      psi_col = lattice[z_index]
      psi_squared_column = psi_squared[z_index]
      #A_th and  th gradient in rotation 
      th_grad_const = self.C*(self.n * abs(col_A_th)*psi_col - self.N/circumference)**2*index_raise
      #quadratic potential
      energy_col = (self.alpha + th_grad_const)*psi_squared_col
      #z gradient
      energy_col += self.C*(self.dz[z_index]/sqrt_g_z)**2
      #quartic potential
      energy_col += self.u/2 * psi_squared_col**2
      energy_col *= sqrt_g_z
      defect_fraction = min(1, self.defect_area[z]*2*self.N/circumference) #as fraction of circumeference
      energy_col *= circumference*(1-defect_fraction)
      #for a fraction of size defectfraction of the annulus, the field is instead 0
      #energy_col_defect = 0
      #energy_col+= energy_col_defect*circumference*self.defect_fraction
      #and +background energy of reference ordered field
      energy_col += self.reference_field_energy * circumference
      #and +1/2 kBT per degree of freedom for fluctuations
      energy_col += .5*self.temperature
      energy += energy_col
    return energy*self.z_pixel_len


  def step_lattice_all(self, amplitude):
    addition = random.gauss(0,self.sampling_width)
    lattice_addition = np.full(self.z_len, addition)
    new_lattice=np.array([max(0,x+lattice_addition) for x in self.lattice])
    new_psi_squared = np.square(new_lattice) #np.square is elementwise square of complex numbers, z^2 not zz*
    new_energy=self.surface_field_energy(amplitude,  new_lattice, new_psi_squared, self.dz, self.dth) #dz,dth differnces nuchanged by uniform addition
    old_energy=self.energy
    accept= self.me.metropolis_decision(old_energy, new_energy)#TODO check order
    if accept:
       self.lattice=new_lattice
       self.psi_squared=new_psi_squared
       #dz, dth do not change
       self.energy=new_energy
       self.me.energy["field"] = new_energy
    self.update_sigma(accept)


  def run_fixed_amplitude(self, n_steps, n_sub_steps):
    for i in range(n_steps):
      for n in self.fieldsteps_per_ampstep:
        self.energy=self.surface_field_energy(self.amplitude, self.lattice, self.psi_squared, self.dz, self.dth)
        self.step_lattice_all(self.amplitude)
        for j in range(n_sub_steps):
          self.step_lattice(self.amplitude)
      field_energy=self.surface_field_energy(self.amplitude, self.lattice, self.psi_squared, self.dz, self.dth)
      self.me.energy["field"] = field_energy
      self.me.measure_real_system()


  def run(self, n_steps, n_sub_steps):
    for i in range(n_steps):
      self.measure() # add to lists
      surface_accepted = self.me.step_real_group()
      if surface_accepted:
        self.amplitude = self.me.real_params[0]
        #update rescale-related values as applicable:
        self.update_rescale_params(self.amplitude)
      #maybe reset self.energy
      self.energy=self.surface_field_energy(self.amplitude, self.lattice, self.psi_squared, self.dz, self.dth)
      #lattice step
      #step whole lattic at once
      for n in self.fieldsteps_per_ampstep:
        self.step_row_rotate(self.amplitude, self.z_len//4)
        for j in range(n_sub_steps):
          #self.step_lattice_bump(self.amplitude)
          self.step_lattice(self.amplitude)
      field_energy=self.surface_field_energy(self.amplitude, self.lattice, self.psi_squared, self.dz, self.dth)
      #self.field_energy_time_series.append(field_energy)
      self.me.energy["field"] = field_energy
      self.me.measure_real_system()


  def record_avgs(self):
    for z_index in range(self.z_len):
      col = self.lattice[z_index]
      self.avg_amplitude_profile[z_index]+=sum(abs(col)) / len(col)

  def step_lattice(self, amplitude):
    #choose a location
    index_z = random.randrange(-1,(self.z_len-1) )
    self.step_lattice_loc(amplitude, index_z)


  def step_lattice_loc(self, amplitude, index_z, index_th):
    z_loc = self.z_pixel_len * index_z 
    z_loc_interstitial = self.z_pixel_len * (index_z -.5)
    z_loc_neighbor_interstitial = self.z_pixel_len * (index_z +.5)
    #properties of the surface at this point
    A_th= self.surface.A_theta(z=z_loc, amplitude=amplitude)  
    radius_factor = self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude) 
    index_raise = 1/radius_factor**2 
    A_th_neighbor= self.surface.A_theta(z=z_loc_neighbor_interstitial, amplitude=amplitude)  
    neighbor_index_raise = 1/self.surface.g_theta(z=z_loc_neighbor_interstitial, amplitude=amplitude) 
    
    sqrt_g = (self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude)*self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude))
    sqrt_g_neighbor = (self.surface.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)*self.surface.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude))
    stepsize=self.sampling_width#choose something order of magnitude of predicted width of distribution
    value=self.lattice[index_z]
    new_value=random.gauss(value,stepsize)
    while new_value < 0:
      new_value = random.gauss(value,stepsize)
    new_psi_squared = new_value**2
    #energy difference of switching the pixel:
    th_grad_const = self.C*(self.n * abs(col_A_th)*psi_col - self.N/circumference)**2*index_raise
    old_psi_squared = self.psi_squared[index_z]
    old_pixel_energy_density = (self.alpha +th_grad_const)*old_psi_squared + self.u/2*old_psi_squared**2
    new_pixel_energy_density = (self.alpha +th_grad_const)* new_psi_squared + self.u/2*new_psi_squared**2
    diff_energy = new_pixel_energy_density -  old_pixel_energy_density 
    #techncally the diff is *sqrt(g), then
    #normailzed by sqrt(g) again in the end
    
    #effect on derivative at the location
    left_value_z = self.lattice[index_z-1, index_th]
    new_derivative_z = (new_value-left_value_z)/self.z_pixel_len
    diff_derivative_z = (new_derivative_z**2 - self.dz[index_z, index_th]**2)/self.surface.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude)**2
    
    diff_energy += self.C*( diff_derivative_z + diff_derivative_th )

    #effect on derivative at neighboring locations i_z+1, i_th +1
    right_value_z = self.lattice[index_z+1, index_th]
    new_neighbor_derivative_z = (right_value_z-new_value)/self.z_pixel_len
    diff_neighbor_derivative_z = (new_neighbor_derivative_z**2 - self.dz[index_z+1, index_th]**2 )/self.surface.sqrt_g_z(z=z_loc_neighbor_interstitial, amplitude=amplitude)**2
    
    diff_energy += self.C*( diff_neighbor_derivative_z + diff_neighbor_derivative_th )
     
    circumference = 2*math.pi*radius_factor*self.radius
    diff_energy*= self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude)*self.z_pixel_len #line element
    defect_fraction = min(1, self.defect_area[z]*2*self.N/circumference) #as fraction of circumeference
    diff_energy *= circumference*(1-defect_fraction) #decision matters less if some of this location is defect, fixed to psi=0
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
      
  def update_sigma(self, accept):
    self.step_counter+=1
    step_number_factor = max((self.step_counter / self.m, 200))
    steplength_c = self.sampling_width * self.ratio
    if accept:
      self.sampling_width += steplength_c * (1 - self.target_acceptance) / step_number_factor
    else:
      self.sampling_width -= steplength_c * self.target_acceptance / step_number_factor
    assert (self.sampling_width) > 0
    #print(self.step_counter)
    #print("acceptance" , self.acceptance_counter, self.acceptance_counter/self.step_counter,self.sampling_width)

  def plot(self):
    zs = [i for i in range(self.z_len)]
    #draw two sin curves for cylinder shape
    amplitude_scaling = 10
    cylinder_top = [-15+amplitude_scaling*self.amplitude*math.sin(z*2*math.pi/self.z_len) for z in zs]
    cylinder_bottom = [-35-amplitude_scaling*self.amplitude*math.sin(z*2*math.pi/self.z_len) for z in zs]
    plt.plot(zs, cylinder_top, color='g', linewidth=10)
    plt.plot(zs, cylinder_bottom, color='g', linewidth=10)
    #plot binary lattice field
    plt.imshow(abs(self.lattice.transpose()))

    plt.show()
    plt.imshow((self.lattice.transpose()).real)

    plt.show()
    plt.plot(zs, self.avg_amplitude_profile)
    plt.show()

if __name__ == "__main__":
  lattice = Lattice(amplitude=0, wavenumber=.2, radius=1, gamma=1, kappa=0, intrinsic_curvature=0,
                  alpha=-1, u=1, C=1, n=6, temperature=.01, temperature_lattice = .01,
                    dims=(50,25))
  n_steps=10000
  n_sub_steps=lattice.z_len*lattice.th_len
  lattice.run(n_steps, n_sub_steps)
  print(lattice.lattice)
  print(lattice.amplitude)
  print(lattice.lattice_acceptance_counter)
  lattice.plot()

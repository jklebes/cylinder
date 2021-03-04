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

class Lattice():
  def __init__(self, amplitude, wavenumber, radius, gamma, kappa, intrinsic_curvature, alpha,
                u, C, n, temperature, temperature_lattice, dims = (100,50), n_substeps=None ):
    #experiment variables
    self.wavenumber = wavenumber
    self.radius=radius
    self.gamma=gamma
    self.kappa=kappa
    self.initial_amplitude = amplitude
    self.intrinsic_curvature = intrinsic_curvature
    self.initial_amplitude = amplitude
    self.alpha = alpha
    self.u = u
    self.C= C
    self.n= n
    self.Cnsquared = self.C*self.n**2
    self.temperature = temperature
    self.temperature_lattice = temperature_lattice
    self.temperature_factor = self.temperature_lattice/self.temperature#to get the desired lattice temperature divide deltaE by this factor in addition to temperature included in metropolis step
    assert(1/self.temperature*1/self.temperature_factor == 1/self.temperature_lattice)
    #lattice characteristics
    #don't use literally the z-direction number of lattice points provided, but scale with wavenumber
    cylinder_len_z = 2*math.pi / self.wavenumber # = wavelength lambda
    cylinder_radius_th = 2*math.pi*self.radius # circumference - in len units, not radians
    # so that z-direction pixel length is the same and results are comparable with different wavenumber
    self.z_len = int(round(dims[0]/self.wavenumber))
    self.th_len =dims[1]
    if n_substeps is None:
      self.n_substeps =self.z_len*self.th_len
    else:
      self.n_substeps = n_substeps
    self.z_pixel_len = cylinder_len_z /self.z_len # length divided py number of pixels
    #assert(math.isclose(self.z_pixel_len,2*math.pi/float(dims[0]))) #should be same as for wavenumber 1, length 2pi, dims[0] pixels
    self.th_pixel_len = cylinder_radius_th /self.th_len
    self.amplitude = self.initial_amplitude
    #set up metropolis engine coupled to bare cylinder system
    self.surface = system_cylinder.Cylinder(wavenumber=self.wavenumber, radius=self.radius, gamma=self.gamma, kappa=self.kappa, intrinsic_curvature=self.intrinsic_curvature)
    #Psi at each point 
    self.lattice = np.zeros((self.z_len, self.th_len), dtype=complex)
    #running avg at each point
    self.avg_lattice = np.zeros((self.z_len, self.th_len), dtype=complex)
    #values also saved for convenience
    self.psi_squared = np.zeros((self.z_len, self.th_len)) 
    #self.dz_squared = np.zeros((self.z_len, self.th_len))
    self.dz = np.zeros((self.z_len, self.th_len), dtype=complex)
    self.dth = np.zeros((self.z_len, self.th_len), dtype=complex)
    # note: these are left derivatives:
    # derivative stored at i refers to difference between i-1, i positions in corrseponding 
    # grid of lattice values

    self.avg_amplitude_profile=np.zeros((self.z_len))

    self.random_initialize()
    print("initialized\n", self.lattice)

    #simple option with no influence from field energy to surface shape
    energy_fct_surface_term = lambda real_params, complex_params : self.surface.calc_surface_energy(*real_params) 
    # advanced option coupling surface fluctutations to the energy it would cause on field
    energy_fct_field_term = lambda real_params, complex_params : self.surface_field_energy(*real_params)
    energy_fct_by_params_group = { "real": {"surface": energy_fct_surface_term, 
                                            "field": energy_fct_field_term}, 
                                  "all":{ "surface":energy_fct_surface_term,
                                        "field": energy_fct_field_term}}
    self.me = metropolisengine.MetropolisEngine(energy_functions = energy_fct_by_params_group,  initial_complex_params=None, initial_real_params = [float(self.initial_amplitude)], 
                                 covariance_matrix_complex=None, sampling_width=.05, temp=self.temperature
                                 ,complex_sample_method=None)
    self.me.set_reject_condition(lambda real_params, complex_params : abs(real_params[0])>.8 )  
    self.lattice_acceptance_counter = 0
    self.step_counter = 0
    self.amplitude_average= 0
    self.field_average = np.zeros((self.z_len))

    #dynamic step size
    self.acceptance_counter=0
    self.step_counter=0
    self.target_acceptance = .5 #TODO: hard code as fct of nuber of parameter space dims
    self.ppf_alpha = -1 * scipy.stats.norm.ppf(self.target_acceptance / 2)
    self.m = 1
    self.ratio = ( #  a constant - no need to recalculate 
        (1 - (1 / self.m)) * math.sqrt(2 * math.pi) * math.exp(self.ppf_alpha ** 2 / 2) / 2 * self.ppf_alpha + 1 / (
        self.m * self.target_acceptance * (1 - self.target_acceptance)))
    self.sampling_width=.01
  
  def squared(self, c):
    return abs(c*c.conjugate())

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
    if self.amplitude>=0:
      avg_psi = np.array([sum(abs(col))/len(col) for col in self.lattice])
    else: #add mirrored version so that we get wde mart matched to wide part etc
      avg_psi=np.array([sum(abs(col))/len(col) for col in self.lattice[::-1]])
    #if a<0 flip the list
    self.field_average *= self.step_counter/divisor 
    self.field_average += avg_psi / divisor 
    #print(self.amplitude_average,self.field_average[1] )
    
    self.step_counter+=1

  def random_initialize(self):
    #assuming that this is called on amplitude=0 cylinders
    for z_index in range(self.z_len):
      for th_index in range(self.th_len):
        #fill lattice
        value = cmath.rect(random.uniform(0,1), random.uniform(0, 2*math.pi))
        self.lattice[z_index, th_index] = value
        #fill stored energy density, 
        self.psi_squared[z_index, th_index] = self.squared(value) 
    #fill derivatives
    for z_index in range(self.z_len):
      for th_index in range(self.th_len):    
        #dz
        value= self.lattice[z_index, th_index]
        left_value_z = self.lattice[z_index-1, th_index]
        #just a left (backwards) derivative
        self.dz[z_index, th_index]  = value-left_value_z
        
        #dth
        #same, just a left derivative is saved.  (re-) calculating and different a 
        # involves multiplying these by A_th(i-1/2) in various combinations
        left_value_th = self.lattice[z_index, th_index-1]
        self.dth[z_index, th_index]  =   value-left_value_th
    self.dz/= self.z_pixel_len
    self.dth/= self.th_pixel_len

  def surface_field_energy(self, amplitude):
    """calculates energy on proposed amplitude change"""
    energy=0
    for z_index in range(0,self.z_len):
      z_loc = z_index * self.z_pixel_len
      z_loc_interstitial = (z_index-.5) * self.z_pixel_len
      col_sqrtg = self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude)*self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude)
      col_index_raise_and_sqrtg = self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude)/self.surface.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)
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

      energy += energy_col
    #print("amplitude ", amplitude, "would get field energy", energy*self.z_pixel_len*self.th_pixel_len)
    return energy*self.z_pixel_len*self.th_pixel_len

  def run(self, n_steps, n_sub_steps):
    for i in range(n_steps):
      #print(i, self.amplitude)
      #metropolis step shape
      self.measure_avgs() #running avgs amplitude, field profile
      #self.measure() # add to lists
      surface_accepted = self.me.step_real_group()
      if surface_accepted:
        self.amplitude = self.me.real_params[0]
        #maybe reset self.energy
      #lattice step
      for i in range(n_sub_steps):
        self.step_lattice(self.amplitude)
      #pass on info on change in field energy to metropolis engine, when field changed but amplitude didn;t
      self.me.energy["field"] = self.surface_field_energy(self.amplitude)
      self.me.measure()
  
  def plot_save(self, exp_dir, title):
    #plot and save field energy history?
    #plot and save average field profile <|Psi|>
    field_avg=self.field_average
    df = pd.DataFrame(data=field_avg)
    df.to_csv(os.path.join(exp_dir, title + "_profile.csv"))
    plt.plot([z for z in range(len(field_avg))], field_avg)
    plt.savefig(os.path.join(exp_dir,title+"_profile.png"))
    plt.close()
    #also dump final snapshot - df of complex values
    df_snapshot = pd.DataFrame(data=self.lattice)
    df_snapshot.to_csv(os.path.join(exp_dir, title + "_snapshot.csv"))
    df_snapshot = pd.DataFrame(data=self.avg_lattice)
    df_snapshot.to_csv(os.path.join(exp_dir, title + "_avglattice.csv"))

  def record_avgs(self):
    for z_index in range(self.z_len):
      col = self.lattice[z_index]
      self.avg_amplitude_profile[z_index]+=sum(abs(col)) / len(col)

  def step_lattice(self, amplitude):
    #choose a location
    index_z, index_th = (random.randrange(-1,(self.z_len-1)), random.randrange(-1,(self.th_len-1)))
    self.step_lattice_loc(amplitude, index_z, index_th)

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

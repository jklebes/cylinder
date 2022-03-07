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
  The base class for complex field Psi obeying Ginzburg-Landau theory.
  Rescaled versions build on this.
  Works together with a surface shape and a metropolis engine.
  """

  def __init__(self, alpha, u, C, n, dims, wavenumber, radius=1, n_substeps=None):
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
    self.psi = np.zeros((self.z_len, self.th_len), dtype=complex)
    #values also saved for convenience
    self.psi_squared = np.zeros((self.z_len, self.th_len)) 
    #self.dz_squared = np.zeros((self.z_len, self.th_len))
    self.dz = np.zeros((self.z_len, self.th_len), dtype=complex)
    self.dth = np.zeros((self.z_len, self.th_len), dtype=complex)
    # note: these are left derivatives:
    # derivative stored at i refers to difference between i-1, i positions in corrseponding 
    # grid of lattice values

    #locals running avgs #TODO keep?
    self.avg_lattice = np.zeros((self.z_len, self.th_len), dtype=complex)
    self.avg_amplitude_profile=np.zeros((self.z_len))
    #self.avg_amplitude_history()

    self.random_initialize()
    print("initialized")

  def squared(self, c):
    """
    a local utility
    """
    return abs(c*c.conjugate()) #abs is needed, does complex->float

  def get_profiles_df(self):
    #make and return a dataframe of <Psi>(z), <|Psi|>(z) (theta direciton avergaes, mirror corrected) over time
    #initially we have
    #self.field_profile_history, self.field_abs_profile_history,
    # lists of np arrays over time
    df_profile = pd.DataFrame(np.array(self.field_profile_history)/self.th_len) #remember to divide by 50 in the end
    df_abs_profile = pd.DataFrame(np.array(self.field_abs_profile_history)/self.th_len)
    dfs = (df_profile, df_abs_profile)
    #print(dfs)
    return dfs

  def measure(self):
    #if self.amplitude>=0:
    avg_psi = np.array([sum(col) for col in self.psi])
    #else: #add mirrored version so that we get wde mart matched to wide part etc
    #  avg_psi=np.array([sum(col) for col in self.psi[::-1]])
    self.field_profile_history.append(avg_psi)
    #if self.amplitude>=0:
    avg_abs_psi = np.array([sum(abs(col)) for col in self.psi])
    #else: #add mirrored version so that we get wde mart matched to wide part etc
    #  avg_abs_psi=np.array([sum(abs(col)) for col in self.psi[::-1]])
    self.field_abs_profile_history.append(avg_abs_psi)
    #still need to divide the chole thing by column length -most likely 50


  def random_initialize(self):
    for z_index in range(self.z_len):
      for th_index in range(self.th_len):
        #fill lattice
        value = cmath.rect(random.uniform(0,.1), random.uniform(0, 2*math.pi))
        self.psi[z_index, th_index] = value
        #fill stored energy density, 
        self.psi_squared[z_index, th_index] = self.squared(value) 
    #fill derivatives
    for z_index in range(self.z_len):
      for th_index in range(self.th_len):    
        #dz
        value= self.psi[z_index, th_index]
        left_value_z = self.psi[z_index-1, th_index]
        #just a left (backwards) derivative
        self.dz[z_index, th_index]  = value-left_value_z
        #dth
        #same, just a left derivative is saved.  (re-) calculating and different a 
        # involves multiplying these by A_th(i-1/2) in various combinations
        left_value_th = self.psi[z_index, th_index-1]
        self.dth[z_index, th_index]  =   value-left_value_th
    #TODO line element is wrong for nonflat!
    self.dz/= self.z_pixel_len
    self.dth/= self.th_pixel_len
  

  def total_field_energy(self, shape):
    """calculates energy on proposed amplitude change or whole lattice change"""
    energy=0
    for z_index in range(0,self.z_len):
      z_loc = z_index * self.z_pixel_len
      z_loc_interstitial = (z_index-.5) * self.z_pixel_len
      z_spacing = shape.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude)
      th_spacing = shape.sqrt_g_theta(z=z_loc, amplitude=amplitude)
      col_sqrtg = z_spacing*th_spacing
      col_index_raise_and_sqrtg = shape.sqrt_g_z(z=z_loc, amplitude=amplitude)/shape.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)
      col_A_th = shape.A_theta(z=z_loc_interstitial, amplitude=amplitude)
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

  """
  TODO move to run file
  def run_fixed_amplitude(self, n_steps, n_sub_steps):
    for i in range(n_steps):
      self.measure_avgs() #running avgs amplitude, field profile 
      self.measure() # add to lists
      for n in self.fieldsteps_per_ampstep:
        self.energy=shape_field_energy(self.amplitude, self.amplitude, self.psi, self.psi_squared, self.dz, self.dth)
        self.step_lattice_all(self.amplitude)
        for j in range(n_sub_steps):
          self.step_lattice(self.amplitude)
      field_energy=shape_field_energy(self.amplitude, self.amplitude, self.psi, self.psi_squared, self.dz, self.dth)
      self.me.energy["field"] = field_energy
      self.me.measure_real_system()

  def run(self, n_steps, n_sub_steps):
    n_sub_steps = 50
    for i in range(500):
      for j in range(n_sub_steps):
        self.step_lattice(self.amplitude)
    for i in range(n_steps):
      #print(i, self.amplitude)
      #metropolis step shape
      self.measure_avgs() #running avgs amplitude, field profile 
      #TODO the problem is that this is over all time, not starting from equilibration
      #solution: record profile in history: 
      self.measure() # add to lists
      surface_accepted = self.me.step_real_group()
      if surface_accepted:
        self.amplitude = self.me.real_params[0]
        #update rescale-related values as applicable:
        self.update_rescale_params(self.amplitude)
      #maybe reset self.energy
      self.energy=shape_field_energy(self.amplitude, self.amplitude, self.psi, self.psi_squared, self.dz, self.dth)
      #lattice step
      #step whole lattic at once
      for n in self.fieldsteps_per_ampstep:
        for j in range(n_sub_steps):
          #self.step_lattice_bump(self.amplitude)
          self.step_lattice(self.amplitude)
      #for j in range(self.z_len):
      #  self.step_row_rotate(self.amplitude, maxrows=self.z_len//4)
      #pass on info on change in field energy to metropolis engine, when field changed but amplitude didn;t
      field_energy=shape_field_energy(self.amplitude, self.amplitude, self.psi, self.psi_squared, self.dz, self.dth)
      #self.field_energy_time_series.append(field_energy)
      self.me.energy["field"] = field_energy
      self.me.measure_real_system()
  """

  def update_rescale_params(self, amplitude):
    """
    hook for subclasses to update characteristics of each cell (background energy per area, 1st order corrections)
    that change with cell size- must be updated and remembered when amplitude changes
    """
    pass
  
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
    df_snapshot = pd.DataFrame(data=self.psi)
    df_snapshot.to_csv(os.path.join(exp_dir, title + "_snapshot.csv"))
    df_avglattice = pd.DataFrame(data=self.avg_lattice)
    df_avglattice.to_csv(os.path.join(exp_dir, title + "_avglattice.csv"))

  def record_avgs(self):
    for z_index in range(self.z_len):
      col = self.psi[z_index]
      self.avg_amplitude_profile[z_index]+=sum(abs(col)) / len(col)

  def step_lattice(self, shape, sampling_width, me):
    """
    A single location step - dims*n_substeps of these make up a sweep
    """
    #choose a location
    index_z, index_th = (random.randrange(-1,(self.z_len-1)), random.randrange(-1,(self.th_len-1)))
    self.step_lattice_loc(index_z, index_th, shape, sampling_width, me)
    #TODO output energy, make metropolis decision here

  def step_lattice_loc(self, index_z, index_th, shape, sampling_width, me):
    """
    Main energy calc of single location change
    """
    z_loc = self.z_pixel_len * index_z 
    z_loc_interstitial = self.z_pixel_len * (index_z -.5)
    z_loc_neighbor_interstitial = self.z_pixel_len * (index_z +.5)
    #properties of the surface at this point
    A_th= shape.A_theta(z=z_loc_interstitial, amplitude=amplitude)  #only needed at i-1/2 s
    index_raise = 1/shape.g_theta(z=z_loc_interstitial, amplitude=amplitude)  #only needed at i-1/2 s
    A_th_neighbor= shape.A_theta(z=z_loc_neighbor_interstitial, amplitude=amplitude)  #only needed at i-1/2 s
    neighbor_index_raise = 1/shape.g_theta(z=z_loc_neighbor_interstitial, amplitude=amplitude)  #only needed at i-1/2 s
    
    sqrt_g = (shape.sqrt_g_theta(z=z_loc, amplitude=amplitude)*shape.sqrt_g_z(z=z_loc, amplitude=amplitude))
    #TODO choose which one to use
    #sqrt_g_interstitial = (shape.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)*shape.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude))
    # random new value with magnitude similar to old value
    #TODO dynamic stepsize
    stepsize=sampling_width#choose something order of magnitude of predicted width of distribution
    #stepsize *= self.sqrt_g
    #print("sampling stepsize", stepsize)
    value=self.psi[index_z, index_th]
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
    left_value_z = self.psi[index_z-1, index_th]
    left_value_th = self.psi[index_z, index_th-1]
    new_derivative_z = (new_value-left_value_z)/self.z_pixel_len
    new_derivative_th = (new_value -left_value_th)/self.th_pixel_len
    #term |d_i psi|^2 in energy
    diff_derivative_z = (self.squared(new_derivative_z) - self.squared(self.dz[index_z, index_th]))/shape.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude)**2
    diff_derivative_th = (self.squared(new_derivative_th) - self.squared(self.dth[index_z, index_th]))*index_raise/shape.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)**2
    
    diff_energy += self.C*( diff_derivative_z + diff_derivative_th )

    #effect on derivative at neighboring locations i_z+1, i_th +1
    right_value_z = self.psi[index_z+1, index_th]
    right_value_th = self.psi[index_z, index_th+1]
    new_neighbor_derivative_z = (right_value_z-new_value)/self.z_pixel_len
    new_neighbor_derivative_th = (right_value_th-new_value)/self.th_pixel_len
    diff_neighbor_derivative_z = (self.squared(new_neighbor_derivative_z) - self.squared(self.dz[index_z+1, index_th]) )/shape.sqrt_g_z(z=z_loc_neighbor_interstitial, amplitude=amplitude)**2
    diff_neighbor_derivative_th = (self.squared(new_neighbor_derivative_th) - self.squared(self.dth[index_z, index_th+1]))*neighbor_index_raise/shape.sqrt_g_theta(z=z_loc_neighbor_interstitial, amplitude=amplitude)**2
    
    diff_energy += self.C*( diff_neighbor_derivative_z + diff_neighbor_derivative_th )
    
    #diff in cross-term in(A_th* Psi* d_th Psi) and complex conjugate, at spatial location i_th-1/2
    # (stored at array location i)
    #i(A_th* Psi* d_th Psi)+c.c. = 2*Im(A_th*Psi* d_th Psi)
    old_cross_term = (A_th.conjugate()*self.psi[index_z,index_th].conjugate()*
                      self.dth[index_z, index_th]).imag #except for *index raise,*2nC done later to both
    new_interstitial_psi =  (new_value)# +left_value_th)/2
    new_cross_term = (A_th.conjugate()*new_interstitial_psi.conjugate()*
                      new_derivative_th).imag
    diff_energy += self.C*2*self.n*index_raise*(new_cross_term - old_cross_term) 

    #diff in cross-term in(A_th* Psi* d_th Psi) and complex conjugate, for neightbor at i+1
    old_neighbor_cross_term = (A_th_neighbor.conjugate()*self.psi[index_z,index_th+1].conjugate()*
                      self.dth[index_z, index_th+1]).imag #except for *index raise,*2nC done later to both
    new_neighbor_cross_term = (A_th_neighbor.conjugate()*self.psi[index_z,index_th+1].conjugate()*
                      new_neighbor_derivative_th).imag
    diff_energy += self.C*2*self.n*neighbor_index_raise*(new_neighbor_cross_term - old_neighbor_cross_term)

    diff_energy += self.Cnsquared*index_raise*self.squared(A_th)*(self.squared(new_value)-
                   self.squared(self.psi[index_z,index_th]))
    #The naive or 0th order scaling: \alpha, u by cell area; c by cell area but was /lx^2, /ly^2 earlier
    diff_energy*=sqrt_g
    diff_energy*=self.z_pixel_len*self.th_pixel_len #- included in renormalizing coefficients
    accept = me.metropolis_decision(0,diff_energy)
    if accept:
      #change stored value of pixel
      self.psi[index_z,index_th] = new_value
      #change stored values of dth, dz across its boundaries
      #self.energy += diff_energy
      #fill stored energy density, 
      self.psi_squared[index_z, index_th]  = new_psi_squared
      #dz
      self.dz[index_z, index_th]  = new_derivative_z
      self.dz[index_z+1, index_th]  = new_neighbor_derivative_z
      #dth
      self.dth[index_z, index_th] = new_derivative_th 
      self.dth[index_z, index_th+1] = new_neighbor_derivative_th
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

  
  
  ########################################  extras  ############################################
  ############## unused group sampling schemes that might be more efficient ####################

  def step_lattice_bump(self, amplitude, bumpdims=(math.pi/25, math.pi/25)):
    """TODO Unused sampling  proposal"""
    bump_center_z, bump_center_th = (random.randrange(-1,(self.z_len-1)), random.randrange(-1,(self.th_len-1)))
    addition = random.gauss(0,self.bump_sampling_width)+random.gauss(0,self.bump_sampling_width)*1j 
    # find sublattice affected
    exact_width_z = bumpdims[0]/(self.z_pixel_len * shape.sqrt_g_z(z=bump_center_z, amplitude=amplitude))#how many cells make up length math.pi/25, equivalent to one cell on flat surface
    bump_width_z = math.ceil(exact_width_z)#size of array needed to represent this
    exact_width_th =bumpdims[1]/(self.th_pixel_len * shape.sqrt_g_theta(z=bump_center_z, amplitude=amplitude))
    bump_width_th = math.ceil(exact_width_th)
    #goes from back because python  handles negative indices better than too large
    bump_start_z = (bump_center_z - bump_width_z//2)%self.z_len
    bump_start_th = (bump_center_th - bump_width_th//2)%self.th_len
    lattice_addition = self.bump_array(addition, exact_width_z, exact_width_th, bump_width_z, bump_width_th)
    #in case the bump wrapped around the cylinder more than once:
    # this was handled by bumparray still being <= lattice  dims
    #we never need its true width again, just the dimensions and location of the area affected
    # keep bump ends
    bump_width_z, bump_width_th = lattice_addition.shape
    #starts will also have changed because lattice_addtiion has (partial) halo of 0s in 
    # left / - direction (if it wraps around the full lattice return with a left halo of duplicated column)
    bump_end_z = bump_start_z + bump_width_z
    bump_end_th = bump_start_th + bump_width_th
    if bump_end_z < self.z_len and bump_end_th < self.th_len:
      sublattice = self.psi[bump_start_z:bump_end_z, bump_start_th : bump_end_th]
      old_psi_squared = self.psi_squared[bump_start_z:bump_end_z, bump_start_th : bump_end_th]
      old_dz = self.dz[bump_start_z:bump_end_z, bump_start_th : bump_end_th]
      old_dth = self.dth[bump_start_z:bump_end_z, bump_start_th : bump_end_th]
    else:
      extended_lattice = np.block([[self.psi, self.psi], [self.psi, self.psi]])
      extended_psi_squared = np.block([[self.psi_squared, self.psi_squared], [self.psi_squared, self.psi_squared]])
      extended_dz = np.block([[self.dz, self.dz], [self.dz, self.dz]])
      extended_dth = np.block([[self.dth, self.dth], [self.dth, self.dth]])
      sublattice = extended_lattice[bump_start_z:bump_end_z, bump_start_th : bump_end_th]
      old_psi_squared = extended_psi_squared[bump_start_z:bump_end_z, bump_start_th : bump_end_th]
      old_dz = extended_dz[bump_start_z:bump_end_z, bump_start_th : bump_end_th]
      old_dth = extended_dth[bump_start_z:bump_end_z, bump_start_th : bump_end_th]
    new_lattice=sublattice+lattice_addition
    new_psi_squared = np.multiply(new_lattice, new_lattice.conjugate()) #np.square is elementwise square of complex numbers, z^2 not zz*
    new_dz=np.zeros((bump_width_z, bump_width_th))
    new_dth=np.zeros((bump_width_z, bump_width_th))
    for i,z_index in enumerate(range(bump_start_z, bump_end_z)):
      for j,th_index in enumerate(range(bump_start_th, bump_end_th)): #leave 0s in 1st row & col   
        if i>0 and j>0:
          value= new_lattice[i, j]
          left_value_z = new_lattice[i-1, j]
          new_dz[i, j]  = value-left_value_z
          left_value_th = new_lattice[i, j-1]
          new_dth[i, j]  =   value-left_value_th
    new_dz/= self.z_pixel_len
    new_dth/= self.th_pixel_len
    new_energy=self.sublattice_field_energy(amplitude, bump_start_z, new_lattice, new_psi_squared, new_dz, new_dth) #dz,dth differnces nuchanged by uniform addition
    old_energy=self.sublattice_field_energy(amplitude, bump_start_z, sublattice, old_psi_squared, old_dz, old_dth) 
    accept= self.me.metropolis_decision(old_energy, new_energy)#TODO check order
    if accept:
      try:
        self.psi[bump_start_z:bump_end_z, bump_start_th: bump_end_th]=new_lattice
        self.psi_squared[bump_start_z:bump_end_z, bump_start_th: bump_end_th]=new_psi_squared
        self.dth[bump_start_z:bump_end_z, bump_start_th: bump_end_th]=new_dth
        self.dz [bump_start_z:bump_end_z, bump_start_th: bump_end_th] = new_dz
      except ValueError:
        if bump_end_z > self.z_len and bump_end_th > self.th_len:
          diff_z=self.z_len - bump_end_z
          diff_th=self.th_len - bump_end_th
          bump_end_th-=self.th_len
          bump_end_z -= self.z_len
          self.psi[bump_start_z:, bump_start_th: ]=new_lattice[:diff_z, :diff_th]
          self.psi_squared[bump_start_z:, bump_start_th:]=new_psi_squared[:diff_z,:diff_th]
          self.dth[bump_start_z:, bump_start_th: ]=new_dth[:diff_z,:diff_th]
          self.dz [bump_start_z:, bump_start_th: ] = new_dz[:diff_z,:diff_th]
          self.psi[:bump_end_z, bump_start_th: ]=new_lattice[diff_z:,:diff_th]
          self.psi_squared[:bump_end_z, bump_start_th: ]=new_psi_squared[diff_z:,:diff_th]
          self.dth[:bump_end_z, bump_start_th: ]=new_dth[diff_z:,:diff_th]
          self.dz [:bump_end_z, bump_start_th: ] = new_dz[diff_z:,:diff_th]
          self.psi[bump_start_z:, :bump_end_th ]=new_lattice[:diff_z,diff_th:]
          self.psi_squared[bump_start_z:, :bump_end_th ]=new_psi_squared[:diff_z,diff_th:]
          self.dth[bump_start_z:, :bump_end_th ]=new_dth[:diff_z,diff_th:]
          self.dz [bump_start_z:, :bump_end_th ] = new_dz[:diff_z,diff_th:]
          self.psi[bump_start_z:, bump_start_th: ]=new_lattice[:diff_z,:diff_th]
          self.psi_squared[bump_start_z:, bump_start_th: ]=new_psi_squared[:diff_z,:diff_th]
          self.dth[bump_start_z:, bump_start_th: ]=new_dth[:diff_z,:diff_th]
          self.dz [bump_start_z:, bump_start_th: ] = new_dz[:diff_z,:diff_th]
        elif bump_end_z > self.z_len :
          diff=self.z_len - bump_end_z
          bump_end_z -= self.z_len
          self.psi[bump_start_z:, bump_start_th: bump_end_th]=new_lattice[:diff, :]
          self.psi_squared[bump_start_z:, bump_start_th: bump_end_th]=new_psi_squared[:diff,:]
          self.dth[bump_start_z:, bump_start_th: bump_end_th]=new_dth[:diff,:]
          self.dz [bump_start_z:, bump_start_th: bump_end_th] = new_dz[:diff,:]
          self.psi[:bump_end_z, bump_start_th: bump_end_th]=new_lattice[diff:,:]
          self.psi_squared[:bump_end_z, bump_start_th: bump_end_th]=new_psi_squared[diff:,:]
          self.dth[:bump_end_z, bump_start_th: bump_end_th]=new_dth[diff:,:]
          self.dz [:bump_end_z, bump_start_th: bump_end_th] = new_dz[diff:,:]
        elif bump_end_th > self.th_len:
          diff=self.th_len - bump_end_th
          bump_end_th-=self.th_len
          self.psi[bump_start_z:bump_end_z, bump_start_th: ]=new_lattice[:, :diff]
          self.psi_squared[bump_start_z:bump_end_z, bump_start_th: ]=new_psi_squared[:, :diff]
          self.dth[bump_start_z:bump_end_z, bump_start_th: ]=new_dth[:, :diff]
          self.dz [bump_start_z:bump_end_z, bump_start_th: ] = new_dz[:, :diff]
          self.psi[bump_start_z:bump_end_z, : bump_end_th]=new_lattice[:, diff:]
          self.psi_squared[bump_start_z:bump_end_z, : bump_end_th]=new_psi_squared[:, diff:]
          self.dth[bump_start_z:bump_end_z, : bump_end_th]=new_dth[:, diff:]
          self.dz [bump_start_z:bump_end_z, : bump_end_th] = new_dz[:, diff:]
    self.update_bump_sigma(accept)

  def bump_array(self, addition, width_z, width_th, bump_width_z, bump_width_th):
    """
    TODO Unused sampling  proposal

    this needs to return a rectangular array holding a gaussian bump stretched out form 1x1 pixel to
    bump_idth_z x bump_width_th pixels
    with (complex) amplitude equivalent to a 1x1 bump with amplitude 'addition'
    additionally the returnde array is not larger than lattice z len +1 , lattice th len +1.  
    If the strecthed out bump wraps around the lattice more than once wrap it around and add up periodically
    Pad out the returned lattice with a left half-halo of 0s or, if periodic, duplicate rowi and col to left/bottom
    """
    #the non-wrapped, left-padded bump function
    # bump version 1 is just a rectangle, height 1, all sides padded with 0,
    #unless big enough to be periodic, then not padded
    array = np.zeros((bump_width_z+2, bump_width_th+2), dtype=complex)
    center_z = bump_width_z//2+1
    center_th = bump_width_th//2+1
    for i in (1, bump_width_z+1):
      for j in (1, bump_width_th+1):
          array[i,j]=1
    array*=addition
    if bump_width_th+2 > self.th_len:
      #cut out center section of larger array
      half_len = self.th_len//2
      rest_len = self.th_len-half_len
      wrapped_array =  array[:,center_th-half_len:center_th +rest_len]
      return wrapped_array
    else:
      return array

  def step_row_rotate(self, amplitude, maxrows=1):
    """TODO Unused sampling  proposal"""
    n_rot = random.choice([-1,1]) #give the row a twist of n_rot discrete rotations
    phase = random.uniform(0, 2*math.pi)
    num_rows = random.randint(1,maxrows)
    angle_per_cell = math.pi*2 / self.th_len
    # only +-1, they will stack up to higher disceret integera
    #repurposing bump sampling width to give this its own sampling width 
    row_index = random.randint(0, self.z_len-1) #this range is inclusive
    new_lattice =  np.zeros((num_rows+1, self.th_len), dtype=complex)# have to consider energy change of 3-row sublattice
    new_dth = np.zeros((num_rows+1, self.th_len), dtype=complex)
    new_dz = np.zeros((num_rows+1, self.th_len), dtype=complex)
    old_lattice =  np.zeros((num_rows+1, self.th_len), dtype=complex)# have to build copy of old 3 rows in complicated way
    old_dth = np.zeros((num_rows+1, self.th_len), dtype=complex) # due to periodicity, indexing issues
    old_dz = np.zeros((num_rows+1, self.th_len), dtype=complex)
    for i,row in enumerate(range(row_index, row_index+num_rows)):
      try:
        old_lattice[i] = self.psi[row, :]
        new_row = np.array([x*np.exp(1j*(angle_per_cell*n*n_rot+phase)) for n,x in enumerate(self.psi[row,:])]) #change phase of field values 
        new_lattice[i]=new_row
        #filling the changed row
        old_dth[i]=self.dth[row_index,:]
        old_dz[i]=self.dz[row_index,:]
        #calcualte row's dew dth (internal), dz(by looking down to row n-1)
        new_dth[i]= np.array([x-new_row[n-1] for n,x in enumerate(new_row)]) #no index problem lookng to index -1
        new_dz[i]= np.array([x-value_below for x, value_below in zip(new_row, self.psi[row-1])])
      except IndexError:
        row-=self.z_len
        old_lattice[i] = self.psi[row, :]
        new_row = np.array([x*np.exp(1j*(angle_per_cell*n*n_rot+phase)) for n,x in enumerate(self.psi[row,:])]) #change phase of field values 
        new_lattice[i]=new_row
        #filling the changed row
        old_dth[i]=self.dth[row_index,:]
        old_dz[i]=self.dz[row_index,:]
        #calcualte row's dew dth (internal), dz(by looking down to row n-1)
        new_dth[i]= np.array([x-new_row[n-1] for n,x in enumerate(new_row)]) #no index problem lookng to index -1
        new_dz[i]= np.array([x-value_below for x, value_below in zip(new_row, self.psi[row-1,:])])
    #filling extra row, potential index problem if index too big
    try:
      old_lattice[-1] = self.psi[row_index+1, :]
      new_lattice[-1] = old_lattice[-1]
      old_dth[-1] = self.dth[row_index+1, :]
      new_dth[-1] = old_dth[-1]
      old_dz[-1] = self.dz[row_index+1, :]
      new_dz[-1] = np.array([value_above-x for x, value_above in zip(new_lattice[-2,:], old_lattice[-1,:])])
    except IndexError:
      index = row_index+1-self.z_len
      old_lattice[-1] = self.psi[index, :]
      new_lattice[-1] = old_lattice[-1]
      old_dth[-1] = self.dth[index, :]
      new_dth[-1] = old_dth[-1]
      old_dz[-1] = self.dz[index, :]
      new_dz[-1] = np.array([value_above-x for x, value_above in zip(new_lattice[-2,:], old_lattice[-1,:])])
    new_dz/= self.z_pixel_len
    new_dth/= self.th_pixel_len
    new_psi_squared = np.multiply(new_lattice, new_lattice.conjugate()) 
    old_psi_squared = np.multiply(old_lattice, old_lattice.conjugate()) 
    new_energy=self.sublattice_field_energy(amplitude, row_index,new_lattice, new_psi_squared, new_dz, new_dth) 
    old_energy=self.sublattice_field_energy(amplitude, row_index, old_lattice, old_psi_squared, old_dz, old_dth)
    accept= self.me.metropolis_decision(old_energy, new_energy)
    if accept:
      start=row_index
      end = row_index+num_rows+1
      try: 
        self.psi[start:end,: ]=new_lattice
        self.psi_squared[start:end,: ]=new_psi_squared
        self.dth[start:end,: ]=new_dth
        self.dz[start:end,: ]=new_dz
      except (IndexError, ValueError):
        split = self.z_len - row_index
        end-=self.z_len
        self.psi[start:,: ]=new_lattice[:split]
        self.psi_squared[start:,: ]=new_psi_squared[:split]
        self.dth[start:,: ]=new_dth[:split]
        self.dz[start:,: ]=new_dz[:split]
        self.psi[:end,: ]=new_lattice[split:]
        self.psi_squared[:end,: ]=new_psi_squared[split:]
        self.dth[:end,: ]=new_dth[split:]
        self.dz[:end,: ]=new_dz[split:]
      print("changed row s", row_index,"+",num_rows , "by ", n_rot)
    #self.update_bump_sigma(accept)

  def step_lattice_wavevector(self, amplitude, wavevector):
    """TODO Unused sampling  proposal"""
    addition = random.gauss(0,self.bump_sampling_width)+random.gauss(0,self.bump_sampling_width)*1j
    #repurposing bump sampling width to give this its own sampling width 
    lattice_addition = self.wave_array(addition, wavevector, self.z_len, self.th_len)
    new_lattice=self.psi+lattice_addition
    new_psi_squared = np.multiply(new_lattice, new_lattice.conjugate()) #np.square is elementwise square of complex numbers, z^2 not zz*
    new_dz=np.zeros((self.z_len, self.th_len))
    new_dth=np.zeros((self.z_len, self.th_len))
    for z_index in range(self.z_len):
      for th_index in range(self.th_len):    
        value= new_lattice[z_index, th_index]
        left_value_z = new_lattice[z_index-1, th_index]
        new_dz[z_index, th_index]  = value-left_value_z
        left_value_th = new_lattice[z_index, th_index-1]
        new_dth[z_index, th_index]  =   value-left_value_th
    new_dz/= self.z_pixel_len
    new_dth/= self.th_pixel_len
    new_energy=shape_field_energy(amplitude, amplitude, new_lattice, new_psi_squared, new_dz, new_dth) #dz,dth differnces nuchanged by uniform addition
    old_energy=self.energy#calculated just previously in run()
    accept= self.me.metropolis_decision(old_energy, new_energy)
    if accept:
      self.psi=new_lattice
      self.psi_squared=new_psi_squared
      self.dth = new_dth
      self.dz = new_dz
      self.energy=new_energy
      #self.me.energy["field"] = field_energy #ok to not do this until later as next steps is field internal
    me.update_sigma(accept, "field_bump")
  
  def step_lattice_random_wavevector(self, amplitude, qzmax=4, qthmax=8):
    """TODO Unused sampling  proposal"""
    wavevector = (random.randint(-qzmax, qzmax+1)*2*math.pi, random.randint(-qthmax, qthmax+1)*2*math.pi)
    self.step_lattice_wavevector(amplitude,wavevector)

  def wave_array(self,base, wavevector, z_len, th_len):
    """TODO Unused sampling  proposal"""
    qz, qth= wavevector
    # sine, not complex exponent becaue I want spatially varying amplitude
    #complex exponent when I want to prod rotational states
    return np.fromfunction(lambda i,j: base*np.exp(1j*(i*qz/z_len+j*qth/th_len)) ,(z_len, th_len)) 

  def sublattice_field_energy(self, amplitude, shape):
    """TODO Unused sampling  proposal"""
    energy=0
    z_dim, th_dim = lattice.shape
    #again passed end index because negative indeices out of range are better handled
    for i, z_index in enumerate(range(z_index_start,z_index_start+z_dim)): # not taking energy from first row & col, which are padding
      z_loc = z_index * self.z_pixel_len
      z_loc_interstitial = (z_index-.5) * self.z_pixel_len
      z_spacing = shape.sqrt_g_z(z=z_loc_interstitial, amplitude=amplitude)
      th_spacing = shape.sqrt_g_theta(z=z_loc, amplitude=amplitude)
      col_sqrtg = z_spacing*th_spacing
      col_index_raise_and_sqrtg = shape.sqrt_g_z(z=z_loc, amplitude=amplitude)/shape.sqrt_g_theta(z=z_loc_interstitial, amplitude=amplitude)
      col_A_th = shape.A_theta(z=z_loc_interstitial, amplitude=amplitude)
      psi_col = lattice[i]
      psi_squared_column = psi_squared[i]
      #TODO could do *sqrtg at the end to whole column, if covariant laplacian returned value/sqrt_g
      energy_col = (self.alpha*sum(psi_squared_column)+self.u/2*sum(psi_squared_column**2)) *col_sqrtg
      dz_col = dz[i]/z_spacing #dz does contain /z_pixel_len, but not sqrtgz adjustment
      #TODO likelyt problem part
      energy_col += self.C*sum(self.squared(dz_col))*col_sqrtg #TODO check this squares elementwise, then sums
      dth_col = dth[i]#/th_spacing <- this is the index raise?
      #C|dth Psi(x)|^2 part of energy density
      energy_col += self.C*sum(self.squared(dth_col))*col_index_raise_and_sqrtg
      #-iCn(A_th* Psi* dth Psi(x)) and c.c. part of energy density
      energy_col += self.C*self.n*2*sum((col_A_th.conjugate()*psi_col.conjugate()*dth_col).imag)*col_index_raise_and_sqrtg
      # Cn^2|A_th Psi(x)|^2  part of energy density
      energy_col += self.Cnsquared*sum(self.squared(psi_col))*self.squared(col_A_th)*col_index_raise_and_sqrtg
      energy += energy_col
    #print("amplitude ", amplitude, "would get field energy", energy*self.z_pixel_len*self.th_pixel_len)
    return energy*self.z_pixel_len*self.th_pixel_len
  
  #####################################   end extras   #########################################

if __name__ == "__main__":
  """
  main for test
  """
  import matplotlib.pyplot as plt

  n_substeps=10
  base_dims = (50, 50)
  wavenumber=.9
  radius=1
  dims = (int(math.floor(base_dims[0]/wavenumber)), base_dims[1])
  lattice = Lattice(alpha=-4, u=1, C=1.5, n=6, dims=dims, wavenumber=wavenumber, radius=radius, n_substeps=n_substeps)
  
  #mock of run loop fixed_amplitude in main.py:
  #makes a shape object
  gamma=1
  kappa=0
  amplitude=.6
  cy = system_cylinder.Cylinder(gamma=gamma, kappa=kappa, wavenumber=wavenumber, radius=radius)

  #makes a metropolis object
  temperature=.001
  sigmas_initial = {"field":.007}
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
    print(i)

  #mock output
  plt.plot(field_energy_history)
  plt.show()
  plt.imshow(lattice.psi_squared)
  plt.show()

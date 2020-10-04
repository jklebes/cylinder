import numpy as np
import random
import math
import cmath
import matplotlib.pyplot as plt

import system_cylinder
import metropolisengine

class Lattice():
  def __init__(self,dims = (100,50) ):
    #experiment variables
    self.wavenumber = .2
    self.radius=1
    self.kappa=0
    self.amplitude = 0
    self.alpha = -1
    self.u = 1
    self.C= 1
    self.n= 0
    self.Cnsquared = self.C*self.n**2
    self.temperature = .001
    #lattice characteristics
    self.z_len, self.th_len = dims
    cylinder_len_z = 2*math.pi / self.wavenumber # = wavelength lambda
    cylinder_radius_th = 2*math.pi*self.radius # circumference - in len units, not radians
    self.z_pixel_len = cylinder_len_z /self.z_len # length divided py number of pixels
    self.th_pixel_len = cylinder_radius_th /self.th_len
    #set up metropolis engine coupled to bare cylinder system
    self.surface = system_cylinder.Cylinder(wavenumber=self.wavenumber, radius=self.radius, kappa=self.kappa)
    #Psi at each point 
    self.lattice = np.zeros((self.z_len, self.th_len), dtype=complex)
    #values also saved for convenience
    self.psi_squared = np.zeros((self.z_len, self.th_len)) 
    #self.dz_squared = np.zeros((self.z_len, self.th_len))
    self.dz = np.zeros((self.z_len, self.th_len), dtype=complex)
    self.dth = np.zeros((self.z_len, self.th_len), dtype=complex)
    # note: these are left derivatives:
    # derivative stored at i refers to difference between i-1, i positions in corrseponding 
    # grid of lattice values

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
    self.me = metropolisengine.MetropolisEngine(energy_functions = energy_fct_by_params_group,  initial_complex_params=None, initial_real_params = [float(self.amplitude)], 
                                 covariance_matrix_complex=None, sampling_width=.05, temp=self.temperature
                                 , complex_sample_method=None)
    self.me.set_reject_condition(lambda real_params, complex_params : abs(real_params[0])>=.99 )  
    self.lattice_acceptance_counter = 0
  
  def squared(self, c):
    return abs(c*c.conjugate())

  def random_initialize(self):
    #assuming that this is called on amplitude=0 cylinders
    assert(self.amplitude==0) #if this fails implement initialize on a!=0 
    for z_index in range(self.z_len):
      for th_index in range(self.th_len):
        #fill lattice
        value = cmath.rect(random.uniform(0,1), random.uniform(0, 2*math.pi))
        self.lattice[z_index, th_index] = value
        #fill stored energy density, 
        self.psi_squared[z_index, th_index] = self.squared(value) 
    for z_index in range(self.z_len):
      for th_index in range(self.th_len):    
        #dz
        value= self.lattice[z_index, th_index]
        left_value_z = self.lattice[z_index-1, th_index]
        try:
          right_value_z = self.lattice[z_index+1, th_index]
        except IndexError:
          right_value_z = self.lattice[0, th_index]
        self.dz[z_index, th_index]  = self.covariant_laplacian_z( value, left_value_z, right_value_z,
                                      z_loc=z_index*self.z_pixel_len, amplitude=0)
        
        #dth
        left_value_th = self.lattice[z_index, th_index-1]
        try:
          right_value_th = self.lattice[z_index, th_index+1]
        except IndexError:
          right_value_z = self.lattice[z_index, 0]
        self.dth[z_index, th_index]  =  self.covariant_laplacian_theta(value, left_value_th, right_value_th,
                                       z_loc=z_index*self.z_pixel_len, amplitude=0)

  
  def left_derivative(self, index_z, index_th):
    #choose a formula for numerical derivative on a lattice
    #left or right derivative is fine for an energy density of |d_thPsi|^2, |d_zPsi|^2 parts
    #'left': Psi_i - Psi_(i-1)
    #'right': Psi_(i+1)- Psi_i
    #                    problem: derivative in cross-term A_thPsi*d_thPsi has to be done differently
    # symmetrized: avg(left + right) ~ (Psi_(i+1)- Psi_(i-1)) does not include the point itself,
    #              risks splitting into two disconnected sublattices
    # five-point: does not include the point itself but couples lattice strongly,
    #             computationally intensive: change to psi_i affects 4 other values of d Psi 
    # energy from new differences to neighbors
    old_dz_squared_left = self.dz_squared[index_z, index_th] 
    old_dth_squared_up = self.squared(self.dth[index_z, index_th])
    new_dz_squared_left  = self.squared(self.lattice[index_z-1, index_th] - new_value)
    try:
      old_dth_squared_down = self.squared(self.dth[index_z, index_th+1])
      new_dth_down =  (self.lattice[index_z, index_th+1] - new_value)
      new_dth_squared_down= self.squared(new_dth_down)
    except IndexError:
      old_dth_squared_down = self.squared(self.dth[index_z, 0])
      new_dth_down=  (self.lattice[index_z, 0] - new_value )
      new_dth_squared_down= self.squared(new_dth_down)
    try:
      old_dz_squared_right = self.dz_squared[index_z+1, index_th]
      new_dz_squared_right= self.squared(new_value - self.lattice[index_z+1, index_th])
    except IndexError:
      old_dz_squared_right = self.dz_squared[0, index_th]
      new_dz_squared_right= self.squared(new_value - self.lattice[0, index_th])
    new_dth_up = (new_value - self.lattice[index_z, index_th-1])
    new_dth_squared_up = self.squared(new_dth_up)
    diff_dz_left= new_dz_squared_left - old_dz_squared_left
    diff_dth_down= new_dth_squared_down - old_dth_squared_down
    diff_dz_right= new_dz_squared_right - old_dz_squared_right
    diff_dth_up = new_dth_squared_up - old_dth_squared_up
    pass

  def covariant_laplacian_z(self,value, left_value, right_value, z_loc, amplitude):
    """
    a value for |d_z Psi|^2 sqrt{g} at a point derived by using the formula:
    |d_z Psi|^2 sqrt{g} = - Psi* d_z ( sqrt{g} eta^{zz} d_z Psi) = - Psi* d_z ( sqrt{g} d_z Psi)
    (eta^{zz} of cylindrical coordinate system, for index raising, is 1)
    with operator :
    d_z (sqrt{g} d_z Psi)
    applied to Psi
    """
    leftderivative = value - left_value
    rightderivative = right_value - value
    leftderivative *= (self.surface.sqrt_g_z(z=z_loc-.5*self.z_pixel_len, amplitude=amplitude)*
                     self.surface.sqrt_g_theta(z=z_loc-.5*self.z_pixel_len, amplitude=amplitude)) #* sqrt_g  at point i-1/2
    rightderivative *=  (self.surface.sqrt_g_z(z=z_loc+.5*self.z_pixel_len, amplitude=amplitude)*
                     self.surface.sqrt_g_theta(z=z_loc+.5*self.z_pixel_len, amplitude=amplitude))#* sqrt_g  at point i+1/2
    secondderivative = rightderivative - leftderivative
    return -value.conjugate()*secondderivative

  def covariant_laplacian_theta(self,value, left_value, right_value, z_loc, amplitude):
    """
    a value for |D_th Psi|^2 sqrt{g} at a point derived by using the formula:
    |D_th Psi|^2 sqrt{g} = - Psi* D_th* ( sqrt{g} eta^{thth} D_th Psi) = - Psi* D_th* ( sqrt_g_z / sqrt_g_th D_th Psi)
    (index raising eta^{thth}=1/R^2=1/sqrt_gth^2 , partially cancels with sqrt{g}=sqrt_g_z*sqrt_g_th)
    with operator :
    D_th* (sqrt_g_z/sqrt_g_th D_th Psi)
    applied to Psi
    D_th = (d_th - inA_th) so this has 4 parts
    1)- operator d_th* (sqrt_g_z/sqrt_g_th d_th Psi)
    2)- operator A_th* (sqrt_g_z/sqrt_g_th A_th Psi) is just multiply by |A_th|^2, sqrt_g_z/sqrt_g_th
    3)- cross terms a) A_th* (sqrt_g_z/sqrt_g_th d_th Psi)
        and         b) d_th* (sqrt_g_z/sqrt_g_th A_th Psi) = sqrt_g_z/sqrt_g_th A_th d_thPsi = c.c. of 3a)
    3a) and 3b) together = 2 Im(3a)
    """
    sqrt_g_z = self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude) #same for the locatio and its theta-direction neightbors
    sqrt_g_th = self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude)
    sqrt_g_and_index_raise = sqrt_g_z/sqrt_g_th 
    A_th = self.surface.A_theta(z=z_loc, amplitude=amplitude)
    n = self.n
    #1) same scheme as in covariant_laplacian_z
    leftderivative = value - left_value
    rightderivative = right_value- value
    leftderivative *= sqrt_g_and_index_raise
    rightderivative *=  sqrt_g_and_index_raise
    secondderivative = rightderivative - leftderivative
    term1 =  -value.conjugate()*secondderivative
    term2 = - n**2 *self.squared(A_th)*self.squared(value)* sqrt_g_and_index_raise
    #3a: avg of left derivative and right derivative, times A_th*
    term3 = - n * value.conjugate()*A_th.conjugate() * (leftderivative+rightderivative)/2 # avg derivative doesnt include value at the point i
    return term1 + term2 + 2* term3.imag

  def surface_field_energy(self, amplitude):
    """calculates energy on proposed amplitude change"""
    energy_cols = np.zeros(self.z_len)
    for z_index in range(-1,self.z_len-1):
      z_loc = z_index * self.z_pixel_len
      psi_col = self.lattice[z_index]
      psi_squared_column = self.psi_squared[z_index]
      #TODO could do *sqrtg at the end to whole column, if covariant laplacian returned value/sqrt_g
      energy_col = (self.alpha*sum(psi_squared_column)+self.u/2*sum(psi_squared_column**2)) *self.surface.sqrt_g_z(amplitude=amplitude, z=z_loc)
      for th_index in range(-1,self.th_len-1):
        energy_col += self.covariant_laplacian_theta(value = psi_col[th_index], left_value = psi_col[th_index-1], right_value=psi_col[th_index+1],
                                                    z_loc = z_loc, amplitude=amplitude)*self.C
        energy_col += self.covariant_laplacian_theta(value = psi_col[th_index], left_value = self.lattice[z_index-1, th_index], right_value=self.lattice[z_index+1, th_index],
                                                    z_loc = z_loc, amplitude=amplitude)*self.C
      energy_cols[z_index] = energy_col
    print("amplitude ", amplitude, "would get field energy", sum(energy_cols)*self.z_pixel_len*self.th_pixel_len)
    return sum(energy_cols)*self.z_pixel_len*self.th_pixel_len

  def fill_derivatives(self, amplitude):
    #sadly all derivative values have to be recalculated when amplitude changes
    # and it can't be done as simply as multiplying by something (in covarant laplacian scheme)
    for z_index in range(-1, self.z_len-1):
      z_loc = z_index*self.z_pixel_len
      for th_index in range(-1,self.th_len-1):
        value=self.lattice[z_index, th_index-1]
        self.dth[z_index, th_index]= self.covariant_laplacian_theta(value = value, 
                                                    left_value =self.lattice[z_index, th_index-1], right_value=self.lattice[z_index, th_index+1],
                                                    z_loc = z_loc, amplitude=amplitude)
        self.dz[z_index, th_index]= self.covariant_laplacian_z(value = value, 
                                                    left_value = self.lattice[z_index-1, th_index], right_value=self.lattice[z_index+1, th_index],
                                                    z_loc = z_loc, amplitude=amplitude)

  def run(self, n_steps, n_sub_steps):
    for i in range(n_steps):
      #metropolis step shape
      surface_accepted = self.me.step_real_group()
      if surface_accepted:
        self.amplitude = self.me.real_params[0]
        self.fill_derivatives(self.amplitude)
      #lattice step
      for i in range(n_sub_steps):
        self.step_lattice(self.amplitude)


  def step_lattice(self, amplitude):
    #choose a location
    index_z, index_th = (random.randrange(self.z_len), random.randrange(self.th_len))
    z_loc = self.z_pixel_len * index_z +.5*self.z_pixel_len
    #properties of the surface at this point
    A_th= self.surface.A_theta(z=z_loc, amplitude=amplitude)  #should alos have units - per th distance?
    #sqrt_g = (self.surface.sqrt_g_theta(z=z_loc, amplitude=amplitude)*self.surface.sqrt_g_z(z=z_loc, amplitude=amplitude))
    
    # random new value with magnitude similar to old value
    new_value = cmath.rect(random.gauss(abs(self.lattice[index_z, index_th]),1), random.uniform(0, 2*math.pi))
    new_psi_squared = self.squared(new_value)
    #energy difference of switching the pixel:
    old_psi_squared = self.psi_squared[index_z, index_th]
    old_pixel_energy_density = self.alpha *old_psi_squared + self.u/2*old_psi_squared**2
    new_pixel_energy_density = self.alpha * new_psi_squared + self.u/2*new_psi_squared**2
    diff_pixel_energy_density = new_pixel_energy_density -  old_pixel_energy_density

    left_value_z = self.lattice[index_z-1, index_th]
    try:
      right_value_z = self.lattice[index_z+1, index_th]
    except IndexError:
      right_value_z = self.lattice[0, index_th]
    left_value_th = self.lattice[index_z, index_th-1]
    try:
      right_value_th = self.lattice[index_z, index_th+1]
    except IndexError:
      right_value_th = self.lattice[index_z, 0]
    new_derivative_z = self.covariant_laplacian_z(new_value, left_value_z, right_value_z, z_loc=z_loc, amplitude=amplitude)
    new_derivative_th = self.covariant_laplacian_theta(new_value, left_value_th, right_value_th, z_loc=z_loc, amplitude=amplitude)
    diff_derivative_z = new_derivative_z - self.dz[index_z, index_th] 
    diff_derivative_th = new_derivative_th - self.dth[index_z, index_th] 
    
    diff_energy = self.C*( diff_derivative_z + diff_derivative_th )
    #TODO optioanlly also add effect on derivative at neighboring locations
    try:
      new_derivative_zplusone = self.covariant_laplacian_z(value=right_value_z, left_value = new_value, right_value = self.lattice[index_z+2, index_th], z_loc=z_loc, amplitude=amplitude)
    except IndexError:
      z_plus_two = index_z + 2 - self.z_len
      new_derivative_zplusone = self.covariant_laplacian_z(value=right_value_z, left_value = new_value, right_value = self.lattice[z_plus_two, index_th], z_loc=z_loc, amplitude=amplitude)
    new_derivative_zminusone = self.covariant_laplacian_z(value=left_value_z, left_value = self.lattice[index_z-2, index_th], right_value = new_value, z_loc=z_loc, amplitude=amplitude)
    diff_energy += self.C*(new_derivative_zminusone - self.dz[index_z-1, index_th])
    try:
      diff_energy += self.C*(new_derivative_zplusone - self.dz[index_z+1, index_th])
    except IndexError:
      diff_energy += self.C*(new_derivative_zplusone - self.dz[0, index_th])
    try:
      new_derivative_thplusone = self.covariant_laplacian_theta(value=right_value_th, left_value = new_value, right_value = self.lattice[index_z, index_th+2], z_loc=z_loc, amplitude=amplitude)
    except IndexError:
      th_plus_two = index_th + 2 - self.th_len
      new_derivative_thplusone = self.covariant_laplacian_theta(value=right_value_th, left_value = new_value, right_value = self.lattice[index_z, th_plus_two], z_loc=z_loc, amplitude=amplitude)
    new_derivative_thminusone = self.covariant_laplacian_theta(value=left_value_th, left_value = self.lattice[index_z, index_th-2], right_value = new_value, z_loc=z_loc, amplitude=amplitude)
    diff_energy += self.C*(new_derivative_thminusone - self.dth[index_z, index_th-1])
    try:
      diff_energy += self.C*(new_derivative_thplusone - self.dz[index_z, index_th+1])
    except IndexError:
      diff_energy += self.C*(new_derivative_thplusone - self.dz[index_z, 0])
    
    diff_energy/= self.surface.sqrt_g_theta(amplitude=amplitude, z=z_loc)*self.surface.sqrt_g_z(amplitude=amplitude, z=z_loc)
    diff_energy += diff_pixel_energy_density
  
    #diff_energy *= self.z_pixel_len*self.th_pixel_len #leavin this out scales effective temperature of lattice part of metropolis simulation
                                                      # relative ot surface part
    #make metropolis decision with energy difference, same temperature as engine is set to
    if self.me.metropolis_decision(0,diff_energy):
      #change stored value of pixel
      self.lattice[index_z,index_th] = new_value
      #change stored values of dth, dz across its boundaries
      #self.energy += diff_energy
      self.lattice_acceptance_counter+=1
      #fill stored energy density, 
      self.psi_squared[index_z, index_th]  = new_psi_squared
      #dz
      self.dz[index_z, index_th]  = new_derivative_z
      try:
        self.dz[index_z+1, index_th]  = new_derivative_zplusone
      except IndexError:
        self.dz[0, index_th]  = new_derivative_zplusone
      self.dz[index_z-1, index_th]  = new_derivative_zminusone
      #dth
      self.dth[index_z, index_th] = new_derivative_th 
      try:
        self.dth[index_z, index_th+1] = new_derivative_thplusone
      except IndexError:
        self.dth[index_z, 0] = new_derivative_thplusone
      self.dth[index_z, index_th-1] = new_derivative_thminusone
      # these incorporate ampltitude and position-dependent quatities in complicated ways, only valid until ampltiude changes!

      
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

if __name__ == "__main__":
  lattice = Lattice()
  n_steps=1600
  n_sub_steps=lattice.z_len*lattice.th_len
  lattice.run(n_steps, n_sub_steps)
  print(lattice.lattice)
  print(lattice.amplitude)
  print(lattice.lattice_acceptance_counter)
  lattice.plot()

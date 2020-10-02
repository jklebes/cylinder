import numpy as np
import random
import math
import matplotlib.pyplot as plt

import system_cylinder
import metropolisengine

class Lattice():
  def __init__(self):
    self.z_len = 100
    self.th_len = 50
    self.wavenumber = .5
    self.radius=1
    self.z_pixel_len = 2*math.pi/(self.wavenumber *self.z_len)
    self.th_pixel_len = 2*math.pi*self.radius/self.th_len
    self.kappa=0
    self.amplitude = 0
    self.alpha = -1
    self.alpha_effective = self.alpha
    self.u = 1
    C=.1
    n=1
    self.Cnsquared = C*n**2
    self.line_tension = 0
    self.lattice = np.zeros((self.th_len, self.z_len))
    self.energy = 0
    self.temperature = .1
    #set up metropolis engine coupled to bare cylinder system
    self.surface = system_cylinder.Cylinder(wavenumber=self.wavenumber, radius=self.radius, kappa=self.kappa)
    self.ones_count = np.zeros(self.z_len)
    self.z_lines_count = np.zeros(self.z_len)
    self.th_lines_count = np.zeros(self.z_len)
    self.random_initialize()
    print("initialized\n", self.lattice)
    print("energy:", self.energy)
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
  
  def surface_field_energy(self,amplitude):
    print("called for field energy with amplitude", amplitude)
    energy = 0
    for z_index in range(self.z_len):
      z_loc = z_index* self.z_pixel_len
      #distortion of line elements at the queried amplitude and at this z location
      sqrt_g_zz = self.surface.sqrt_g_z(amplitude=amplitude, z=z_loc)
      sqrt_g_thth = self.surface.sqrt_g_z(amplitude=amplitude, z=z_loc)
      # get energy from all the '1' field values being weighted by sqrt(g) (function of z location)
      energy += self.ones_count[z_index]* self.energy_pixel(z_loc=z_loc, amplitude=amplitude)
      #get energy from all the --- boundary lines weighted by sqrt(g_zz)
      energy += self.z_lines_count[z_index]*self.line_tension*sqrt_g_zz*self.z_pixel_len
      # get energy from all | coundary lines - weighted by sqrt(g_thth)
      energy += self.th_lines_count[z_index]*self.line_tension*sqrt_g_thth*self.th_pixel_len
    return energy

  def random_initialize(self):
    #fill lattice with random 1s and 0s and calculate its energy
    for i in range(self.z_len):
      z_loc = i * self.z_pixel_len
      for j in range(self.th_len):
        value = random.choice([0,1])
        self.lattice[j,i] = value
        #energy from pixel values
        if value:
          self.energy += self.energy_pixel(z_loc, amplitude=self.amplitude)
    print(sum(sum(self.lattice))," on pixels", self.energy, "energy")
    #scale the whole thing by area 
    print("scaled surface down by * ",self.z_pixel_len * self.th_pixel_len)
    #add energy from line tensions
    line_energy = 0
    #get lists of line element lengths from already-initalized surface
    #self.th_line_lengths =[surface.sqrt_g_th(amplitude=self.amplitude, z) for z in z_len] 
    #self.z_line_lengths = [surface.sqrt_g_z(amplitude=self.amplitude, z) for z in z_len] 
    for i in range(self.z_len):
      z_loc = i * self.z_pixel_len
      for j in range(self.th_len):
        #look down(+th direction) and right(+z direction only)
        try:
          diff_th = (self.lattice[j,i] != self.lattice[j+1, i])
        except IndexError:
          diff_th = (self.lattice[j,i] != self.lattice[0, i])
        try:
          diff_z = (self.lattice[j,i] != self.lattice[j, i+1])
        except IndexError:
          diff_z = (self.lattice[j,i] != self.lattice[j, 0])
        if diff_th:
          self.z_lines_count[i]+=1
          line_energy+=self.surface.sqrt_g_theta(amplitude=self.amplitude, z=z_loc)*self.th_pixel_len
        if diff_z:
          self.th_lines_count[i]+=1
          line_energy+=self.surface.sqrt_g_z(amplitude=self.amplitude, z=z_loc)*self.z_pixel_len
    self.energy += line_energy*self.line_tension
    self.ones_count = sum(self.lattice)
    print(self.ones_count)
    assert(all([count % 2 == 0 for count in self.z_lines_count]))

  def energy_pixel(self,z_loc,amplitude):
    # energy of having field |Psi|=1 rather than |Psi|=0 in this location
    #inherent energy advatnage of ordered state |Psi| = sqrt(alpha/u)
    # - correspoinding energy is -1/2 (alpha**2 / u)
    #plus disadvantage from connection +Cn^2|A_th(z,a)|^2
    #weighted by size of area element
    #TODO : could be faster by doing all at a given z_loc at the same time
    area_element = self.z_pixel_len * self.th_pixel_len*self.surface.sqrt_g_z(amplitude=amplitude, z=z_loc)* self.surface.sqrt_g_z(amplitude=amplitude, z=z_loc)
    return (self.alpha**2/self.u* (-.5 + self.Cnsquared/self.u* self.surface.A_theta(amplitude=
                    amplitude, z = z_loc)**2) )* area_element

  def run(self):
    for i in range(n_steps):
      #metropolis step shape
      surface_accepted = self.me.step_real_group()
      #effective alpha is alpha + Cn^2|A_i(x)|^2 - need to save A_i(z)
      # as function or list
      self.amplitude = self.me.real_params[0]
      for i in range(n_sub_steps):
        self.step_lattice()

  def step_lattice(self):
    #choose a location
    index_z, index_th = (random.randrange(self.z_len), random.randrange(self.th_len))
    z_loc = self.z_pixel_len * index_z
    #energy difference of switching the pixel:
    # sign of switch * area element weighting for the location and amplitude 
    old_value = self.lattice[index_th, index_z]
    if old_value == 0:
      sign = +1
    else:
      sign = -1
    diff_pixel = sign * self.energy_pixel(z_loc, amplitude=self.amplitude)

    #figure out whether differce to neighbors is added or eliminated
    #get length of line segments at the location
    #metropolis step the pixel
    sign_down, sign_up, sign_left, sign_right = (1,1,1,1)
    try:
      if self.lattice[index_th, index_z] != self.lattice[index_th+1, index_z]:
        sign_down = -1
    except IndexError:
      if self.lattice[index_th, index_z] != self.lattice[0, index_z]:
        sign_down = -1
    try:
      if self.lattice[index_th, index_z] != self.lattice[index_th, index_z+1]:
        sign_right = -1
    except IndexError:
      if self.lattice[index_th, index_z] != self.lattice[index_th, 0]:
        sign_right = -1 # x -> x
    if self.lattice[index_th, index_z] != self.lattice[index_th-1, index_z]:
      sign_up = -1
    if self.lattice[index_th, index_z] != self.lattice[index_th, index_z-1]:
      sign_left = -1
    #line element lengths
    left_line_element = self.surface.sqrt_g_theta(amplitude = self.amplitude, z = (z_loc-self.z_pixel_len)) * self.th_pixel_len # |x
    right_line_element = self.surface.sqrt_g_theta(amplitude = self.amplitude, z = (z_loc))* self.th_pixel_len              #  x|
    up_line_element = self.surface.sqrt_g_z(amplitude = self.amplitude, z = (z_loc)) * self.z_pixel_len                     #  _
    down_line_element = up_line_element                                                                     #  x
    diff_left_line = sign_left * left_line_element
    diff_right_line = sign_right * right_line_element
    diff_up_line = sign_up * up_line_element 
    diff_down_line = sign_down * down_line_element

    diff_energy = diff_pixel + diff_left_line + diff_right_line + diff_up_line + diff_down_line
    #make metropolis decision with energy difference, same temperature as engine is set to
    if self.me.metropolis_decision(0,diff_energy):
      self.lattice[index_th,index_z] = int(not old_value)
      self.energy += diff_energy
      self.lattice_acceptance_counter+=1
      # these are used to feed back to cylinder shape, from them the changes in field
      # energy on  of a can be derived
      self.ones_count[index_z] += sign
      #print("changing from " ,old_value,"to", self.lattice[index_th, index_z], "sign", sign)
      self.z_lines_count[index_z] += sign_up + sign_down
      self.th_lines_count[index_z] += sign_left + sign_right
    #print(self.lattice)
    #print(self.amplitude, self.energy)
    #print(self.lattice_acceptance_counter)
    #print(self.ones_count)
      
  def plot(self):
    zs = [i for i in range(self.z_len)]
    #draw two sin curves for cylinder shape
    amplitude_scaling = 10
    cylinder_top = [-15+amplitude_scaling*self.amplitude*math.sin(z*2*math.pi/self.z_len) for z in zs]
    cylinder_bottom = [-35-amplitude_scaling*self.amplitude*math.sin(z*2*math.pi/self.z_len) for z in zs]
    plt.plot(zs, cylinder_top, color='g', linewidth=10)
    plt.plot(zs, cylinder_bottom, color='g', linewidth=10)
    #plot binary lattice field
    plt.imshow(self.lattice)

    plt.show()

if __name__ == "__main__":
  lattice = Lattice()
  n_steps=1000
  n_sub_steps=lattice.z_len*lattice.th_len
  lattice.run()
  print(lattice.lattice)
  print(lattice.ones_count)
  print(lattice.amplitude)
  lattice.plot()
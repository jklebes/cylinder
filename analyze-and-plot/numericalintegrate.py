import scipy
import numpy as np 
import matplotlib.pyplot as plt
import math
import pandas as pd

import system1D as system

"""
in the case of no-field simulations (cylinders with surface tension and ebnding rigidity), the parameter space to minimize over is one-dimensional: a=-1 to 1.
Markov chain Monte Carlo simulation is not strictly necessary to discover the shape and minimum of the potential H(a) = gamma * surface area + kappa * curvature integral.  Surface area can be anlytically solved; curvature integral as a function of a is here integrated and minimized numberically.
"""

def cylinder_energy():
  pass

def sphere_energy(kappa, kappa_gaussian, wavenumber, radius=1, gamma=1):
  """not terribly useful, because a given "wavenumber" of spheres
  is only some metastable state"""
  #hard code the energy of an equalivalent-volume sphere:
  #surface area term + mean curvautre terms + gaussian curvature terms
  # TODO optimize facotrs of pi etc
  volume = radius*4*math.pi**2 / wavenumber #volume of cylinder with radius r=1, circumfernce = radius*2pi, height = wavelength = 2pi/wavenumber
  sphere_radius = (volume * 3 / (4.0 * math.pi) )**(1.0/3.0)  #associated sphere radius
  surface_area = 4*math.pi*sphere_radius**2
  mean_curvature_squared = surface_area * (2 / sphere_radius)**2 # mean curvature is everywhere 2/R, mean curvature integrated over surface is SA*2/R
  gauss_curvature =  surface_area / (sphere_radius )**2 #gaussian curvature is everywhere 1/R^2
  return gamma*surface_area + kappa*mean_curvature_squared * kappa_gaussian*gauss_curvature

def integrate(wavenumber, kappa, precision, gamma, radius):
  sys = system.System(wavenumber=wavenumber, radius=radius, alpha=-1, C=1, u=1, n=1, kappa=kappa, gamma=gamma)
  #range should not include 1, -1 because radius is in places 0 at this shape (g_zz is 0, sometimes divided by)
  a_s = np.arange(-1+precision, 1, precision)
  surface_area_potential = [2*math.pi*sys.calc_surface_area(amplitude=a) for a in a_s]
  curvature_potential = [2*math.pi*sys.calc_bending_energy(amplitude=a) for a in a_s]
  potential = [gamma*x+kappa*y for (x,y) in zip(surface_area_potential, curvature_potential)]
  return a_s, surface_area_potential, curvature_potential, potential

def graph(wavenumber, kappa, kappa_gaussian=1, precision=.01, gamma=1, radius=1):
  a_s, surface_area_potential, curvature_potential, potential = integrate(wavenumber, kappa, precision, gamma, radius)
  sphereenergy = sphere_energy(kappa=kappa, kappa_gaussian=kappa_gaussian, wavenumber=wavenumber, radius=radius, gamma=gamma)
  sphere_potential = [sphereenergy for a in a_s]
  plt.plot(a_s,surface_area_potential, label = "surface area")
  #plt.plot(a_s, curvature_potential, label = "curvature term")
  #plt.plot(a_s, potential, color="black", label = "total potential")
  #plt.plot(a_s, sphere_potential, linestyle="dashed", label = "sphere potential")
  plt.xlabel("a")
  plt.ylabel("Energy")
  plt.legend()
  plt.show()

def numerical_minimize(wavenumber, kappa,  precision=.01, gamma=1, radius=1):
  """
  numerically integrate to find H(a) at different values of a (interval given by precision)
  and find minima of one-dimensional potential H(a) = surface area + bending term
  """
  a_s, surface_area_potential, curvature_potential, potential = integrate(wavenumber, kappa, precision, gamma, radius)
  #minimum of potential by simple search of precision is low 
  min_ = min(potential)
  index= potential.index(min_)
  a_min = (a_s[0])+index*precision
  return a_min, min_

def wavenumber_kappa_loop(wvn_range, kappa_range, precision=.01, gamma=1, radius=1):
  minima_loc = np.zeros((len(wvn_range), len(kappa_range)))
  minima_E = np.zeros((len(wvn_range), len(kappa_range)))
  for (i,wvn) in enumerate(wvn_range):
    a_s, surface_area_potential, curvature_potential, potential = integrate(wavenumber, 1, precision, gamma, radius)
    # integrate for surface area, curvature terms once at arbitrary kappa=1
    #different linear combinatios are the potentials at different kappa
    for (j,kappa) in enumerate(kappa_range):
      print(wvn, kappa)
      potential = [gamma*x+kappa*y for (x,y) in zip(surface_area_potential, curvature_potential)]
      min_ = min(potential)
      index= potential.index(min_)
      a_min = (a_s[0])+index*precision
      loc, energy = numerical_minimize(wvn, kappa, precision =precision)
      minima_loc[i,j] = abs(loc)
      minima_E[i,j] = energy
  print(minima_loc)
  print(minima_E)
  #save
  loc_df = pd.DataFrame(minima_loc, columns=kappa_range, index=wvn_range)
  energy_df = pd.DataFrame(minima_E, columns=kappa_range, index=wvn_range)
  loc_df.to_csv("numerical_minima_loc.csv")
  loc_df.to_csv("numerical_minima_energy.csv")
  #plot
  plt.imshow(loc_df)
  plt.savefig("numerical_minima_loc.png")
  plt.close()
  plt.imshow(energy_df)
  plt.savefig("numerical_minima_energy.png")
  plt.close()


if __name__ == "__main__":
  wavenumber=2
  kappa=0
  kappa_gaussian = 0
  graph(wavenumber, kappa)
  #min_loc, min_energy = numerical_minimize(wavenumber, kappa, precision = .001)
  #print("wavenumber", wavenumber, "kappa", kappa, ": min is at ", min_loc, " with energy ", min_energy)
  wvn_range = np.arange(0.05, 2, .05)
  kappa_range = np.arange(0, .5, .03)
  #wavenumber_kappa_loop(wvn_range, kappa_range, precision=.01)

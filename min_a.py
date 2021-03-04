import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import pandas as pd
import seaborn as sb

from surfaces_and_fields import system_cylinder

def minimize(stepsize, wavenumber, kappa):
  surface_energys=[]
  field_bending_energys=[]
  total_energys=[]
  for a in np.arange(0,1,stepsize):
      cylinder = system_cylinder.Cylinder(wavenumber=wavenumber, radius=1, kappa=kappa)
      surface_energy = cylinder.calc_surface_energy(amplitude=a)
      #field_bending_energy = cylinder.calc_field_bending_energy(amplitude=a, Cnsquared=C*n*n, magnitudesquared=alpha**2/(u**2))
      total_energy = surface_energy#+field_bending_energy
      surface_energys.append(surface_energy)
      #field_bending_energys.append(field_bending_energy)
      total_energys.append(total_energy)
  minimum = min(total_energys)
  min_a = total_energys.index(minimum)*stepsize
  return(min_a,minimum)

if __name__=="__main__":
  n=1
  u=1
  r=1
  kappa=1
  wavenumber = .1
  alpha=-1

  #show the potential for a case where first order energy difference and this 
  # (numerical integral,) both of potential with ideal straight field, disagree

  
  
  results_a = defaultdict(dict)
  results_energy = defaultdict(dict)
  for k in np.arange(0.01, 1.2,.05):
    length = 2*math.pi / k
    for kappa in np.arange(0, .45, .02):
       print(k,kappa)
       min_a, minimum_e= minimize(.01, wavenumber=k, kappa=kappa)
       results_a[k][kappa]= min_a
       results_energy[k][kappa] = minimum_e/length #energy per unit length
  #print(results_a, results_energy)
  data=pd.DataFrame.from_dict(results_a)
  data_e = pd.DataFrame.from_dict(results_energy)
  data.to_csv( "min_a_stkappa.csv")
  data_e.to_csv("min_e_stkappa.csv")
  sb.heatmap(data)
  plt.show()
  

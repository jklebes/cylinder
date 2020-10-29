import scipy
import numpy as np 
import matplotlib.pyplot as plt
import math
import pandas as pd
try:
  import surfaces_and_fields.system_cylinder as system_cylinder
except ModuleNotFoundError:
  #differnt if this file is run ( __name__=="__main__" )
  #then relatve import:
  import cylinder.surfaces_and_fields.system_cylinder as system_cylinder


if __name__ == "__main__":
  kappa=0
  kappa_gaussian = 0
  intrinsic_curvature = 0
  wvn_range = np.arange(0.01, 1.5, .1)
  a_range=np.arange(-.9,.9, 0.01)
  #for a range of |a|, plot E(k) for bare cylinders
  for k in wvn_range:
    #make a cylinder object of this aspect ratio
    cylinder = system_cylinder.Cylinder()
    #query total energy for a range of |a|
    for a in a_range:
      energy= 5
      energy_row.append(energy)
    #save energy divided by length
    section_length = 2*math.pi/k
    df.add(row)
  
  #save energy per unit length to 
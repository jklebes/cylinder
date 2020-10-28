import scipy
import numpy as np 
import matplotlib.pyplot as plt
import math
import seaborn as sb
import pandas as pd
import surfaces_and_fields.system_cylinder as system_cylinder


if __name__ == "__main__":
  kappa=0
  radius=1
  intrinsic_curvature = 0
  wvn_range = np.arange(0.8, 1.405, .01)
  a_range=np.arange(-.9,.9005, 0.01)
  #for a range of |a|, plot E(k) for bare cylinders
  df = pd.DataFrame(columns=[round(a,5) for a in a_range])
  energy_2D=[]
  for k in wvn_range:

    section_length = 2*math.pi/k
    #make a cylinder object of this aspect ratio
    cylinder = system_cylinder.Cylinder(wavenumber=k, radius=radius, 
                                        kappa=kappa, intrinsic_curvature=intrinsic_curvature)
    energy_row=[]
    #query total energy for a range of |a|
    for a in a_range:
      energy= cylinder.calc_surface_energy(amplitude=a)
      energy_row.append(energy/section_length)
    #save energy divided by length
    energy_2D.append(energy_row)
  df= pd.DataFrame(np.array(energy_2D), index=[str(round(k,5)) for k in wvn_range], 
                                    columns=[str(round(a,5)) for a in a_range])
  print(df)
  #save energy per unit length to 
  #df.to_csv()
  heatmap=sb.heatmap(df, xticklabels=16,  yticklabels=16, cbar_kws={'label': 'Energy per unit length'})
  plt.title("Energy of membrane as a function of wavenumber\n and amplitude of modulation")
  plt.xlabel("amplitude a")
  plt.ylabel("wavenumber k")
  plt.savefig("baremembraneenergy_.png")
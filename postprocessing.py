import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

#plot a time series
def plot_timeseries(data):
  xs = list(data.index)
  colnames = list(data.columns.values)
  print(colnames)
  colnames = [ "abs_c_1", "abs_c_-1"]#, "abs_amplitude"]
  for name in colnames:
    #print(xs, data.loc[:name])
    plt.plot(xs, data.loc[:,name], label=name)
  #plt.xscale('log')
  plt.legend()
  plt.show()

#statistics of a time series

#replot heatmap
def replot_heatmap(data, outdir, title):
  sb.heatmap(data, xticklabels=5, yticklabels=7,cmap ="viridis")
  plt.title("mean perturbation amplitude |a|")
  plt.ylabel("wavenumber k")
  plt.xlabel("bending rigidity kappa_b")
  plt.savefig(os.path.join(outdir, title))

if __name__=="__main__":

  dir_ = os.path.join("out","exp-2020-05-10-16-44-13" )
  file_ = "ncoeffs1_fsteps5_other.csv"
  data = pd.read_csv(os.path.join(dir_,file_), index_col=0)
  #print(data)
  plot_timeseries(data)

  #dir_ = os.path.join("out", "wavenumber-kappa")
  #file_ = "wavenumber_kappa_abs_amplitude_.csv" 
  #data = pd.read_csv(os.path.join(dir_,file_), index_col=0)
  #print(data)
  #replot_heatmap(data=data,outdir=dir_, title = "abs_amplitudxe.png")

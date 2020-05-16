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
  ax=sb.heatmap(data, xticklabels=3, yticklabels=3,cmap ="hot") #viridis and hot
  plt.title("variance field Fourier component 0")
  plt.xlabel("C")
  plt.ylabel("amplitude of surface curve")
  ax.invert_yaxis()
  plt.savefig(os.path.join(outdir, title))

if __name__=="__main__":

  dir_ = os.path.join("out","fixed-amplitude-nc1-withabs-relphase" )
  file_ = "amplitude_C_cov_c0_.csv"

  data = pd.read_csv(os.path.join(dir_,file_), index_col=0)
  print(data)
  replot_heatmap(data=data,outdir=dir_, title = "cov_c0.png")

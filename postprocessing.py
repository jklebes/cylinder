import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import cmath

def phase_histogram(data):
  pass

#plot a time series
def plot_timeseries(data, data2, title="timeseries.png"):
  xs = list(data.index)
  colnames = list(data.columns.values)
  colnames2 = list(data2.columns.values)
  print(colnames)
  select = ['c_0']
  for name in colnames2:
    if any([i in name for i in select]):
      print(xs[35], data2.loc[35, name])
      plt.scatter(xs, data2.loc[:,name], label=name)
  for name in colnames:
    if any([i in name for i in select]):
      #print([complex(x).real for x in data.loc[:,name]])
      plt.scatter(xs, [cmath.phase(complex(x)) for x in data.loc[:,name]], label=name+'_phase')
  plt.legend()
  plt.show()
  #plt.savefig(title)

#statistics of a time series

#replot heatmap
def replot_heatmap(data, outdir, title):
  ax=sb.heatmap(data, xticklabels=1, yticklabels=1,cmap ="hot") #viridis and hot
  plt.title("variance amplitude")
  plt.xlabel("alpha")
  plt.ylabel("wavenumber/r_0")
  #ax.invert_yaxis()
  plt.savefig(os.path.join(outdir, title))

if __name__=="__main__":

  dir_ = os.path.join("out","wavenumber-alpha-best" )
  file_ = "wavenumber_amplitude_variance_.csv"
  #file2_ = "ncoeffs6_fsteps1_other.csv"

  data = pd.read_csv(os.path.join(dir_,file_), index_col=0)
  data = data.applymap(lambda x : complex(x).real)
  #data2 = pd.read_csv(os.path.join(dir_,file2_), index_col=0)
  print(data)
  replot_heatmap(data=data,outdir=dir_, title = "variance_a.png")

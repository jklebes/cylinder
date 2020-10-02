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
  select_all = True
  #select = ['c_0']
  #for name in colnames2:
    #if select_all or any([i in name for i in select]):
      #print(xs[35], data2.loc[35, name])
      #plt.scatter(xs, data2.loc[:,name], label=name, s=1)
  for name in colnames:
    if select_all or any([i in name for i in select]):
      #print([complex(x).real for x in data.loc[:,name]])
      plt.scatter(xs, [abs(complex(x)) for x in data.loc[:,name]], label=name+'_abs', s=1)
      plt.scatter(xs, [complex(x).real for x in data.loc[:,name]], label=name+'_real', s=1)
      if "amplitude" not in name:
        plt.scatter(xs, [cmath.phase(complex(x)) for x in data.loc[:,name]], label=name+'_phase', s=1)
        plt.scatter(xs, [complex(x).imag for x in data.loc[:,name]], label=name+'_img', s=1)
  plt.legend()
  #plt.show()
  plt.xscale('log')
  plt.savefig(title)
  plt.close()

#statistics of a time series

#replot heatmap
def replot_heatmap(data, outdir, title):
  #round labels
  data.index = [round(i,3) for i in data.index]
  ax=sb.heatmap(data, xticklabels=1, yticklabels=1,cmap ="viridis", fmt='.2g')#, vmax=1, vmin=0) #viridis and hot
  plt.title("|a| as a function of k, H_0")
  plt.xlabel("H_0")
  plt.ylabel("k")
  #ax.invert_yaxis()
  plt.savefig(os.path.join(outdir, title))
  plt.close()

if __name__=="__main__":

  dir_ = os.path.join("out","wavenumber-alpha-withC" )
  #file_ = "wavenumber_amplitude_.csv"
  file_ = "wvn0.5_alpha0.0.csv"
  file2_ = "wvn0.5_alpha0.0_other.csv"

  data = pd.read_csv(os.path.join(dir_,file_), index_col=0)
  #data = data.applymap(lambda x : complex(x).real)
  data2 = pd.read_csv(os.path.join(dir_,file2_), index_col=0)
  print(data)
  #replot_heatmap(data=data,outdir=dir_, title = "a_wvn_alpha_withC.png")
  plot_timeseries(data,data2, title=os.path.join(dir_,"timeseries.5-0aall_log.png"))

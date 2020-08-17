import pandas as pd
import os
from collections import defaultdict
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colorsys

amplitude_scaling = 10
z_coords = np.arange(0, 2*np.pi,.05)
theta_coords = np.arange(0, 2*np.pi, 0.1)
fig = plt.figure()
ax = plt.axes(xlim=(0, len(z_coords)), ylim=(-50, len(theta_coords)))
data = np.zeros((len(theta_coords), len(z_coords)))
im=plt.imshow(data, vmin=0, vmax=1.5)
line_amplitude, = ax.plot(range(len(z_coords)), np.sin(z_coords), color='g', linewidth=10)
line_amplitude2, = ax.plot(range(len(z_coords)), np.sin(z_coords), color='g', linewidth=10)

running_avg = None
stepcounter = 0

def init():
  im.set_data(np.zeros((len(theta_coords), len(z_coords))))
  line_amplitude.set_ydata([np.nan] * len(z_coords))
  line_amplitude2.set_ydata([np.nan] * len(z_coords))
  return im, line_amplitude, line_amplitude2

def animate(i):
  complex_snapshot=dict([])
  for key in complex_series:
    complex_snapshot[key] = complex_series[key][i]
  data = visualize_snapshot(complex_snapshot)
  im.set_array(data)
  #print(data)
  line_amplitude.set_ydata(get_amplitude_line(i,z_coords))
  line_amplitude2.set_ydata(get_amplitude_line2(i,z_coords))
  return im, line_amplitude, line_amplitude2

def file_to_df(data_file):
  df = pd.read_csv(data_file, index_col=0)
  return df

def get_amplitude_line(i,zs):
  return [-15+amplitude_scaling*amplitude_series[i]*math.sin(z) for z in zs]
def get_amplitude_line2(i,zs):
  return [-35-amplitude_scaling*amplitude_series[i]*math.sin(z) for z in zs]


def get_complex_series(data):
  len_series = len(data.index.values) # length in time
  time_series = defaultdict(lambda: np.zeros(len_series)) # dict (z_index, th_index) : time series of values of c_{z th}
  print(data)
  for column_name in data.columns.values:
    print(column_name)
    if "sampling_width" in column_name or "energy" in column_name:
      pass
    elif "amplitude" in column_name or "ampiltude" in column_name or ("param_0" in column_name and "abs" not in column_name and "squared" not in column_name):
      print(type(data[column_name][0]))
      if isinstance(data[column_name][0], str):
        amplitude_series = [complex(a).real for a in data[column_name]]
      else:
        amplitude_series = data[column_name]
        print("read amplitude_series" , amplitude_series)
    elif "img" in column_name:
      coeff_number = int(column_name.split("_")[2])
      print("found img part of c ", coeff_number)
      time_series[coeff_number] += np.array([0+float(value)*1j for value in data[column_name] ])
    elif "real" in column_name:
      coeff_number = int(column_name.split("_")[2])
      print("found real part of c ", coeff_number)
      time_series[coeff_number] += np.array([float(value) +0j for value in data[column_name] ])
    elif "c_" in column_name or (("coeff" in column_name or "param_" in column_name) and "abs" not in column_name and "squared" not in column_name): 
      # index math required to translate from paramnumber (1 to ... ) to z_index, th_index in range -m..m x -n ... n
      #the dict coeff_number handles the conversion
      if "param" in column_name:
        coeff_number = coeff_numbers[int(column_name.split("_")[1])] # lookup from dict given for this file, if coeffs are labeled 0 to n*m
      elif "coeff" in column_name:
        coeff_number = (int(column_name.split("_")[1]), int(column_name.split("_")[2]))
      time_series[coeff_number] = [complex(value) for value in data[column_name]]
  return time_series, amplitude_series

def visualize_snapshot(complex_snapshot):
  #get sum of c_j * e^ijz for a range of z values between 0 and 2pi
  zs = z_coords
  ths = theta_coords
  field = np.zeros((len(ths), len(zs))).tolist() #complex field, real parts, img parts as a function of z
  reals = []
  imgs = []
  for (i,z) in enumerate(zs):
    for (j,th) in enumerate(ths):
      f = 0+0j #complex value of the field (sum of all field modes) at the point z, th
      real = 0
      img = 0
      for (z_index, th_index) in complex_snapshot:
        f += complex_snapshot[(z_index, th_index)] *( math.cos(z_index*z) + math.sin(z_index*z)*1j)*(math.cos(th_index*th)+ math.sin(th_index*th)*1j)
      field[j][i]=complex_to_rgb(f) # indexes are in this order -> display on screen is --z-->  
  return field

def complex_to_rgb(c):
  a=.3
  h = cmath.phase(c)/(2*math.pi)
  l = (1-a**(abs(c))) *1
  s = 1
  return colorsys.hls_to_rgb(h, l, s)

if __name__=="__main__":
  data_dir = os.path.join("out", "exp-2020-08-17-10-18-00")
  #data_dir = "."
  data_file = os.path.join(data_dir, "ncoeffs(0, 0)_fsteps1.csv")
  n,m  =  (3,1) # values n (z index goes from -n to n) , m (theta index goes from -m to +m) - can be found in data directory's notes.csv
  # if it's 1D data put 0 as second element
  # conversion key from param number to (z_index, theta_index)
  # there are this many complex parameters:
  num_complex_params = (2*n+1)*(2*m+1)
  coeff_numbers = dict([(i, (-n+( (i-1) % (2*n+1)  ) , -m+( (i-1) // (2*n+1) ))  ) for i in range(1,num_complex_params+1)])
  data = file_to_df(data_file)
  print("data", data)
  complex_series, amplitude_series = get_complex_series(data)
  #values_vs_time_f, real, img = visualize_snapshot(complex_snapshot)
  #x = np.arange(0, 2*np.pi, 0.01)
  ani = animation.FuncAnimation(fig, animate, interval=400)#,  save_count=50)
  #ax.set_ylim([-4.3,2])
  #ax.set_xlim([-.2, 2*math.pi+.2])
  #plt.plot([-1,2*math.pi+1], [0]*2, color='black')
  plt.yticks([])
  plt.xticks=([])
  #plt.legend(loc=3)
  plt.xlabel('z')
  plt.xticks=([])
  plt.show()
  #ani.save("2Danimation31.mp4")
  plt.close()
  matrixr = [ (0 + i*1j) for i in np.arange(-2, 2, .01)]
  matrix = [[complex_to_rgb((i + x)) for i in matrixr] for x in np.arange(-2,2,.01)]
  plt.imshow(matrix)
  plt.xticks=([])
  plt.yticks=([])
  plt.axis('off')
  plt.savefig("cmoplex.png")

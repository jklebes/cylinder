import pandas as pd
import os
from collections import defaultdict
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line_field, = ax.plot(x, np.sin(x), color='b', label = "field magnitude")
line_field_avg, = ax.plot(x, np.sin(x), color='black', label="running average")
line_amplitude, = ax.plot(x, np.sin(x), color='g', linewidth=10)
line_amplitude2, = ax.plot(x, np.sin(x), color='g', linewidth=10)

running_avg = None
stepcounter = 0

def init():  # only required for blitting to give a clean slate.
  global running_avg
  line_field.set_ydata([np.nan] * len(x))
  running_avg = np.zeros(len(x))
  line_field_avg.set_ydata([np.nan] * len(x))
  line_amplitude.set_ydata([np.nan] * len(x))
  line_amplitude2.set_ydata([np.nan] * len(x))
  return line_field, line_field_avg, line_amplitude, line_amplitude2

def animate(i):
  line_field.set_ydata(get_magnitude_line(i,x)) 
  line_field_avg.set_ydata(get_avg_line(i,x)) 
  line_amplitude.set_ydata(get_amplitude_line(i,x))  
  line_amplitude2.set_ydata(get_amplitude_line2(i,x))  
  return line_field, line_field_avg, line_amplitude, line_amplitude2

def file_to_df(data_file):
  df = pd.read_csv(data_file, index_col=0)
  return df

def get_amplitude_line(i,zs):
  return [-1.2+amplitude_series[i]*math.sin(z) for z in zs]
def get_amplitude_line2(i,zs):
  return [-3.2-amplitude_series[i]*math.sin(z) for z in zs]

def get_avg_line(i,zs):
  return running_avg

def get_magnitude_line(i, zs):
  global running_avg
  global stepcounter
  complex_snapshot=dict([])
  for key in complex_series:
    complex_snapshot[key] = complex_series[key][i]
  line = []
  for z in zs:
    f = 0+0j
    for index in complex_snapshot:
      f+= complex_snapshot[index]*(math.cos(index*z)+ math.sin(index*z)*1j)
      #if z==.01:
        #print(index, math.cos(index*0.01), complex_snapshot[index], f.imag)
    line.append(abs(f))
  stepcounter +=1
  running_avg *= (stepcounter -1) / float(stepcounter)
  running_avg += np.array(line)/float(stepcounter) 
  return line


def get_complex_series(data):
  len_series = len(data.index.values)
  time_series = defaultdict(lambda: np.zeros((len_series), dtype='complex128'))
  for column_name in data.columns.values:
    if "img" in column_name:
      coeff_number = int(column_name.split("_")[2])
      print("found img part of c ", coeff_number)
      time_series[coeff_number] += np.array([0+float(value)*1j for value in data[column_name] ])
    elif "real" in column_name:
      coeff_number = int(column_name.split("_")[2])
      print("found real part of c ", coeff_number)
      time_series[coeff_number] += np.array([float(value) +0j for value in data[column_name] ])
    elif "amplitude" in column_name or "ampiltude" in column_name:
      print(type(data[column_name][0]))
      if isinstance(data[column_name][0], str):
        amplitude_series = [complex(a).real for a in data[column_name]]
      else:
        amplitude_series = data[column_name]
    elif "c" in column_name:
      coeff_number = int(column_name.split("_")[1])
      time_series[coeff_number] = [complex(value) for value in data[column_name]]
  return time_series, amplitude_series

def visualize_snapshot(complex_snapshot):
  #get sum of c_j * e^ijz for a range of z values between 0 and 2pi
  zs = np.arange(0, 2*math.pi, .0002)
  field = [] #complex field, real parts, img parts as a function of z
  reals = []
  imgs = []
  for z in zs:
    f = 0+0j
    real = 0
    img = 0
    for index in complex_snapshot:
      f += complex_snapshot[index] *( math.cos(index*z) + math.sin(index*z)*1j)
    field.append(f)
    reals.append(f.real)
    imgs.append(f.imag)
  #plt.plot(zs, reals, label="real part of field")
  #plt.plot(zs, imgs, label="img part of field")
  #plt.plot(zs, [abs(f) for f in field], label="magnitude of field")
  #plt.plot(zs, [a*math.sin(z) for z in zs], label=("sine with amplitude "+str(a)))
  #plt.legend()
  #plt.show()
  return(field, reals, imgs)



if __name__=="__main__":
  data_dir = os.path.join("out", "nc6-on-frozen")
  data_file = os.path.join(data_dir, "ncoeffs6_fsteps1.csv")
  data = file_to_df(data_file)
  complex_series, amplitude_series = get_complex_series(data)
  #print(complex_series)
  #arbitrary_time = 178
  #for key in complex_series:
    #complex_snapshot[key] = complex_series[key][arbitrary_time]
  #print(complex_snapshot)
  #values_vs_time_f, real, img = visualize_snapshot(complex_snapshot)
  x = np.arange(0, 2*np.pi, 0.01)
  ani = animation.FuncAnimation(fig, animate, init_func=init, interval=5, blit=True, save_count=50)
  ax.set_ylim([-4.3,2])
  ax.set_xlim([-.2, 2*math.pi+.2])
  plt.plot([-1,2*math.pi+1], [0]*2, color='black')
  plt.yticks([0,1,2])
  plt.legend()
  plt.show()

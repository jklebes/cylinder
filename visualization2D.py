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
#line_field_component, = ax.plot(x, np.sin(x), color='brown', label = "field component real")
#line_field_component_img, = ax.plot(x, np.sin(x), color='orange', label = "field component img")
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
  #line_field_component.set_ydata([np.nan] * len(x))
  #line_field_component_img.set_ydata([np.nan] * len(x))
  line_amplitude.set_ydata([np.nan] * len(x))
  line_amplitude2.set_ydata([np.nan] * len(x))
  return line_field, line_field_avg, line_amplitude, line_amplitude2
  #return line_field, line_field_component, line_field_component_img, line_field_avg, line_amplitude, line_amplitude2
  #return line_field, line_amplitude, line_amplitude2

def animate(i):
  line_field.set_ydata(get_magnitude_line(i,x)) 
  #line_field_component.set_ydata(get_component_line(i,x,1)) 
  #line_field_component_img.set_ydata(get_component_line_img(i,x,1)) 
  line_field_avg.set_ydata(get_avg_line(i,x)) 
  line_amplitude.set_ydata(get_amplitude_line(i,x))  
  line_amplitude2.set_ydata(get_amplitude_line2(i,x))  
  return line_field, line_field_avg, line_amplitude, line_amplitude2
  #return line_field, line_field_component, line_field_component_img, line_field_avg, line_amplitude, line_amplitude2
  #return line_field, line_amplitude, line_amplitude2

def file_to_df(data_file):
  df = pd.read_csv(data_file, index_col=0)
  return df

def get_amplitude_line(i,zs):
  return [-1.2+amplitude_series[i]*math.sin(z) for z in zs]
def get_amplitude_line2(i,zs):
  return [-3.2-amplitude_series[i]*math.sin(z) for z in zs]

def get_avg_line(i,zs):
  return running_avg

def get_component_line_img(i, zs, key):
  global running_avg
  global stepcounter
  complex_snapshot= complex_series[key][i]
  line = []
  for z in zs:
    f= complex_snapshot*(math.cos(key*z)+ math.sin(key*z)*1j)
    line.append(f.imag)
  stepcounter +=1
  return line

def get_component_line(i, zs, key):
  global running_avg
  global stepcounter
  complex_snapshot= complex_series[key][i]
  line = []
  for z in zs:
    f= complex_snapshot*(math.cos(key*z)+ math.sin(key*z)*1j)
    line.append(f.real)
  stepcounter +=1
  return line

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
  len_series = len(data.index.values) # length in time
  time_series = defaultdict(lambda: np.zeros(len_series), dtype='complex128') # dict (z_index, th_index) : time series of values of c_{z th}
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
    elif "c_" in column_name or ("param_" in column_name and "abs" not in column_name and "squared" not in column_name): 
      # index math required to translate from paramnumber (1 to ... ) to z_index, th_index in range -m..m x -n ... n
      #the dict coeff_number handles the conversion
      coeff_number = coeff_numbers[int(column_name.split("_")[1])] # lookup from dict given for this file
      time_series[coeff_number] = [complex(value) for value in data[column_name]]
  return time_series, amplitude_series

def visualize_snapshot(complex_snapshot):
  #get sum of c_j * e^ijz for a range of z values between 0 and 2pi
  zs = np.arange(0, 2*math.pi, .02)
  ths = np.arange(0, 2*math.pi, .02)
  field = np.zeros((len(zs), len(ths))) #complex field, real parts, img parts as a function of z
  reals = []
  imgs = []
  for z in zs:
      for th in ths:
        f = 0+0j #complex value of the field (sum of all field modes) at the point z, th
        real = 0
        img = 0
        for (z_index, th_index) in complex_snapshot:
          f += complex_snapshot[index] *( math.cos(z_index*z) + math.sin(z_index*z)*1j)*(math.cos(th_index*th)+ math.sin(th_index*th)*1j)
        field[z,th]=f
  return field



if __name__=="__main__":
  data_dir = os.path.join("out", "2D")
  data_file = os.path.join(data_dir, "wvn0.65_kappa0.0.csv")
  n,m  =  (1,1) # values n (z index goes from -n to n) , m (theta index goes from -m to +m) - can be found in data directory's notes.csv
  # if it's 1D data put 0 as second element
  # conversion key from param number to (z_index, theta_index)
  # there are this many complex parameters:
  num_complex_params = (2*n+1)*(2*m+1)
  coeff_numbers = dict([(i, (-n+( (i-1) % (2*n+1)  ) , -m+( (i-1) // (2*n+1) ))  ) for i in range(1,num_complex_params+1)])
  print("conversion dict", coeff_numbers)
  data = file_to_df(data_file)
  complex_series, amplitude_series = get_complex_series(data)
  #values_vs_time_f, real, img = visualize_snapshot(complex_snapshot)
  #x = np.arange(0, 2*np.pi, 0.01)
  ani = animation.FuncAnimation(fig, animate, init_func=init, interval=40, blit=True, save_count=50)
  ax.set_ylim([-4.3,2])
  ax.set_xlim([-.2, 2*math.pi+.2])
  plt.plot([-1,2*math.pi+1], [0]*2, color='black')
  plt.yticks([0,0.5,1, 1.5, 2])
  plt.legend(loc=3)
  plt.xlabel('z')
  plt.show()
  #ani.save("kept_for_animation.mp4")

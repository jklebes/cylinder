import cmath
import math
import os
import matplotlib.pyplot as plt
import visualization2D as v

#compare whether there are thermal fluctuations for a case with higher and lower C

index = (0,-1)

data_dir = os.path.join("out", "08-19-14")
data_file = os.path.join(data_dir, "a0_C0.1.csv")
n,m  =  (6,1) # values n (z index goes from -n to n) , m (theta index goes from -m to +m) - can be found in data directory's notes.csv
num_complex_params = (2*n+1)*(2*m+1)
coeff_numbers = dict([(i, (-n+( (i-1) % (2*n+1)  ) , -m+( (i-1) // (2*n+1) ))  ) for i in range(1,num_complex_params+1)])
data = v.file_to_df(data_file)
complex_series, amplitude_series = v.get_complex_series(data)
data_file2 = os.path.join(data_dir, "a0_C1.csv")
data2 = v.file_to_df(data_file2)
complex_series2, amplitude_series2 = v.get_complex_series(data2)
plt.close()
index_str = "c_"+ str(index[0])+ str(index[1])
plt.plot([i for i,x in enumerate(complex_series[index])],[abs(x) for x in complex_series[index]], label = "C=.1 simulation")
#plt.plot([i for i,x in enumerate(complex_series[index])],[(x.imag) for x in complex_series[index]], label = "imag part")
#plt.plot([i for i,x in enumerate(amplitude_series)],[math.sqrt(1-(abs(x)**2*.1*.8**2)) for x in amplitude_series], label = "epectedfield from effective alpha")
plt.plot([i for i,x in enumerate(complex_series2[index])],[abs(x) for x in complex_series2[index]], label = "C=1 simulation")
#plt.plot([i for i,x in enumerate(amplitude_series2)],[abs(x) for x in amplitude_series2], label = "amplitude C=1 simulation")
#plt.plot([i for i,x in enumerate(amplitude_series2)],[math.sqrt(1-(abs(x)**2*1)) for x in amplitude_series2], label = "expected field from effective alpha")
plt.legend()
plt.ylabel("|"+index_str+"|")
plt.savefig("./abs_"+index_str+".png")
plt.close()

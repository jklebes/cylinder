import numpy as np
import metropolis_engine as me
import random
import matplotlib.pyplot as plt
#calculating the correlation matrix between two complex numbers

#a simple energy landscape:  two complex numbers like to be aligned. length bounded as 1.
def toy_potential(z,w):
  #definitely produces correlations of z and w
  #can be identified with quartic-quadratic potential plus coupling cross terms (symmetric, noncacelling)
  e = (abs(z)-1)**4 + (abs(w)-1)**4 - z*w.conjugate() - w*z.conjugate()
  return e.real

def toy_potential_uncorrelated(z,w):
  e = (abs(z)-1)**4 + (abs(w)-1)**4 
  return e.real

#example metropolis algorithms
width=0.1
z=0+0j
w=0+0j
zs=[z]
ws=[w]
m=me.MetropolisEngine({-1:0+0j, 0:z,1:w},0, temp=.5)
old_energy = toy_potential(z,w)
n=1000
for i in range(n):
  z_prop=z+m.gaussian_complex(width)
  w_prop=w+m.gaussian_complex(width)
  new_energy = toy_potential(z_prop, w_prop)
  if m.metropolis_decision(old_energy, new_energy):
    z=z_prop
    w=w_prop
    old_energy=new_energy
  zs.append(z)
  ws.append(w)
plt.scatter([z.real for z in zs], [z.imag for z in zs], s=5, marker='x', label='z')
plt.scatter([w.real for w in ws], [w.imag for w in ws], s=5, marker='.', color='r', label='w')
plt.legend()
plt.savefig("complex_correlated.png")

#try correlation matrix
mean_z = sum(zs)/n
mean_w = sum(ws)/len(ws)
print("after", n, "steps")
print("mean z", mean_z, "mean w", mean_w)
mean = np.matrix([mean_z, mean_w])
corr = 0
for i in range(n):
  state=np.matrix([zs[i], ws[i]])
  corr += state.conjugate().transpose()@state
corr -= n* mean.conjugate().transpose()@mean
corr /= n-1
print("correlation", corr.round(3))
#### result ####
# covariance matrix =
# ( (3.7+0j, 3.7+0j), (3.7+0j, 3.7+0j)) for large n

#interpretation: real part measures alignement, img part cross product

#next - how to draw from gaussian distribution with this complex correlation matrix to step the two complex parameters?

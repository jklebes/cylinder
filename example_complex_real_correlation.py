import numpy as np
import metropolis_engine as me
import random
import matplotlib.pyplot as plt
#calculating the correlation matrix between a complex number and a number whose magnitude only should matter.  Represented as a real number, but arbitrary 'real' direction on complex plane should not have effect. 


def toy_potential(a,z):
  e = a**2 + (abs(z)-1)**4 - abs(a)*z*z.conjugate()
  return e.real

def toy_potential_2(a,z):
  """bad/difficult example because only real part of z matters in coupling
  no terms like this expected in actual system"""
  e = a**2 + (abs(z)-1)**4 - abs(a)*z-abs(a)*z.conjugate()
  return e.real

def toy_potential_uncorrelated(a,z):
  e = (abs(a)-1)**4 + (abs(z)-1)**4 
  return e.real

#example metropolis algorithms
width=0.1
z=0+0j
a=0
zs=[z]
a_s=[abs(a)]
abs_zs=[abs(z)]
m=me.MetropolisEngine({0:z}, a, temp=.5)
old_energy = toy_potential(a,z)
n=5000000
for i in range(n):
  z_prop=z+m.gaussian_complex(width)
  a_prop=a+random.gauss(0,width)
  new_energy = toy_potential(a_prop, z_prop)
  if m.metropolis_decision(old_energy, new_energy):
    a=a_prop
    z=z_prop
    old_energy=new_energy
  zs.append(z)
  abs_zs.append(abs(z))
  a_s.append(abs(a))

#try correlation matrix
mean_z = sum(zs)/n
mean_a = sum(a_s)/n
print("after", n, "steps")
print("mean a", mean_a, "mean z", mean_z)
mean = np.matrix([mean_a, mean_z])
corr = 0
for i in range(n):
  state=np.matrix([a_s[i], zs[i]])
  corr += state.conjugate().transpose()@state
corr -= n* mean.conjugate().transpose()@mean
corr /= n-1
print("correlation", corr.round(3))
###result ###
# something like 
# (( 0.5+0j, .006 + .02j),
# (c.c. , 7+0j))


#real correlation matrix abs only
mean_z_abs = sum(abs_zs)/n
mean_a = sum(a_s)/n
print("after", n, "steps")
print("mean a", mean_a, "mean z_abs", mean_z_abs)
mean = np.matrix([mean_a, mean_z_abs])
corr = 0
for i in range(n):
  state=np.matrix([a_s[i], abs_zs[i]])
  corr += state.transpose()@state
corr -= n* mean.transpose()@mean
corr /= n-1
print("correlation in magnitudes", corr.round(3))
###result ###

#next - how to draw from gaussian distribution with this complex correlation matrix to step the two complex parameters?

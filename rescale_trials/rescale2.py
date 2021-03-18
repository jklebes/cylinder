import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import cmath

def getlambda(maxwavevector, Lx):
  Ly=Lx
  #this codes up functions defining vectors lambda
  # which retrieve real/img parts at boundary points mui
  lambdai=np.zeros((maxwavevector[0]+1, maxwavevector[1]+1), dtype=complex)
  #the version that works with standard fourier decomposition Psi = sum of Psi_q e^(iqx/(2l))
  
  for nx in range( maxwavevector[0]+1):
    for ny in range( maxwavevector[1]+1):
      try:
        lambdai[nx, ny]=(0+1j)*Lx*(-1+cmath.rect(1,math.pi*nx*2/Lx))/nx #integral
      except ZeroDivisionError:
        lambdai[nx, ny]= 2*math.pi
      try:
        lambdai[nx,ny]*=(0+1j)*Ly*(-1+cmath.rect(1,math.pi*ny*2/Ly))/ny
      except ZeroDivisionError:
        lambdai[nx, ny]*= 2*math.pi
  #print(lambdai.real)
  return lambdai.real 



def get_VWmatrix(field, maxwavevector, alpha, n, C, l, Lx):
  lambdas=dict([(i, getlambda(maxwavevector, Lx)) for i in range(1)])
  #print(lambdas)
  Aprime = np.zeros(((maxwavevector[0]+1)* (maxwavevector[1]+1)+1, (maxwavevector[0]+1)* (maxwavevector[1]+1)+1))
  for qx in range(maxwavevector[0]+1):
    for qy in range(maxwavevector[1]+1):
        Aprime[qx*(maxwavevector[0]+1)+qy, qx*(maxwavevector[0]+1)+qy] = (l*l*alpha+n**2 *C*(qx**2+qy**2))
  for i in range(1):
    for qx in range(maxwavevector[0]+1):
      for qy in range(maxwavevector[1]+1):
        Aprime[qx*(maxwavevector[0]+1)+qy, (maxwavevector[1]+1)*(maxwavevector[0]+1)+i] = .5*lambdas[i][qx, qy]
        Aprime[(maxwavevector[1]+1)*(maxwavevector[0]+1)+i,qx*(maxwavevector[0]+1)+qy] = .5*lambdas[i][qx, qy]
  #print(Aprime)
  return np.linalg.inv(Aprime)


Q=(10,10)
alpha=1
n=1
C=5
u=1
l=1
cutoff_wavelength=1
exampleBC=(1+0j)
coeffs_real = get_VWmatrix(field='imag', maxwavevector=Q, alpha=alpha, n=n, C=C, l=l)
l_s=[2, 3,4,6,8,14]
alphas_rho=[]
Cs_rho=[]
alphas_th=[]
Cs_th=[]
for l in l_s:
  Q=(int(l/cutoff_wavelength),int(l/cutoff_wavelength))
  print(Q)
  #coeffs_real = get_VWmatrix(field='imag', maxwavevector=Q, alpha=a, n=n, C=C, l=l)
  C_par=C
  if alpha>0:
    try:
      rho_rms = min(math.sqrt(.5/math.sqrt(u)), math.sqrt(1/(2*alpha)))
    except ZeroDivisionError:
      rho_rms =math.sqrt(.5/math.sqrt(u))
  else:
    try:
      rho_rms = max(math.sqrt(.5/math.sqrt(u)), math.sqrt(-alpha/u))
    except ZeroDivisionError:
      rho_rms =math.sqrt(.5/math.sqrt(u))
  if alpha<0:  
    alpha *=-28
  C_perp = rho_rms**2 * C #converts from energy/len^2 units to energy/radians^2
  coeffs_real_rho = get_VWmatrix(field='real', maxwavevector=Q, alpha=alpha, n=n, C=C_par, l=l, Lx=Lx)
  #det = np.linalg.det(coeffs_real_rho)
  alpha_=-.5*coeffs_real_rho[-1,-1]
  alphas_rho.append(alpha_)#+4*negative2C_)
  #coeffs_real_th = get_VWmatrix(field='real', maxwavevector=Q, alpha=0, n=n, C=C_perp, l=l)
  #alpha_plusC_=-.5*coeffs_real_th[-1,-1]
  #negative2C_=-.5*coeffs_real_th[-1, -2]
  #alphas_th.append(alpha_plusC_+4*negative2C_)
  #Cs_th.append(-.5*negative2C_)
plt.plot( [l*l for l in l_s], [a/(l*l) for (a,l) in zip(alphas_rho, l_s)], label="alpha'")
#plt.plot(q_s, Cs_rho, label="C")
#plt.plot(a_s, alphas_th, label="a'")
#plt.plot(q_s, Cs_th, label="C_perp")
plt.legend()
plt.show()
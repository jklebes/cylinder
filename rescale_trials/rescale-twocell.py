import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import cmath
import scipy.integrate
import pandas as pd

def getlambda(i, field, maxwavevector):
  sqrt2=math.sqrt(2)
  #this codes up functions defining vectors lambda
  # which retrieve real/img parts at boundary points mui
  lambdai=np.zeros((2*maxwavevector[0]-1, 2*maxwavevector[1]-1), dtype=complex)
  #the version that works with standard fourier decomposition Psi = sum of Psi_q e^(iqx/(2l))
  if i==2:
    for (nx,qx) in enumerate(range(-maxwavevector[0]+1, maxwavevector[0])):
      for (ny,qy) in enumerate(range(-maxwavevector[1]+1, maxwavevector[1])):
        ans = (-1j)**(qy//2)*(-1)**(qy//2)
        if qy%2==1:
            ans*=(1+1j)/sqrt2
        lambdai[nx, ny]=ans
  elif i==4:
    for (nx,qx) in enumerate(range(-maxwavevector[0]+1, maxwavevector[0])):
      for (ny,qy) in enumerate(range(-maxwavevector[1]+1, maxwavevector[1])):
        ans = (-1j)**(qx//2)*(-1)**(qx//2)
        if qx%2==1:
            ans*=(1+1j)/sqrt2
        ans*=(-1j)**qy*(-1)**qy
        lambdai[nx, ny]=ans
  elif i==0:
    for (nx,qx) in enumerate(range(-maxwavevector[0]+1, maxwavevector[0])):
      for (ny,qy) in enumerate(range(-maxwavevector[1]+1, maxwavevector[1])):
        if qx!=0 and qy!=0:
          lambdai[nx,ny] = -4/math.pi**2 *minus1exphalfpi(qx)*minus1exphalfpi(qy)*(1+
                           minus1exphalfpi(qy))/(qx*qy)
        elif qx!=0:
          lambdai[nx,ny] = -2j/math.pi**2 * minus1exphalfpi(qx) / qx
        elif qy!=0:
          lambdai[nx,ny] = -2j/math.pi**2 * minus1exphalfpi(qy) *(1+
                           minus1exphalfpi(qy))/ qy
        else:
          lambdai[nx,ny]=1
  elif i==3:
    for (nx,qx) in enumerate(range(-maxwavevector[0]+1, maxwavevector[0])):
      for (ny,qy) in enumerate(range(-maxwavevector[1]+1, maxwavevector[1])):
        ans = (-1j)**(qx//2)*(-1)**(qx//2)
        if qx%2==1:
            ans*=(1+1j)/sqrt2
        lambdai[nx, ny]=ans
  elif i==1:
    for (nx,qx) in enumerate(range(-maxwavevector[0]+1, maxwavevector[0])):
      for (ny,qy) in enumerate(range(-maxwavevector[1]+1, maxwavevector[1])):
        if qx!=0 and qy!=0:
          lambdai[nx,ny] = -4/math.pi**2 *minus1exphalfpi(qx)*minus1exphalfpi(qy)/(qx*qy)
        elif qx!=0:
          lambdai[nx,ny] = -2j/math.pi**2 * minus1exphalfpi(qx) / qx
        elif qy!=0:
          lambdai[nx,ny] = -2j/math.pi**2 * minus1exphalfpi(qy)/ qy
        else:
          lambdai[nx,ny]=1
  if field=='real':
      return lambdai
  elif field=='imag':
      return lambdai
  else:
      print("choose field='real' or 'imag'")
      return None

def minus1exphalfpi(n):
    #(-1 + e^{i/2 n pi})
    if n%4==0:
      return 0
    elif n%2==0:
      return -2
    elif n%4==1:
      return -1+1j
    else:
      return -1-1j

def fill_A_matrix(alpha, n, C, lx,ly, qx_1, qx_2, qy_1, qy_2):
  xdiff = qx_2-qx_1
  ydiff=qy_2-qy_1
  factor = alpha  + C *n**2*(qx_1*qx_2/(4*lx**2)+qy_1*qy_2/(4*ly**2)) 
  if xdiff==0 and ydiff==0:
    return 16*lx*ly*math.pi**2*factor
  elif ydiff==0:
    ans= 0
    if xdiff%2==1:
      ans+= 2*lx*ly*math.pi*8.0j/xdiff *factor
    return ans
  elif xdiff==0:
    ans= 0
    if ydiff%2==1:
      ans+= 2*lx*ly*math.pi*4.0j/ydiff *factor
    return ans
  else:
    ans= -16
    .0 * lx * ly * factor
    ans/= (xdiff*ydiff)
    if ydiff%2==1:
      ans*=-2.0
    else:
      ans =0 
    if xdiff%2==1:
      ans*=-2.0
    else:
      ans=0
    #if (xdiff+ydiff)%2==1:
      #ans*= -1
      #print(ans)
    #print(ans)
    return ans 
  


def get_VWmatrix(field, maxwavevector, alpha, n, C, lx,ly):
  n_lambdas=1
  lambdas=dict([(i, getlambda(i, field, maxwavevector)) for i in range(n_lambdas)])
  Aprime = np.zeros(((2*maxwavevector[0]-1)* (2*maxwavevector[1]-1)+n_lambdas, (2*maxwavevector[0]-1)* (2*maxwavevector[1]-1)+n_lambdas), dtype=complex)
  #print(Aprime)
  for (nx_1, qx_1) in enumerate(range(-maxwavevector[0]+1,maxwavevector[0])):
    for (ny_1,qy_1) in enumerate(range(-maxwavevector[1]+1,maxwavevector[1])):
      for (nx_2, qx_2) in enumerate(range(-maxwavevector[0]+1,maxwavevector[0])):
        for (ny_2,qy_2) in enumerate(range(-maxwavevector[1]+1,maxwavevector[1])):
          #print(nx_1, ny_1, nx_2, ny_2)
          #print((nx_1),'*',(2*maxwavevector[0]-1),'+',ny_1, (nx_2),'*',(2*maxwavevector[0]-1),'+',ny_2)
          #print((nx_1)*(2*maxwavevector[0]-1)+ny_1, (nx_2)*(2*maxwavevector[0]-1)+ny_2)
          Aprime[(nx_1)*(2*maxwavevector[1]-1)+ny_1, (nx_2)*(2*maxwavevector[1]-1)+ny_2] = fill_A_matrix(alpha, n, C, lx,ly, qx_1, qx_2, qy_1, qy_2)
  #print(Aprime.shape)
  #print(Aprime)
  A=Aprime[:-5, :-5].round(3)
  #print(np.diag(A))
  #print("inv A",np.linalg.inv(A)[maxwavevector[0]-2:maxwavevector[0]+3,maxwavevector[1]-2:maxwavevector[1]+3])
  Aprime*=2
  #print("Hermitian part is ", (A + A.T.conj())/2, "and is positive defini", np.all(np.linalg.eigvals(A) > 0))
  for i in range(n_lambdas):
    for (nx,qx) in enumerate(range(-maxwavevector[0]+1,maxwavevector[0])):
      for (ny,qy) in enumerate(range(-maxwavevector[1]+1,maxwavevector[1])):
        Aprime[(nx)*(2*maxwavevector[1]-1)+ny, (2*maxwavevector[1]-1)*(2*maxwavevector[0]-1)+i] = 1j*(lambdas[i][nx, ny]).conjugate()
        Aprime[(2*maxwavevector[1]-1)*(2*maxwavevector[0]-1)+i,(nx)*(2*maxwavevector[1]-1)+ny] = 1j*(lambdas[i][nx, ny])
  #print("Hermitian part is ", (Aprime + Aprime.T.conj())/2, "and is positive definite",] np.all(np.linalg.eigvals(Aprime) > 0))
  #print('corner',Aprime[0,0])
  #print(Aprime)
  #print('det',np.linalg.det(Aprime))
  return np.linalg.inv(Aprime)

def sample_plot(sigma2_1, mu_1, sigma2_2, mu_2):
  #show points in unnormalized 2D gaussian
  point_xs=[]
  point_ys=[]
  for i in range(1000):
    point_xs.append(random.gauss(mu_1, math.sqrt(sigma2_1)))
    point_ys.append(random.gauss(mu_2, math.sqrt(sigma2_2)))
  plt.scatter(point_xs, point_ys, s=5)
  #plt.show()

def sample_plot_rho_theta(sigma2_r, mu_r, sigma2_th, mu_th):
  #show points in unnormalized 2D gaussian
  point_rs=[]
  point_ths=[]
  point_xs=[]
  point_ys=[]
  for i in range(1000):
    point_th = random.gauss(mu_th, math.sqrt(sigma2_th))
    point_r = random.gauss(mu_r, math.sqrt(sigma2_r))
    point_ths.append(point_th)
    point_rs.append(point_r)
    c = cmath.rect(point_r, point_th)
    point_xs.append(c.real)
    point_ys.append(c.imag)
  plt.scatter(point_xs, point_ys, s=5)
  plt.show()
  plt.scatter(point_rs, point_ths, s=5)
  plt.show()


alphas=[0,1]
n=1
Cs=[1,10, 100]
cutoff_wavelength=.2*2*math.pi/100
#coeffs_real = get_VWmatrix(field='imag', maxwavevector=Q, alpha=alpha, n=n, C=C, l=l)
lx_s=[x *2*math.pi/100 for x in [2,3,4,5,6,7]]
ly_s=[x *2*math.pi/100 for x in [2, 4,7]]
#exaine a fixed cell size
matrix_out= dict([]) #start a dict to collect results
for alpha in alphas:
  for C in Cs:
    for lx in lx_s:
      for ly in [1]:
        ly=lx #for isotropic case
        Q=(int(lx/cutoff_wavelength),int(ly/cutoff_wavelength))
        B_matrix = get_VWmatrix(field='real', maxwavevector=Q, alpha=alpha, n=n, C=C, lx=lx, ly=ly)[-2:, -2:]
        #store results
        #matrix -> dict
        print(alpha, C, lx, ly)
        print(B_matrix)
        B_matrix_dict=dict([])
        for i in range(2):
          for j in range(2):
            B_matrix_dict[(i,j)]=B_matrix[i,j]
        #put dict in resultsdict matrixout
        matrix_out[(alpha, C, lx, ly)]= B_matrix_dict

#dict -> dataframe
data=pd.DataFrame.from_dict(matrix_out)
print(data)
#dataframe -> csv file
data.to_csv("quarter_waves_highC.csv")

import matplotlib.pyplot as plt
import cmath
import numpy as np
import math
import system_cylinder1D as system


alpha = -1
u = 1
C =.1
n=0
wavenumber = 1
kappa=0
a=.5
n_coeffs = 10
title = "-.png"

sys = system.Cylinder1D(alpha=alpha, n=n, C=C, u=u, wavenumber = wavenumber, kappa=kappa,
                        radius=1, gamma=1, num_field_coeffs=n_coeffs)
#prompt cylinder to fill A, B, D matrices
sys.evaluate_A_integrals(amplitude=a)
sys.evaluate_B_integrals(amplitude=a)
matrix = alpha*sys.tmp_A_matrix + C* sys.tmp_B_integrals
print(matrix)
plt.imshow(abs(matrix))
plt.xticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.yticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.colorbar()
plt.title("G^-1, a = "+str(a)+ " C = "+str(C)+" n = "+str(n))
plt.savefig(title)
plt.close()
eigenvalues, eigenvectors = np.linalg.eigh(matrix)
plt.imshow(abs(eigenvectors))
plt.xticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.yticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.colorbar()
plt.title("eigenvectors of G^-1, a = "+str(a)+ " C = "+str(C)+" n = "+str(n))
plt.savefig("eigenvectors_"+title)
plt.close()
plt.imshow(abs(np.diag(eigenvalues)))
plt.xticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.yticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.colorbar()
plt.title("diagonalized G^-1, a = "+str(a)+ " C = "+str(C)+" n = "+str(n))
plt.savefig("diagonalized_"+title)
plt.close()
inverse = np.linalg.inv(matrix)
plt.imshow(abs(inverse), vmax=.5)
plt.xticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.yticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.colorbar()
plt.title("abs(G), a = "+str(a)+ " C = "+str(C)+" n = "+str(n))
plt.savefig("abs_inverse_"+title)
plt.close()

plt.imshow(inverse.imag)
print(inverse[0,-1].imag, inverse[0,0].imag,inverse[0,1].imag,)
plt.xticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.yticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.colorbar()
plt.title("G.imag, a = "+str(a)+ " C = "+str(C)+" n = "+str(n))
plt.savefig("imag_inverse_"+title)
plt.close()

plt.imshow(inverse.real)
plt.xticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.yticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.colorbar()
plt.title("G.real, a = "+str(a)+ " C = "+str(C)+" n = "+str(n))
plt.savefig("real_inverse_"+title)
plt.close()
modes=dict([(i,[]) for i in range(-n_coeffs, n_coeffs+1)])
def get_value(z):
  # picture of the field when c_0 has value from a fluctuation
  #c_0=1 with phase (1,0)
  f=(0+0j)
  for i in range(-n_coeffs, n_coeffs+1):
    index = n_coeffs+i
    magnitude = abs(inverse[n_coeffs,index])
    phase = cmath.phase(inverse[n_coeffs,index])
    f+= cmath.rect(magnitude, phase)*(math.cos(i*z) + math.sin(i*z)*1j)
    modes[i].append(cmath.rect(magnitude, phase)*(math.cos(i*z) + math.sin(i*z)*1j))
  return f
xs=np.arange(0,2*math.pi,.02)
upper_sin= [-.0015+.001*(1+a*math.sin(wavenumber*x)) for x in xs]
lower_sin= [-.0025-.001*(1+a*math.sin(wavenumber*x)) for x in xs]
f= np.array([get_value(x) for x in xs])
plt.plot(xs,abs(f))

plt.legend()
plt.plot(xs, upper_sin, color='green', linewidth=10)
plt.plot(xs,lower_sin, color='green', linewidth=10)
plt.xlabel("z")
plt.savefig("cofluctuations"+title)

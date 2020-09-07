import matplotlib.pyplot as plt
import cmath
import numpy as np

import system_cylinder1D as system


alpha = 1
u = 1
C = 1
n=1
wavenumber = 1
kappa=0
a=.1
n_coeffs = 10
title = "matrix_small_a.png"

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
plt.imshow(abs(inverse))
plt.xticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.yticks(range(0, 2*n_coeffs+2, 5), range(-n_coeffs, n_coeffs+1, 5))
plt.colorbar()
plt.title("abs(G), a = "+str(a)+ " C = "+str(C)+" n = "+str(n))
plt.savefig("abs_inverse_"+title)
plt.close()

plt.imshow(inverse.imag)
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




import numpy as np
import timeit
import random
import math


def generic_loop(c,m, n):
  sum_ = 0
  for i in range(n):
    for j in range(n):
      sum_ += c[i] * m[i,j] * c[j].conjugate()
  return sum_


def generic_einsum_transpose(c,m):
  return np.einsum("ji, i, j -> ", m, c, c.conjugate())

def generic_einsum(c,m):
  return np.einsum("ij, i, j -> ", m, c, c.conjugate())

def A_loop(c,m, n):
  sum_ = 0
  for i in range(n):
    for j in range(n):
      sum_ += c[i] * A[i-j-n] * c[j].conjugate()
  return sum_

def A_fill():
  global A_matrix
  for i in range(n):
    A_matrix[i] = A[n-i-1:2*n-1-i]
  

if __name__ == "__main__":
  n = 11
  c = np.array([(random.uniform(-1,1)+random.uniform(-1,1)*1j) for i in range(n)]) #random complex numbers length n  - squareish distribution
  A = np.array([(random.uniform(-1,1)+random.uniform(-1,1)*1j) for i in range(-n+1, n)]) #random complex numbers length n  - squareish distribution
  A_matrix = []
  for i in range(n):
    A_matrix.append(A[i:i+n])
  A_matrix = np.array(A_matrix)
  generic_matrix = []
  for j in range(n):
    generic_matrix.append([(random.uniform(-1,1)+random.uniform(-1,1)*1j) for i in range(n)]) #nrandom complex matrix n*n
  generic_matrix = np.array(generic_matrix)
  repeat = 10

  print("number of field coeffs", n, "repeats of evaluation", repeat)

  print("generic loop")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result1 = generic_loop(c,generic_matrix,n)
  print(timeit.default_timer() - start_time)

  print("generic einsum")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result2 = generic_einsum(c,generic_matrix)
  print(timeit.default_timer() - start_time)

  #print(result1, result2)
  assert(result1 == result2)

  #but what if A has structure 
  # [A0  A1  A0  A2  A3]
  # [A-1 A0  A1  A1  A2]
  # [A-2 A-1 A0  A1  A2]
  # [A-3 A-2 A-1 A0  A1]
  # [A-4 A-3 A-2 A-1 A0]
  # ...
  # and is saved as 1D array A[i]
  print("------ A calculation --------")
  print("loop over lookup from A as 1D list")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result1 = A_loop(c,generic_matrix,n)
  print(timeit.default_timer() - start_time)

  print("einsum, creating matrix A from 1D list every time")
  start_time = timeit.default_timer()
  for i in range(repeat):
    A_fill()
    result2 = generic_einsum_transpose(c,A_matrix)
  print(timeit.default_timer() - start_time)

  print("einsum, if matrix A is restructured from 1D list once (infrequent amplitude change case)")
  start_time = timeit.default_timer()
  A_fill()
  for i in range(repeat):
    result3 = generic_einsum_transpose(c,A_matrix)
  print(timeit.default_timer() - start_time)
  
  assert(math.isclose(result1.real,result2.real))
  assert(math.isclose(result1.real,result3.real))
  assert(math.isclose(result1.imag,result2.imag))
  assert(math.isclose(result1.imag,result3.imag))

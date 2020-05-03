import numpy as np
import timeit
import random
import math
import scipy



def generic_loop(c,m, n):
  sum_ = 0
  for i in range(n):
    for j in range(n):
      sum_ += c[i] * m[i,j] * c[j].conjugate()
  return sum_

def generic_loop_4(c,m, n):
  sum_ = 0
  for i in range(n):
    for j in range(n):
      for k in range(n):
        for l in range(n):
          sum_ += c[i] * c[j]* m[i,j,k,l] * c[k].conjugate() * c[l].conjugate()
  return sum_



def generic_einsum_transpose(c,m):
  return np.einsum("ji, i, j -> ", m, c, c.conjugate())

def generic_einsum_4d_transpose(c,m):
  return np.einsum("lkij, i, j , k, l -> ", m, c,c, c.conjugate(), c.conjugate())


def generic_einsum_scipy(c,m):
  return scipy.einsum("ij, i, j -> ", m, c, c.conjugate())


def generic_multidot(c,m):
  return np.linalg.multi_dot([c, m, c.conjugate().transpose()])

def generic_einsum(c,m):
  return np.einsum("ij, i, j -> ", m, c, c.conjugate())


def generic_einsum_4d(c,m):
  return np.einsum("ijkl, i, j,k, l -> ", m, c, c, c.conjugate(), c.conjugate())


def A_loop(c,m, n):
  sum_ = 0
  for i in range(n):
    for j in range(n):
      sum_ += c[i] * A[i-j+2*n-1] * c[j].conjugate()
  return sum_

def D_loop(c,m, n):
  sum_ = 0
  for i in range(n):
    for j in range(n):
      for k in range(n):
        for l in range(n):
          sum_ += c[i]*c[j] * A[i+j-l-k+2*n-1] * c[k].conjugate() * c[l].conjugate()
  return sum_

def A_fill():
  global A_matrix
  for i in range(n):
    A_matrix[i] = A[2*n-i-1:3*n-1-i]
    # A_matrix[i] = A[2*n+i-1:n+i-1:-1] # for i as forwards index, slice with backwards index - non-transpose version of einsum can be used
                                          # but filling hte matrix with a backwards slice is slower

def D_fill():
  global D_matrix
  for i in range(n):
    for j in range(n):
      for k in range(n):
        # A goes from 0 to 4n+1- middle eleemnt is at 2n-1
        # i =0 j = 0 k =0 : [ A_2n-1 to A_3n-1] 
        # i = max j = max k = max: [A_0 to ] ? ij are (-) indices
        D_matrix[i,j,k] =  A[k-i-j+2*n-1:k-i-j+3*n-1] #??some column constructed from A list
        #if A[-1] in D_matrix[i,j,k]:
          #print("used last element in A")
        #D_matrix[i,j,k] =  A[i+j-k+2*n-1:i+j-k+n-1:-1] # with i,j as forward indices.  filling D matrix in this way is slower
  

if __name__ == "__main__":
  n = 3
  c = np.array([(random.uniform(-1,1)+random.uniform(-1,1)*1j) for i in range(n)]) #random complex numbers length n  - squareish distribution
  A = np.array([(random.uniform(-1,1)+random.uniform(-1,1)*1j) for i in range(-2*n+1, 2*n-1)]) #random complex numbers length n  - squareish distribution
  print(A, len(A))
  A_matrix = []
  for i in range(n):
    A_matrix.append(A[i:i+n])
  A_matrix = np.array(A_matrix)
  generic_matrix = []
  for j in range(n):
    generic_matrix.append([(random.uniform(-1,1)+random.uniform(-1,1)*1j) for i in range(n)]) #nrandom complex matrix n*n
  generic_matrix = np.array(generic_matrix)
  generic_matrix_4d = []
  for i in range(n):
    m1 = []
    for j in range(n):
      m2 = []
      for k in range(n):
        m2.append([(random.uniform(-1,1)+random.uniform(-1,1)*1j) for i in range(n)]) 
      m1.append(m2)
    generic_matrix_4d.append(m1)
  generic_matrix_4d = np.array(generic_matrix_4d)
  D_matrix = generic_matrix_4d #copy to initialize right shape 4d array of complex number type

  repeat = 50

  print("number of field coeffs", n, "repeats of evaluation", repeat)

  print("generic loop")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result1 = generic_loop(c,generic_matrix,n)
  print(timeit.default_timer() - start_time)


  print("generic einsum numpy")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result2 = generic_einsum(c,generic_matrix)
  print(timeit.default_timer() - start_time)


  print("generic einsum scipy")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result3 = generic_einsum_scipy(c,generic_matrix)
  print(timeit.default_timer() - start_time)
  
  print("generic np linalg multidot")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result4 = generic_multidot(c,generic_matrix)
  print(timeit.default_timer() - start_time)

  #print(result1, result2)
  assert(result1 == result2)
  assert(result1 == result3)
  assert(math.isclose(result1.real,result4.real))
  assert(math.isclose(result1.imag,result4.imag))

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


  print( "-------- 4 dimensions --------------")
  print("4d generic loop")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result1 = generic_loop_4(c,generic_matrix_4d,n)
  print(timeit.default_timer() - start_time)


  print("4d generic einsum numpy")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result2 = generic_einsum_4d(c,generic_matrix_4d)
  print(timeit.default_timer() - start_time)


  print("generic einsum scipy")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result3 = generic_einsum_scipy(c,generic_matrix)
  #print(timeit.default_timer() - start_time)
  
  print("generic np linalg multidot")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result4 = generic_multidot(c,generic_matrix)
  #print(timeit.default_timer() - start_time)

  assert(math.isclose(result1.real,result2.real))
  assert(math.isclose(result1.imag,result2.imag))
  #assert(result1 == result3)
  #assert(math.isclose(result1.real,result4.real))
  #assert(math.isclose(result1.imag,result4.imag))

  print("------ D energy calculation --------")
  print("loop over lookup from D=A as 1D list")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result1 = D_loop(c,generic_matrix,n)
  print(timeit.default_timer() - start_time)

  print("einsum, creating matrix D from 1D list every time")
  start_time = timeit.default_timer()
  for i in range(repeat):
    D_fill()
    result2 = generic_einsum_4d_transpose(c,D_matrix)
  print(timeit.default_timer() - start_time)

  print("einsum, if matrix D is restructured from 1D list once (infrequent amplitude change case)")
  start_time = timeit.default_timer()
  D_fill()
  for i in range(repeat):
    result3 = generic_einsum_4d_transpose(c,D_matrix)
  print(timeit.default_timer() - start_time)
 
  print(result1, result2)
  assert(math.isclose(result1.real,result2.real))
  assert(math.isclose(result1.real,result3.real))
  assert(math.isclose(result1.imag,result2.imag))
  assert(math.isclose(result1.imag,result3.imag))

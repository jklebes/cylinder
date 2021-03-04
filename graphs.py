import copy
import numpy as np
import math
import matplotlib.pyplot as plt

def spawn_graphs(N):
  graphs0=dict([(i,[-1,-1]) for i in range(N)])
  graphs0['sign']=1
  #print(graphs0)
  graphs=[graphs0]
  for i in range(N):
    new_graphs=[]
    for graph_start in graphs:
      #print("graph start", graph_start)
      if graph_start[i][0]==-1:
        in_possibilities=[x for x in range(N) if graph_start[x][1]==-1] #if nothing is going out of it yet
      else: 
        in_possibilities = [graph_start[i][0]]
      if graph_start[i][1]==-1:
        out_possibilities=[x for x in range(N) if graph_start[x][0]==-1]
      else: 
        out_possibilities= [graph_start[i][1]]

      if (len(in_possibilities)==1 and len(out_possibilities)==1 
          and in_possibilities[0]!= out_possibilities[0] and -1 not in graph_start[i]):
        #criteria for closing off a cycle of size >=3? - changes sign ot the term
        graph_start['sign'] *=-1
      
      for n in in_possibilities:
          for m in out_possibilities:
            if not ((n==i and m!=i) or (m==i and n!=i)):
              graph_option=copy.deepcopy(graph_start)
              #print(graph_option, 'a')
              graph_option[i] = [n,m]
              graph_option[n][1] = i
              graph_option[m][0] = i
              #print(graph_option, 'b')
              new_graphs.append(graph_option)
    graphs=new_graphs
    #print(graphs)
  print("final",graphs)
  #print("len", len(graphs))
  return graphs

spawn_graphs(3)

def makeChis(n_BCs, maxwavevector, alpha, n, C, l):
  chis=dict([])
  #interpretation of indices: 0,1,2,3,4,5 <-> 1re, 1im, 2re, 2im, 3re... 
  for i in range(n_BCs):
    for j in range(n_BCs):
      chis[(i,j)] = makeChi(i,j,maxwavevector, alpha, n, C, l)
  #syymetrize to X_(ij) = 1/2 (X_ij + X_ji)
  symmetric_chis=dict([])
  for i in range(n_BCs):
    for j in range(i,n_BCs):
      symmetric_chis[(i,j)] = .5 *(chis[(i,j)]+chis[(j,i)])
  return symmetric_chis

def getlambda(i, side, maxwavevector):
  #this codes up functions defining vectors lambda
  # which retrieve real/img parts at boundary points mui
  lambdai=np.zeros((2*maxwavevector[0]+1, 2*maxwavevector[1]+1), dtype=complex)
  #the version that works with standard fourier decomposition Psi = sum of Psi_q e^(iqx/(2l))
  if i==0 or i==1:
    for nx in range(-maxwavevector[0], maxwavevector[0]+1):
      for ny in range(-maxwavevector[1], maxwavevector[1]+1):
        lambdai[nx-maxwavevector[0],ny-maxwavevector[1]]=(0+1j)**ny
  elif i==2 or i==3:
    for nx in range(-maxwavevector[0], maxwavevector[0]+1):
      for ny in range(-maxwavevector[1], maxwavevector[1]+1):
        lambdai[nx-maxwavevector[0],ny-maxwavevector[1]]=(0+1j)**nx
  elif i==4 or i==5:
    for nx in range(-maxwavevector[0], maxwavevector[0]+1):
      for ny in range(-maxwavevector[1], maxwavevector[1]+1):
        lambdai[nx-maxwavevector[0],ny-maxwavevector[1]]=(0+1j)**ny*(-1)**nx
  elif i==6 or i==7:
    for nx in range(-maxwavevector[0], maxwavevector[0]+1):
      for ny in range(-maxwavevector[1], maxwavevector[1]+1):
        lambdai[nx-maxwavevector[0],ny-maxwavevector[1]]=(0+1j)**nx*(-1)**ny
  elif i==8 or i==9:
    for nx in range(-maxwavevector[0], maxwavevector[0]+1):
      for ny in range(-maxwavevector[1], maxwavevector[1]+1):
        if nx%2==1 and ny%2==1:
          lambdai[nx-maxwavevector[0],ny-maxwavevector[1]]=-4/(math.pi**2*nx*ny)
    #nx==0 line
    nx=0
    for ny in range(-maxwavevector[1], maxwavevector[1]+1):
      if ny%2==1:
        lambdai[nx-maxwavevector[0],ny-maxwavevector[1]]=2j/(math.pi*ny)
    #ny==0 line
    ny=0
    for nx in range(-maxwavevector[0], maxwavevector[0]+1):
      if nx%2==1:
        lambdai[nx-maxwavevector[0],ny-maxwavevector[1]]=2j/(math.pi*nx)
    #nx==0, ny==0 value
    nx=0
    ny=0
    lambdai[nx-maxwavevector[0],ny-maxwavevector[1]]=1
  if i%2==0: #the rightlambdai, leftlambdai that sum to real part of Psi*lambda1
    if side=='left':
      return .5 * lambdai
    elif side=='right':
      return .5 *lambdai.conjugate()
  elif i%2==1: #return the rightlambdai or leftlambdai that sum to img part of Psi*lambdai
    if side=='left':
      return (0+1j)*.5 *lambdai
    elif side=='right':
      return -.5 * (0+1j)*lambdai.conjugate()

def makeChi(i, j, maxwavevector, alpha, n, C, l):
  #lock up leftlambdai, rightlambdaj
  leftlambdai = getlambda(i, 'left', maxwavevector)
  rightlambdaj= getlambda(j, 'right', maxwavevector)
  # get A-1 := -2 G = -2 (alpha A + C B)
  invA=np.zeros((2*maxwavevector[0]+1, 2*maxwavevector[1]+1), dtype=complex)
  #it's a 2d array (nx, ny) but it represents the diagonal of a 2d array (q,q')
  for nx in range(-maxwavevector[0], maxwavevector[0]+1):
    for ny in range(-maxwavevector[1], maxwavevector[1]+1):
      invA[nx-maxwavevector[0],ny-maxwavevector[1]] = 1/(alpha*(2 *math.pi * l)**2 + (nx**2 +ny**2) *n**2 * C * (2 *math.pi * l)**2)
  invA /= 2.0
  #could be vector* diagonal matrix*vector product or einsum for faster 
  chi_ij = 0+0j
  for nx in range(0, 2*maxwavevector[0]+1):
    for ny in range(0, 2*maxwavevector[1]+1):
      chi_ij+=leftlambdai[nx,ny]*invA[nx,ny]*rightlambdaj[nx, ny]
      #print(i,j,nx,ny, chi_ij, leftlambdai[nx,ny], rightlambdaj[nx, ny])
  return chi_ij

#print(makeChis((100,100), 1,1,1,1))

def get_U(n_BCs, maxwavevector, alpha, n, C, l):
  # make chi_0 to chi_nBCs (here 10 boundary conditions)
  chis=makeChis(n_BCs, maxwavevector, alpha, n, C, l)
  # hard coding the rule that all products of odd and even chis is 0
  odd_chis = dict([])
  even_chis=dict([])
  for key in chis:
    if key[0]%2==0 and key[1]%2==0:
      even_chis[key] = chis[key]
    elif key[1]%2==1 and key[1]%2==1:
      odd_chis[key]= chis[key]
    else:
      assert(math.isclose(chis[key].real, 0, abs_tol=1e-9))
      assert(math.isclose(chis[key].imag, 0, abs_tol=1e-9))
  # spawn all graphs between odd chis and musltiply out the terms - list of 120 numbers for U=10 (graphs of 5)
  graphs = spawn_graphs(n_BCs//2)
  #spawn all graphs between the (U//2=5) even chis and have list of numbers
  terms_odd=[]
  terms_even=[]
  for graph in graphs:
    #print(graph)
    term_odd = 1
    term_even= 1
    for node in range(n_BCs//2):
      chi_key = [node, graph[node][0]]
      chi_key.sort()
      chi_key_odd = tuple([x * 2 +1 for x in chi_key])
      chi_key_even = tuple([x * 2 for x in chi_key])
      term_odd *= odd_chis[chi_key_odd]
      #print(odd_chis[chi_key_odd])
      term_even *= even_chis[chi_key_even]
    term_odd *= graph['sign']
    term_even *= graph['sign']
    #print("made terms:", term_even, term_odd)
    terms_odd.append(term_odd)
    terms_even.append(term_even)
  # multiply out all 120 x 120 possible combinations and add them up
  U=0
  for i in terms_odd:
    for j in terms_even:
      U+= i*j
  #print(U)
  return U 


def get_V(n_BCs, coeff_index, maxwavevector, alpha, n, C, l):
  # make chi_0 to chi_nBCs (here 10 boundary conditions)
  chis=makeChis(n_BCs, maxwavevector, alpha, n, C, l)
  # hard coding the rule that all products of odd and even chis is 0
  odd_chis = dict([])
  even_chis=dict([])
  for key in chis:
    if key[0]%2==0 and key[1]%2==0:
      even_chis[key] = chis[key]
    elif key[1]%2==1 and key[1]%2==1:
      odd_chis[key]= chis[key]
    else:
      assert(math.isclose(chis[key].real, 0, abs_tol=1e-9))
      assert(math.isclose(chis[key].imag, 0, abs_tol=1e-9))
  # spawn all graphs between odd chis and musltiply out the terms - list of 120 numbers for U=10 (graphs of 5)
  graphs = spawn_graphs(n_BCs//2)
  #spawn all graphs between the (U//2=5) even chis and have list of numbers
  terms_odd=[]
  terms_even=[]
  # odd-indexed terms
  for graph in graphs:
    #everything connected to a Chi_(ii)
    index_in_graph = (coeff_index-1)//2
    if coeff_index%2==0 or graph[index_in_graph] == [index_in_graph, index_in_graph]: 
      # if coeff-index is odd we need to look at only odd graphs which
      # contain the self-loop i-i, corresponding to term Chi_ii
      # (else if coeffinedex is even look at all the odd graphs)
      #multipy out all 9 terms, except Chi_ii == multiply out all 10 terms, divide everything by Chi_ii later
      #print(graph)
      term_odd = 1
      for node in range(n_BCs//2):
        chi_key = [node, graph[node][0]]
        chi_key.sort()
        chi_key_odd = tuple([x * 2 +1 for x in chi_key])
        term_odd *= odd_chis[chi_key_odd]
        #print(odd_chis[chi_key_odd])
      term_odd *= graph['sign']
      #print("made terms odd:", term_odd)
      terms_odd.append(term_odd)
  #even factors
  for graph in graphs:
    #everything connected to a Chi_(ii)
    index_in_graph = (coeff_index)//2
    if coeff_index%2==1 or graph[index_in_graph] == [index_in_graph, index_in_graph]: 
      # same restriction of even graphs to those with i=i selfloop, if i is even
      term_even= 1
      for node in range(n_BCs//2):
        chi_key = [node, graph[node][0]]
        chi_key.sort()
        chi_key_even = tuple([x * 2 for x in chi_key])
        term_even *= even_chis[chi_key_even]
      term_even *= graph['sign']
      #print("made terms even:", term_even)
      terms_even.append(term_even)
  V_i=0
  for i in terms_odd:
    for j in terms_even:
      V_i+= i*j
  V_i /= chis[(coeff_index, coeff_index)]
  #print(V_i)
  return V_i

def get_W(n_BCs, coeff_index, maxwavevector, alpha, n, C, l):
  # make chi_0 to chi_nBCs (here 10 boundary conditions)
  chis=makeChis(n_BCs, maxwavevector, alpha, n, C, l)
  # hard coding the rule that all products of odd and even chis is 0
  odd_chis = dict([])
  even_chis=dict([])
  for key in chis:
    if key[0]%2==0 and key[1]%2==0:
      even_chis[key] = chis[key]
    elif key[1]%2==1 and key[1]%2==1:
      odd_chis[key]= chis[key]
    else:
      assert(math.isclose(chis[key].real, 0, abs_tol=1e-9))
      assert(math.isclose(chis[key].imag, 0, abs_tol=1e-9))
  # spawn all graphs between odd chis and musltiply out the terms - list of 120 numbers for U=10 (graphs of 5)
  graphs = spawn_graphs(n_BCs//2)
  #spawn all graphs between the (U//2=5) even chis and have list of numbers
  terms_odd=[]
  terms_even=[]
  # odd-indexed terms
  for graph in graphs:
    #everything connected to a Chi_(ii)
    index_in_graph = [(x-1)//2 for x in coeff_index]
    if all([c%2==0 for c in coeff_index]) or index_in_graph[1] in graph[index_in_graph[0]]: #includes i-j connection
      # if coeff-index is odd we need to look at only odd graphs which
      # contain the self-loop i-i, corresponding to term Chi_ii
      # (else if coeffinedex is even look at all the odd graphs)
      #multipy out all 9 terms, except Chi_ii == multiply out all 10 terms, divide everything by Chi_ii later
      #print(graph)
      term_odd = 1
      for node in range(n_BCs//2):
        chi_key = [node, graph[node][0]]
        chi_key.sort()
        chi_key_odd = tuple([x * 2 +1 for x in chi_key])
        term_odd *= odd_chis[chi_key_odd]
        #print(odd_chis[chi_key_odd])
      term_odd *= graph['sign']
      #print("made terms odd:", term_odd)
      terms_odd.append(term_odd)
  #even factors
  for graph in graphs:
    #everything connected to a Chi_(ii)
    index_in_graph = [c//2 for c in coeff_index] #only gets used if coeff_indices are even
    if all([c%2==1 for c in coeff_index]) or index_in_graph[1] in graph[index_in_graph[0]]: 
      # same restriction of even graphs to those with i=i selfloop, if i is even
      term_even= 1
      for node in range(n_BCs//2):
        chi_key = [node, graph[node][0]]
        chi_key.sort()
        chi_key_even = tuple([x * 2 for x in chi_key])
        term_even *= even_chis[chi_key_even]
      term_even *= graph['sign']
      #print("made terms even:", term_even)
      terms_even.append(term_even)
  W_ij=0
  for i in terms_odd:
    for j in terms_even:
      W_ij+= i*j
  W_ij /= chis[(min(coeff_index), max(coeff_index))]
  #print(W_ij)
  return W_ij
"""
Q=(20,20)
alpha=1
n=1
C=1
l=1
alphas=[]
Us=[]
a_s=[.1,.2,.3,.4,.8,.9,1.5, .5, 1, 2]
for a in a_s:
  U=get_U(10, Q, a, n,C,l)
  V_8=get_V(10, 8, Q, a, n,C,l) # coefficeint of mu_5^re ^2 
  W_08 = get_W(10, (0,8), Q, a, n,C,l)
  C_= -.5*-1/U*(4*W_08) /4.0
  print("C'=", C_ )
  alpha_=.5*V_8/U-1*C_
  print(a,alpha_)
  Us.append(U)
  alphas.append(alpha_)
print(Us, alphas)
plt.scatter(a_s, alphas)
plt.xlabel("alpha0")
plt.ylabel("alpha'")
plt.savefig("alphavsalpha0")

U=get_U(10, Q, alpha, n,C,l)
V_8=get_V(10, 8, Q, alpha, n,C,l) # coefficeint of mu_5^re ^2 
V_9=get_V(10, 9, Q, alpha, n,C,l) # coefficient of mu_5^im ^2
print(.5*V_8/U) #effective alpha of real field
#print(.5*V_9/U) # effective alpha of imag field
W_08 = get_W(10, (0,8), Q, alpha, n,C,l)
#print(-1*W_08/U)
W_28 = get_W(10, (2,8), Q, alpha, n,C,l)
#print(-1*W_28/U)
W_48 = get_W(10, (4,8), Q, alpha, n,C,l)
print(-1*W_48/U)
W_68 = get_W(10, (6,8), Q, alpha, n,C,l)
#print(-1*W_68/U)
#W_88 = get_W(10, (8,8), Q, alpha, n,C,l)
#print(-2*W_88/U, V_8/U)
C_= -.5*-1/U*(W_08 + W_28 + W_48+ W_68) /4.0
print("C'=", C_)
alpha_=.5*V_8/U-1*C_
print("alpha'=", alpha_)
assert(C_>=0)
assert(alpha_ >= 0)
"""
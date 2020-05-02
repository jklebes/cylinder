"""
to test whether when a complex number is stepped by stepping x and y seperately
(rather than phase and amplitude), the distribution will be biased (squareish on the complex plane
rather than circular)
"""
import random
import matplotlib.pyplot as plt
import numpy as np
n_exps=500
nsteps=2000
sigma=.1
xs_total=[]
ys_total=[]
#plt.add_artist(plt.Circle((0, 0), 1, fill=False))
for i in range(n_exps):
  x,y = 0,0
  xs = [x] 
  ys = [y]
  for n in range(nsteps):
    x += random.gauss(0, sigma)
    y += random.gauss(0,sigma)
    xs.append(x)
    ys.append(y)
  xs_total.extend(xs)
  ys_total.extend(ys)
  #scatter all on the same plot
  plt.scatter(xs, ys, s=1, marker='.')
plt.savefig("x_y_markovchain.png")
plt.close()
H, xedges, yedges = np.histogram2d(xs_total, ys_total, bins=100, range=[[-5,5],[-5,5]])
H=H.T
plt.axis("off")
plt.imshow(H, cmap="jet")
plt.savefig("x_y_markovchain_density.png")

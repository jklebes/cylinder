import math
import matplotlib.pyplot as plt
xs_ = [x*.001 for x in range(0,1000)]
ys=[]
xs = []
for x in xs_:
  if x>0 and (-23*x*x+6*x+1) >0 and math.sqrt(-23*x*x + 6*x +1)+x-1 >0:
    ys.append(.5 * math.sqrt((math.sqrt(-23*x*x+6*x+1)+x-1)/x))
    xs.append(x)
  elif x==0:
    ys.append(1)
    xs . append(0)
plt.plot(xs,ys)
plt.xlabel("bending rigidity kappa_b")
plt.ylabel("wavenumber k*")
plt.xlim((0,.45))
plt.ylim((0,1.27))
plt.title  ("critical wavenumber k* for stability of cylinder")
plt.gca().invert_yaxis()
#plt.show()
plt.savefig("nofield_instabiity_calc.png")
plt.close()

import matplotlib.pyplot as plt
import math
import numpy as np

xs = np.arange(0,10,.1)
ys = [math.sqrt(2/x) for x in xs]
plt.ylabel("fixed surface amplitude |a|")
plt.xlabel("C")
plt.title("phase diagram of 0th order field on fixed cylinder shapes")
plt.ylim((0,1))
plt.xlim((0,10))
plt.plot(xs,ys)
plt.show()

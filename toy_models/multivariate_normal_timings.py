import numpy as np
import timeit
import random
import math
import scipy.stats


if __name__ == "__main__":
  n = 7
  repeat = 500

  print("number of field coeffs", n, "repeats of evaluation", repeat)

  print("numpy's multivariate normal, having to default mean cov")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result1 = np.random.multivariate_normal(mean=np.zeros(n),cov=np.identity(n))
  print(timeit.default_timer() - start_time)


  print("scipy's multivariate normal, defaults")
  start_time = timeit.default_timer()
  for i in range(repeat):
    result2 = scipy.stats.multivariate_normal(cov=np.identity(n))
  print(timeit.default_timer() - start_time)


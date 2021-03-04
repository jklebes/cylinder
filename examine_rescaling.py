import pandas as pd
import matplotlib.pyplot as plt
import math

data=pd.read_csv("first_Bmatrix_result.csv", header=[0,1,2], index_col=[0,1])

idx = pd.IndexSlice
#how to select values: data.loc[idx[:, 1], idx[:, :, :]]) 
# i,j ; alpha, c, l

Ns=[x for x in [2,3,4,5,6,7]]
print(Ns)
ls = [math.pi/100 *x for x in [2,3,4,5,6,7]]
alpha=1
for C in (1,5,10):
  B55 = [complex(x).real for x in data.loc[idx[4,4], idx[str(alpha),str(C),:]]]
  B11 = [complex(x).real for x in data.loc[idx[0,0], idx[str(alpha),str(C),:]]]
  B15 = [complex(x).real for x in data.loc[idx[0,4], idx[str(alpha),str(C),:]]]
  B13 = [complex(x).real for x in data.loc[idx[0,2], idx[str(alpha),str(C),:]]]
  series=[]
  for n in range(len(B55)):
      series.append((.5* B55[n] + .5 *B11[n] - .5* B13[n] + B15[n])/(9*C))
  print(series)
  series2=[x/C for x in B13]
  #plt.plot([l for l in ls], [s for (s,l) in zip(series,ls)], label= "alpha'/C0, C0="+str(C))
  plt.plot(ls, [s for (s,l) in zip(series2,ls)], label= "B13/C0, C0="+str(C))
#plt.legend()
#plt.show()

meaninvA_alpha1=[(3.667154682002879e-05+2.2161435420844943e-13j),(7.856147049414978e-06+3.579602317349035e-14j),
          (1.8024515410294011e-06+6.008957321286671e-13j), (6.802992865168801e-07-6.139375110564569e-16j),
          (3.751458312400601e-07+1.1113998785891044e-13j), (1.7494570907333165e-07-1.8975445427388573e-14j)]
meanA_alpha1=[(1.6421045188419363+0j), (1.644601750711744-4.107299369553775e-17j), (1.6454920691731483-8.177208667468463e-17j),
             (1.6458717808299002-2.59999828011719e-16j), (1.6461546939034972+6.065595011293969e-17j),
             1]
mean_lambda5=[(0.002770083102493075+0j), (0.001371742112482853+0j), (0.0006574621959237343+2.919718670940583e-19j), 
              (0.00041649312786339027+0j), (0.0003077870113881194-1.3668489068946217e-19j),
              (0.0002100399075824407+0j)]

#plt.plot(ls, [1/(x.real) for x in meanA_alpha1])
#plt.plot(ls, [1/(l*2*5+1)**2 for l in ls])
#plt.plot(ls, [1/(l)**2*.001 for l in ls])
#plt.plot(ls, [6.3/math.log((10*n-1)) for n in Ns], linewidth=4, linestyle='dashed',label="k/log(N)")
##plt.plot(ls, [1/((10*n-1)**2*(x.real)) for (n,x) in zip(Ns, mean_lambda5)], label="1/N^2")
#plt.plot(ls, [1/(x.real) for x in mean_lambda5], label="1/(mean of lambda5)")
#plt.plot(ls, [(10*n-1)**2 for n in Ns], label="N")

plt.xlabel("l")

plt.legend()
plt.show()
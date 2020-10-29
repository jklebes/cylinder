import cmath
import pickle
import os
import datetime
import timeit
import argparse
import pandas as pd
import run 


if __name__ == "__main__":
  #everything temporarily hardcoded here, should be input file
  n_steps = 100
  field_type = "lattice"
  method= "sequential"  
  #coefficients
  alpha = -1
  C = 1
  u = 1
  n = 6
  kappa = 0
  gamma = 1
  temp = .1
  intrinsic_curvature = 0

  # system dimensions
  amplitude= 0  #also fixed system amplitude for when amplitude is static
  radius = 1
  wavenumber = .5

  # simulation details - for fourier fields
  num_field_coeffs = (2,1) # z-direction modes indices go from -.. to +..; theta direction indices go from -.. to +..
  initial_sampling_width = .025
  measure_every =10
  fieldsteps_per_ampstep = 10  #only relevant for sequential

  #for lattice simulations
  dims=(100,50)
  temperature_lattice=temp
  n_substeps = dims[0]*dims[1]

  parser = argparse.ArgumentParser(description='get description')
  #from varfile
  #first line of file: what to vary
  parser.add_argument('--varnames', dest = 'varnames',  type=str, nargs=2, required=True)
  parser.add_argument('--varline', dest = 'varline',  type=str, nargs=2, required=True)
  args=parser.parse_args()
  var1name=args.varnames[0]
  var2name=args.varnames[1]
  var1=args.varline[0]
  var2=args.varline[1]

  #meta move to assign value from file to alpha, C, wavenumber, or whatever
  exec(var1name+" = "+ var1)
  exec(var2name+" = "+ var2)
  #overwriting the relevant fiedls in the hardcoded list abovve

  #decide on file name
  filename = var1name+"_"+var1+"_"+ var2name+"_"+var2

  #run a single simulation as in run file
  results = run.single_run(n_steps=n_steps, field_type=field_type, method=method,
                num_field_coeffs=num_field_coeffs, fieldsteps_per_ampstep=fieldsteps_per_ampstep, 
                measure_every=measure_every,
                alpha=alpha, C=C, n=n, u=u, 
                gamma=gamma, kappa=kappa, radius=radius, intrinsic_curvature=intrinsic_curvature,
                dims=dims, temperature_lattice=temperature_lattice, n_substeps=n_substeps,
                wavenumber=wavenumber,  
                amplitude=amplitude, field_coeffs=None, outdir = '.', title = filename)
  #returns me.params_names, me.observables_names, result_means (dict), me.covariance_matrix_complex
  #save results 
  data =pd.DataFrame(result_means)
  data.to_csv(filename+"_mean.csv")


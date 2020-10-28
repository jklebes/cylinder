import cmath
import pickle
import os
import datetime
import timeit
import argparse
import run 

if __name__ == "__main__":
  #everything temporarily hardcoded here, should be input file
  n_steps = 1000
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
  parser.add_argument('--var1name', dest = 'var1name',  type=str, nargs=1, required=True)
  parser.add_argument('--var2name', dest = 'var2name',  type=str, nargs=1, required=True)
  var1name=parser.parse_args().var1name
  var2name=parser.parse_args().var2name

  #input as command ilne argument via taskarray:
  parser.add_argument('--var1', dest = "var1",  type=str, nargs=1, required=True)
  parser.add_argument('--var2', dest = "var2",  type=str, nargs=1, required=True)  
  var1=parser.parse_args().var1
  var2=parser.parse_args().var2

  #meta move to assign value from file to alpha, C, wavenumber, or whatever
  exec(var1name+" = "+ var1)
  exec(var2name+" = "+ var2)
  #overwriting the relevant fiedls in the hardcoded list abovve

  #decide on file name
  filename = var1name+"_"+str(round(var1,5)+"_"+ var2name+"_"+str(round(var2,5))

  #run a single simulation as in run file
  run.single_run(n_steps=n_steps, field_type=field_type, method=method,
                num_field_coeffs=num_field_coeffs, fieldsteps_per_ampstep=fieldsteps_per_ampstep, 
                measure_every=measure_every,
                alpha=alpha, C=C, n=n, u=u, 
                gamma=gamma, kappa=kappa, radius=radius, intrinsic_curvature=intrinsic_curvature,
                dims=dims, temperature_lattice=temperature_lattice, n_substeps=n_substeps,
                wavenumber=wavenumber,  radius=radius,
                amplitude=amplitude, field_coeffs=None, outdir = '.', title = filename)
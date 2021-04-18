import cmath
import pickle
import os
import datetime
import timeit
import argparse
import pandas as pd
import run 


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='get description')
  parser.add_argument('-i', '--input', default="infile.txt",
                      help='file to read parameters from', dest='infilename',
                      type=str)
  #from varfile
  #first line of file: what to vary
  parser.add_argument('--varnames', dest = 'varnames',  type=str, nargs=2, required=True)
  parser.add_argument('--varline', dest = 'varline',  type=str, nargs=2, required=True)
  args=parser.parse_args()
  var1name=args.varnames[0]
  var2name=args.varnames[1]
  var1=args.varline[0]
  var2=args.varline[1]
  infilename = args.infilename

  #new argparser to read input file
  fileparser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  fileparser.convert_arg_line_to_args = convert_arg_line_to_args
  fileparser.add_argument('--temp', type=float)
  fileparser.add_argument('--temp_final', type=float, required=False, default=None)
  fileparser.add_argument('--n_steps', type=int)
  fileparser.add_argument('--field_type', type=str)
  fileparser.add_argument('--method', type=str)
  fileparser.add_argument('--num_field_coeffs', required=False, nargs=2)
  fileparser.add_argument('--fieldsteps_per_ampstep', required=False)
  fileparser.add_argument('--measure_every', type=int)
  fileparser.add_argument('--alpha', type=float)
  fileparser.add_argument('--C', type=float)
  fileparser.add_argument('--n', type=float)
  fileparser.add_argument('--u', type=float)
  fileparser.add_argument('--gamma', type=float)
  fileparser.add_argument('--kappa', type=float)
  fileparser.add_argument('--radius', type=float)
  fileparser.add_argument('--intrinsic_curvature', type=float)
  fileparser.add_argument('--dims', type=int, nargs=2)
  fileparser.add_argument('--temperature_lattice', required=False, default=None, type=float)
  fileparser.add_argument('--n_substeps', required=False, default=None, type=int)
  fileparser.add_argument('--wavenumber', type=float)
  fileparser.add_argument('--amplitude', type=float)
  fileparser.add_argument('--field_coeffs', required=False, default=None)
  fileparser.add_argument('--outdir', required=False, default='.', type=str)
  args= fileparser.parse_args(['@'+infilename])
  if args.temperature_lattice is None:
    args.temperature_lattice=1
  if args.n_substeps is None:
    args.n_substeps = args.dims[0]*args.dims[1]
  if args.temp_final is None:
    args.temp_final = args.temp

  temp=args.temp
  temp_final=args.temp_final
  n_steps=args.n_steps
  field_type=args.field_type
  method=args.method
  num_field_coeffs=args.num_field_coeffs
  fieldsteps_per_ampstep=args.fieldsteps_per_ampstep
  measure_every=args.measure_every
  alpha=args.alpha
  C=args.C
  n=args.n
  u=args.u
  gamma=args.gamma
  kappa=args.kappa
  radius=args.radius
  intrinsic_curvature=args.intrinsic_curvature
  dims=args.dims
  temperature_lattice=args.temperature_lattice
  n_substeps=args.n_substeps
  wavenumber=args.wavenumber
  amplitude=args.amplitude
  field_coeffs=args.field_coeffs
  outdir=args.outdir

  #meta move to assign value from file to alpha, C, wavenumber, or whatever
  exec(var1name+" = "+ var1)
  exec(var2name+" = "+ var2)
  #overwriting the relevant fiedls in the hardcoded list abovve

  #decide on file name
  filename = var1name+"_"+var1+"_"+ var2name+"_"+var2
  
  #run a single simulation as in run file
  results = run.single_run(temp=temp, temp_final=temp_final, n_steps=n_steps, field_type=field_type, 
                method=method,
                num_field_coeffs=num_field_coeffs, fieldsteps_per_ampstep=fieldsteps_per_ampstep, 
                measure_every=measure_every,
                alpha=alpha, C=C, n=n, u=u, 
                gamma=gamma, kappa=kappa, radius=radius, 
                intrinsic_curvature=intrinsic_curvature,
                dims=dims, n_substeps=n_substeps,
                wavenumber=wavenumber,  
                amplitude=amplitude, field_coeffs=field_coeffs, outdir = outdir, title = filename)
  #returns me.params_names, me.observables_names, result_means (dict), me.covariance_matrix_complex
  #save results 
  d = dict([(key, [results[2][key]]) for key in results[2]])
  data =pd.DataFrame.from_dict(d)
  data.to_csv(filename+"_mean.csv")


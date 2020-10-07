import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import math
import collections
import timeit
import seaborn as sb
import argparse
import metropolisengine
import datetime
import os
import surfaces_and_fields.system_cylinder as cylinder
import surfaces_and_fields.On_lattice_simple as latticeOn
me_version = "0.2.19"#metropolisengine.__version__ #save version number TODO get it automatically 

def loop(single_run_lambda, range1, range2):
  """
  generic loop: runs a 2D grid of single simulations; 
  information about which two simulation parameters to vary is built into argument single_run_lambda()
  
  :param single_run_lambda: the function to launch a single simulation, taking two arguments
  :type single_run_lambda: function var1 var2 -> simulation output
  :param range1: iterable of values that the function can take as var1
  :param range2: iterable of values that the function can take as var2
  :returns results: dict of 2D arrays
  :rtype: dict 
  """
  results = collections.defaultdict(list)
  for var1 in range1:
    results_line = collections.defaultdict(list)
    for var2 in range2:
      start_time = timeit.default_timer()
      amplitude_avg, field_avg = single_run_lambda(var1, var2) 
      time = timeit.default_timer() - start_time
      results_line["average_amplitude"].append(amplitude_avg)
      results_line["average_field"].append(field_avg)
      results_line["time_per_experiment"].append(time)
    for name in results_line:
      results[name].append(results_line[name])
  return results 

functions_dict = {("num_field_coeffs", "fieldsteps_per_ampstep"): lambda num, fsteps: single_run(n_steps=n_steps, outdir = outdir, title = ("ncoeffs"+str(num)+"_fsteps"+str(fsteps)),
                                          alpha=alpha, C=C, n=n, u=u, gamma=gamma, kappa=kappa, radius=radius,
                                          wavenumber=wavenumber, intrinsic_curvature = intrinsic_curvature),
                  ("amplitude", "C"): lambda var_a, var_C: single_run(n_steps=n_steps,  outdir = outdir, title = ("a"+str(round(var_a,2))+"_C"+str(round(var_C,2))),
                                          alpha=alpha, C=var_C, n=n, u=u, gamma=gamma, kappa=kappa, radius=radius,
                                          wavenumber=wavenumber, intrinsic_curvature = intrinsic_curvature, amplitude=var_a),
                  ("wavenumber", "kappa"): lambda var_wvn, var_kappa: single_run(n_steps=n_steps,  outdir = outdir, title = ("wvn"+str(round(var_wvn,2))+"_kappa"+str(round(var_kappa,2))),
                                          alpha=alpha, C=C, n=n, u=u, gamma=gamma, kappa=var_kappa, radius=radius,
                                          wavenumber=var_wvn, intrinsic_curvature = intrinsic_curvature),
                  ("wavenumber", "alpha"): lambda var_wvn, var_alpha: single_run(n_steps=n_steps, outdir = outdir, title = ("wvn"+str(round(var_wvn,2))+"_alpha"+str(round(var_alpha,2))),
                                          alpha=var_alpha, C=C, n=n, u=u, gamma=gamma, kappa=kappa, radius=radius,
                                          wavenumber=var_wvn, intrinsic_curvature = intrinsic_curvature),                   
                  ("wavenumber", "intrinsic_curvature"): lambda var_wvn, var_H0: single_run(n_steps=n_steps,  outdir = outdir, title = ("wvn"+str(round(var_wvn,2))+"_H0"+str(round(var_H0,2))),
                                          alpha=alpha, C=C, n=n, u=u, gamma=gamma, kappa=kappa, radius=radius,
                                          wavenumber=var_wvn, intrinsic_curvature = var_H0),                   
                  ("alpha", "C"): lambda var_alpha, var_C: single_run(n_steps=n_steps,outdir = outdir, title = ("alpha"+str(round(var_alpha,2))+"_C"+str(round(var_C))),
                                          alpha=var_alpha, C=var_C, n=n, u=u, gamma=gamma, kappa=kappa, radius=radius,
                                          wavenumber=wavenumber, intrinsic_curvature = intrinsic_curvature)                   
                  }

def single_run(n_steps, outdir, title, alpha, C, n, u, gamma, kappa, radius, wavenumber, intrinsic_curvature, amplitude=0, n_substeps=None):
    #initialize a Lattice object
    # this automatically initializes a Cylinder(no field) object
    # and a MetropolisEngine to step Cylinder object

    print("single run")
    lattice = latticeOn.Lattice(amplitude=amplitude, alpha=alpha, C=C, n=n, u=u, gamma=gamma, kappa=kappa, radius=radius, wavenumber=wavenumber, intrinsic_curvature = intrinsic_curvature, temperature=temp)
    #protocol to run the lattice
    print("single run")
    for i in range(n_initial_steps):
      #metropolis step shape
      surface_accepted = lattice.me.step_real_group()
      if surface_accepted:
        lattice.amplitude = lattice.me.real_params[0]
        #maybe reset self.energy
      #lattice step
      for i in range(n_sub_steps):
        lattice.step_lattice(lattice.amplitude)
    n_measure_steps = n_steps - n_initial_steps
    for i in range(n_measure_steps):
      #metropolis step shape
      surface_accepted = lattice.me.step_real_group()
      if surface_accepted:
        lattice.amplitude = lattice.me.real_params[0]
        #maybe reset self.energy
      for i in range(n_sub_steps):
        lattice.step_lattice(lattice.amplitude)
      lattice.measure()
    return lattice.amplitude_average, lattice.field_average

def run_experiment(exp_type,  range1, range2):
  """
  Runs a 2D grid of single simulations 
  :param exp_type: tuple of strings, indicating which two variables will be varied
  :param experiment_title:
  :param range1: list of values for first variable to take
  :param range2: list of values for second variable to take
  :rtype: None
  """
  #make directory for the experiment
  global outdir
  now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  exp_dir= os.path.join("out", "exp-l-"+now)
  outdir = exp_dir
  os.mkdir(exp_dir)
  #save descritpion about how the experiment was run
  exp_notes = {"experiment type": " ".join(exp_type), "n_steps": n_steps, "temp":temp, 
              "C": C,"kappa": kappa, "gamma": gamma, "alpha": alpha, "n":n, "u":u, 
              "range1":range1, "range2": range2, 
              "radius":radius, "amplitude":initial_amplitude, "wavenumber": wavenumber, 
              "notes":notes, "start_time": now, 
              "me-version": me_version, "n_initial_steps": n_initial_steps}
  exp_notes = pd.DataFrame.from_dict(exp_notes, orient="index", columns=["value"])
  exp_notes.to_csv(os.path.join(exp_dir, "notes.csv"))
  # run experiment of the requested type
  try: 
    results = loop(functions_dict[exp_type], range1, range2)
  except KeyError:
    print("experiment type not found: ", exp_type)
    print("please set exp_type to one of", list(functions_dict.keys()))
    results = {}
  # save and plot results
  for name in results:
    plot_save(range1=range1, range2=range2, varname1 = exp_type[0],varname2 = exp_type[1],results=results[name], title=name, exp_dir=exp_dir)

def plot_save(range1, range2, varname1, varname2, results, title, exp_dir):
  print(results)
  if isinstance(results[0][0], float) or isinstance(results[0][0], int):
    #save as csv
    df = pd.DataFrame(index=range1, columns=range2, data=results)
    df.to_csv(os.path.join(exp_dir, title + ".csv"))
    #plot
    print("plotting png")
    sb.heatmap(df, cmap = "viridis")
    plt.savefig(os.path.join(exp_dir, title + ".png"))
    plt.close()
  else:
    print("lists")
    #save csv
    for (n,i) in enumerate(range1):
      for (m,j) in enumerate(range2):
        field_avg = results[n][m]
        df = pd.DataFrame(data=field_avg)
        title_ = title + "_"+ varname1 + "_"+str(round(i,3))+"_"+ varname2 + "_"+str(round(j,3))
        df.to_csv(os.path.join(exp_dir, title_ + ".csv"))
        plt.plot([z for z in range(len(field_avg))], field_avg)
        plt.savefig(os.path.join(exp_dir,title_+".png"))
        plt.close()
 
  
# global params - will use values set here if not loop variable
# coefficients
alpha = -1
C = 1
u = 1
n = 6
kappa = 0
gamma = 1
temp = .01
intrinsic_curvature = 0

# system dimensions
initial_amplitude= 0  #also fixed system amplitude for when amplitude is static
radius = 1
wavenumber = 1

#lattice characteristics
dims = (100,50)

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='get description')
  parser.add_argument('-n', '--notes', dest = 'notes',  type=str, nargs=1,
                    help='add experiment description with --notes', required=True)
  notes=parser.parse_args().notes

  # specify type, range of plot; title of experiment
  loop_type = ("wavenumber", "alpha")
  range1 = np.arange(0.05, 1.3, 2)
  range2 = np.arange(-2, 1.5, 5)
  n_steps = 1500
  n_sub_steps = dims[0]*dims[1]
  n_initial_steps = 0 #before measureing starts

  run_experiment(loop_type,  range1, range2)
import cmath
import pickle
import os
import datetime
import timeit
import argparse

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='get description')
  parser.add_argument('-n', '--notes', dest = 'notes',  type=str, nargs=1,
                    help='add experiment description with --notes', required=True)
  notes=parser.parse_args().notes

  #read input file for most arguments
  #take the ones that vary from argparser
  #overwrite the two relevant variables

  #decide on outputdir name, file name
  
  #run a single simulation as in run file

  run_experiment(loop_type,  range1, range2)
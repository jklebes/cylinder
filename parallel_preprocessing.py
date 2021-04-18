import cmath
import pickle
import os
import datetime
import timeit
import argparse
import numpy as np

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)

if __name__ == "__main__":

  #read args on what 2 things to vary
  parser = argparse.ArgumentParser(description='get description')
  parser.add_argument('-n', '--notes', dest = 'notes',  type=str, 
                    help='add experiment description with --notes', required=True)
  parser.add_argument('--var1name', dest = 'var1name',  type=str,  required=True)
  parser.add_argument('--var2name', dest = 'var2name',  type=str,  required=True)
  parser.add_argument('--var1range', dest = "var1range",  type=float, nargs=3, required=True)
  parser.add_argument('--var2range', dest = "var2range",  type=float, nargs=3, required=True)  
  
  args = parser.parse_args()
  notes=args.notes
  var1name=args.var1name
  var2name=args.var2name
  var1range_list=args.var1range
  var2range_list=args.var2range

  var1range = np.arange(var1range_list[0], var1range_list[1], var1range_list[2])
  var2range = np.arange(var2range_list[0], var2range_list[1], var2range_list[2])

  #write notes file
  notesfile = open("notes.txt", "w")
  notesfile.write(notes)
  notesfile.close()

  #write file of lines over varying parameters
  varfile = open("varfile.txt", "w")
  varfile.write(var1name +" "+var2name+ "\n")
  for var1 in var1range:
      for var2 in var2range:
          varfile.write(str(round(var1, 5)) + " " + str(round(var2, 5)) +"\n")
  varfile.close()


import pandas as pd
import glob
import numpy as np
from collections import defaultdict

if __name__ == "__main__":
  #should be in the relevant directory alread
  filenames = glob.glob("*_mean.csv")
  print(filenames)
  
  filename0 = filenames[0]
  data0= pd.read_csv(filename0, index_col=0, header=0)
  colnames = list(data0.columns)
  # collect averages from each simulation's final output file
  # one dataframe for each header in output file
  data=dict( [(colname, defaultdict(lambda: defaultdict())) for colname in colnames] )

  for filename in filenames:
    filedata= pd.read_csv(filename, index_col=0, header=0)
    vs = filename.split("_")
    #varname1=vs[0]
    #varname2=vs[2]
    filevar1 = vs[1]
    filevar2 = vs[3]
    for colname in colnames:
      data[colname][filevar1][filevar2]= filedata[colname]
  
  for colname in colnames:
    df = pd.DataFrame(data[colname])
    df.to_csv("colname.csv", index_col=0:)

  

 

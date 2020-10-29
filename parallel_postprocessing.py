
import pandas as pd
import glob

if __name__ == "__main__":
  #should be in the relevant directory alread
  filenames = glob.glob("*_mean.csv")
  print(filenames)
  """
  filename0 = filenames[0]
  colnames = []
  # collect averages from each simulation's final output file
  # one dataframe for each header in output file
  data={}
  for colname in colnames:
    data[colname] = pd.DataFrame()

  for filename in filenames:
    filedata= pd.read_csv(index_col=0)
    filevar1 =
    filevar2 =
    for colname in colnames:
    data[colname][filevar1][filevar2]= filedata[colname]
  """

  

 

#Ue current working directory and current modules
#$ -cwd -V
#$ -l h_rt=01:00:00

# Run the application passing in the input and output filenames
python ../../parallel_postprocessing.py

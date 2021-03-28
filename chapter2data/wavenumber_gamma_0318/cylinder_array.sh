#Ue current working directory and current modules
#$ -cwd -V
# Request Wallclock time of 6 hours
#$ -l h_rt=00:30:00

# Run the application passing in the input and output filenames
python ../../parallel_single_run.py --input "infile.txt" --varnames "wavenumber" "gamma" --varline "0.705" "1.85"

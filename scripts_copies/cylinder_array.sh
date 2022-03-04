#Ue current working directory and current modules
#$ -cwd -V
# Request Wallclock time of 6 hours
#$ -l h_rt=00:30:00

numlines=(wc -l $2)
varnames=$(sed -n "1p" $2)

# Tell SGE that this is an array job, with "tasks" numbered from 1 to 150
#$ -t 2-2501
#$ -tc 150

values=$(sed -n "$SGE_TASK_ID p" $2)
# Run the application passing in the input and output filenames
python ../../parallel_single_run.py --input $1 --varnames $varnames --varline $values

date=$(date +%F_%H%M%S)
mkdir cylinder/out/$date
cd cylinder/out/$date
cp ../../infile.txt ./
python3 ../../parallel_preprocessing.py -n "varying intrinsic curvature" --var1name "wavenumber" --var1range 0.005 2 .08 --var2name "intrinsic_curvature" --var2range -4 4 .08
qsub -N array ../../../cylinder_array.sh "infile.txt" "varfile.txt"
qsub -hold_jid array -N postp ../../../postprocessing.sh

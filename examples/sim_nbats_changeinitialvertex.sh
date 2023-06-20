#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time 01:00:00
#SBATCH --mem 10G
#SBATCH --array=0-4
#SBATCH -o limchannels_%A_%a.out
#SBATCH -e limchannels_%A_%a.err
#SBATCH --job-name=limchannels
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thejasvi.beleyur@ab.mpg.de

# JOB DESCRIPTION
# ~~~~~~~~~~~~~~~
# This script is a copy of sim_nbats_raytracing.sh
# Seeing that 8 and 16 bats somehow fail - but only partially 
# forced me to read through Kreissig's thesis. The effect of changing
# the intiial vertex to generate the Fundamental Loops is tested - and
# shown to be effective. Here I will do the same - except only run a few
# iterations to see if the effect it has is noticeable for our type of data. 


module load anaconda/3/2020.02
export LIBIOMP5_PATH=$ANACONDA_HOME/lib/libiomp5.so

# activate the environment 
conda init bash

source ~/.bashrc
conda activate /u/tbeleyur/conda-envs/fresh/


# and now run the file 
cd /u/tbeleyur/pydatemm/examples/

main_out=initialvertex_tests/
for numbats in 8
do
	echo "Now creating sim data and params for $numbats"
	if mkdir -p "$main_out/nbat${numbats}/"; then
	  # Do stuff with new directory
	  echo "Made new directory $main_out/nbat${numbats}"
	else
		:
	fi
done
# Run the simulated audio first and generate the ground truth audio and flight trajectories

# 8 BATS
python  multibatsimulation.py -nbats 8 -ncalls 5 -all-calls-before 0.1 -room-dim '4,9,3' -seed 82319 -input-folder 'initialvertex_tests/nbat8/' -ray-tracing True
# prepare parameter sets
# Run with all channels in the original order
python batsin_simaudio_tests.py -audiopath "initialvertex_tests/nbat8/8-bats_trajectory_simulation_raytracing-1.wav" -arraygeompath "multibat_stresstests/nbat8/mic_xyz_multibatsim.csv" -K 3 -maxloopres 5e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name "nbats8-origorder-82319" -step-size 1e-3 -num-jobs 5 -dest-folder "initialvertex_tests/nbat8/nbats8outdata" 

# Run with ch 1 as initial vertex 
python batsin_simaudio_tests.py -audiopath "initialvertex_tests/nbat8/8-bats_trajectory_simulation_raytracing-1.wav" -arraygeompath "multibat_stresstests/nbat8/mic_xyz_multibatsim.csv" -K 3 -maxloopres 5e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name "nbats8-ch1initial-82319" -step-size 1e-3 -num-jobs 5 -dest-folder "initialvertex_tests/nbat8/nbats8outdata"  -channels '1,0,2,3,4,5,6,7'

# Run with ch 7 as initial vertex 
python batsin_simaudio_tests.py -audiopath "initialvertex_tests/nbat8/8-bats_trajectory_simulation_raytracing-1.wav" -arraygeompath "multibat_stresstests/nbat8/mic_xyz_multibatsim.csv" -K 3 -maxloopres 5e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name "nbats8-ch7initial-82319" -step-size 1e-3 -num-jobs 5 -dest-folder "initialvertex_tests/nbat8/nbats8outdata"  -channels '7,0,1,2,3,4,5,6'


# now perform the tracking with the various parameter sets
echo "Now pydatemm begins..."
echo $(date)

echo "Beginning original channel order tracking"
python -m pydatemm -paramfile "initialvertex_tests/nbat8/nbats8outdata/paramset_nbats8-origorder-82319_${SLURM_ARRAY_TASK_ID}.yaml"
echo $(date)
echo "8 BAT original order tracking done"


echo "Beginning original channel 1 initial vertex tracking"
python -m pydatemm -paramfile "initialvertex_tests/nbat8/nbats8outdata/paramset_nbats8-ch1initial-82319_${SLURM_ARRAY_TASK_ID}.yaml"
echo $(date)
echo "8 BAT original order tracking done"


echo "Beginning original channel 7 initial vertex tracking"
python -m pydatemm -paramfile "initialvertex_tests/nbat8/nbats8outdata/paramset_nbats8-ch7initial-82319_${SLURM_ARRAY_TASK_ID}.yaml"
echo $(date)
echo "8 BAT original order tracking done"





echo $(date)

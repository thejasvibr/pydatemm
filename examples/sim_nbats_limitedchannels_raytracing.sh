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
# Seeing that 8 and 16 bats somehow fail in performance got me thinking about 
# the TOA overlaps caused by large arrays for each emitted sound. In this job submission I will
# test the effect of using small arrays that naturally lead to a lower spread in TOA across channels, 
# and thus a smaller chance of overlap in general. 

module load anaconda/3/2020.02
export LIBIOMP5_PATH=$ANACONDA_HOME/lib/libiomp5.so

# activate the environment 
conda init bash

source ~/.bashrc
conda activate /u/tbeleyur/conda-envs/fresh/


# and now run the file 
cd /u/tbeleyur/pydatemm/examples/

main_out=multibat_stresstests/
for numbats in 4 8 16
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

# 4 BATS
python  multibatsimulation.py -nbats 4 -ncalls 5 -all-calls-before 0.1 -room-dim '4,9,3' -seed 82319 -input-folder 'multibat_stresstests/nbat4/' -ray-tracing True
# prepare parameter sets
python batsin_simaudio_tests.py -audiopath "multibat_stresstests/nbat4/4-bats_trajectory_simulation_raytracing-1.wav" -arraygeompath "multibat_stresstests/nbat4/mic_xyz_multibatsim.csv" -K 3 -maxloopres 5e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name "nbats4-limchannels82319" -step-size 1e-3 -num-jobs 5 -dest-folder "multibat_stresstests/nbat4/nbats4outdata" -channels '0,1,3,5'

# 8 BATS
python  multibatsimulation.py -nbats 8 -ncalls 5 -all-calls-before 0.1 -room-dim '4,9,3' -seed 82319 -input-folder 'multibat_stresstests/nbat8/' -ray-tracing True
# prepare parameter sets
python batsin_simaudio_tests.py -audiopath "multibat_stresstests/nbat8/8-bats_trajectory_simulation_raytracing-1.wav" -arraygeompath "multibat_stresstests/nbat8/mic_xyz_multibatsim.csv" -K 3 -maxloopres 5e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name "nbats8-limchannels82319" -step-size 1e-3 -num-jobs 5 -dest-folder "multibat_stresstests/nbat8/nbats8outdata" -channels '0,1,3,5'

# 16 bats

python  multibatsimulation.py -nbats 16 -ncalls 5 -all-calls-before 0.2 -room-dim '4,9,3' -seed 82319 -input-folder 'multibat_stresstests/nbat16/' -ray-tracing True
# prepare parameter sets
python batsin_simaudio_tests.py -audiopath "multibat_stresstests/nbat16/16-bats_trajectory_simulation_raytracing-1.wav" -arraygeompath "multibat_stresstests/nbat16/mic_xyz_multibatsim.csv" -K 3 -maxloopres 5e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name "nbats16-limchannels82319" -step-size 1e-3 -num-jobs 5 -dest-folder "multibat_stresstests/nbat16/nbats16outdata" -channels '0,1,3,5'



# now perform the tracking
echo "Now pydatemm begins..."

echo $(date)

python -m pydatemm -paramfile "multibat_stresstests/nbat4/nbats4outdata/paramset_nbats4-limchannels82319_${SLURM_ARRAY_TASK_ID}.yaml"
echo "4 BAT tracking done"

echo $(date)
python -m pydatemm -paramfile "multibat_stresstests/nbat8/nbats8outdata/paramset_nbats8-limchannels82319_${SLURM_ARRAY_TASK_ID}.yaml"
echo "8 BAT tracking done"
echo $(date)
python -m pydatemm -paramfile "multibat_stresstests/nbat16/nbats16outdata/paramset_nbats16-limchannels82319_${SLURM_ARRAY_TASK_ID}.yaml"
echo "16 BAT tracking done"
echo $(date)

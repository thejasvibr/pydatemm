#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time 00:45:00
#SBATCH --mem 3G
#SBATCH --array=0-4
#SBATCH -o nbatseffect_%A_%a.out
#SBATCH -e nbatseffect_%A_%a.err
#SBATCH --job-name=nbatseffect
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thejasvi.beleyur@ab.mpg.de

module load anaconda/3/2020.02
export LIBIOMP5_PATH=$ANACONDA_HOME/lib/libiomp5.so

# activate the environment 
conda init bash

source ~/.bashrc
conda activate /u/tbeleyur/conda-envs/fresh/


# and now run the file 
cd /u/tbeleyur/pydatemm/examples/

main_out=multibat_stresstests/
for numbats in 2 4 8 16
do
	echo "Now creating sim data and params for $numbats"
	if mkdir -p "$main_out/nbat${numbats}/"; then
	  # Do stuff with new directory
	  echo "Made new directory $main_out/nbat${numbats}"
	else
		:
	fi
done
# Run the simulated audio first and generate the ground truth audio, arraygeom (true and noisy xyz) and flightpath files

# 2 BATS
python  multibatsimulation.py -nbats 2 -ncalls 5 -all-calls-before 0.1 -room-dim '4,9,3' -seed 78464 -input-folder 'multibat_stresstests/nbat2/' 
# prepare parameter sets
python batsin_simaudio_tests.py -audiopath "multibat_stresstests/nbat2/2-bats_trajectory_simulation_1-order-reflections.wav" -arraygeompath "multibat_stresstests/nbat2/mic_xyz_multibatsim.csv" -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name "nbats2" -step-size 1e-3 -num-jobs 5 -dest-folder "multibat_stresstests/nbat2/nbats2outdata"


# 4 BATS
python  multibatsimulation.py -nbats 4 -ncalls 5 -all-calls-before 0.1 -room-dim '4,9,3' -seed 78464 -input-folder 'multibat_stresstests/nbat4/' 
# prepare parameter sets
python batsin_simaudio_tests.py -audiopath "multibat_stresstests/nbat4/4-bats_trajectory_simulation_1-order-reflections.wav" -arraygeompath "multibat_stresstests/nbat4/mic_xyz_multibatsim.csv" -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name "nbats4" -step-size 1e-3 -num-jobs 5 -dest-folder "multibat_stresstests/nbat4/nbats4outdata"

# 8 BATS
python  multibatsimulation.py -nbats 8 -ncalls 5 -all-calls-before 0.1 -room-dim '4,9,3' -seed 78464 -input-folder 'multibat_stresstests/nbat8/' 
# prepare parameter sets
python batsin_simaudio_tests.py -audiopath "multibat_stresstests/nbat8/8-bats_trajectory_simulation_1-order-reflections.wav" -arraygeompath "multibat_stresstests/nbat8/mic_xyz_multibatsim.csv" -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name "nbats8" -step-size 1e-3 -num-jobs 5 -dest-folder "multibat_stresstests/nbat8/nbats8outdata"

# 16 bats

python  multibatsimulation.py -nbats 16 -ncalls 5 -all-calls-before 0.2 -room-dim '4,9,3' -seed 78464 -input-folder 'multibat_stresstests/nbat16/' 
# prepare parameter sets
python batsin_simaudio_tests.py -audiopath "multibat_stresstests/nbat16/16-bats_trajectory_simulation_1-order-reflections.wav" -arraygeompath "multibat_stresstests/nbat16/mic_xyz_multibatsim.csv" -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name "nbats16" -step-size 1e-3 -num-jobs 5 -dest-folder "multibat_stresstests/nbat16/nbats16outdata"



# now perform the tracking


python -m pydatemm -paramfile "multibat_stresstests/nbat2/nbats2outdata/paramset_nbats2_${SLURM_ARRAY_TASK_ID}.yaml"

python -m pydatemm -paramfile "multibat_stresstests/nbat4/nbats4outdata/paramset_nbats4_${SLURM_ARRAY_TASK_ID}.yaml"

python -m pydatemm -paramfile "multibat_stresstests/nbat8/nbats8outdata/paramset_nbats8_${SLURM_ARRAY_TASK_ID}.yaml"

python -m pydatemm -paramfile "multibat_stresstests/nbat16/nbats16outdata/paramset_nbats16_${SLURM_ARRAY_TASK_ID}.yaml"


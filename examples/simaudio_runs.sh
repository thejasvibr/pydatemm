#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time 00:20:00
#SBATCH --mem 4G
#SBATCH --array=0-4
#SBATCH -o simaudio_%A_%a.out
#SBATCH -e simaudio_%A_%a.err
#SBATCH --job-name=simaudio
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
# Run the simulated audio first and generate the ground truth audio, arraygeom and flightpath files
python  multibatsimulation.py
# setup the parameter files 

# original xyz
python batsin_simaudio_tests.py -audiopath simaudio_input/3-bats_trajectory_simulation_1-order-reflections.wav -arraygeompath simaudio_input/mic_xyz_multibatsim.csv -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name '0mmxyzerror' -step-size 1e-3 -num-jobs 5 -dest-folder simtests

# 5 mm xyz error 
python batsin_simaudio_tests.py -audiopath simaudio_input/3-bats_trajectory_simulation_1-order-reflections.wav -arraygeompath simaudio_input/mic_xyz_multibatsim_noisy0.005m.csv -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name '5mmxyzerror' -step-size 1e-3 -num-jobs 5 -dest-folder simtests

# 1 cm xyz error
python batsin_simaudio_tests.py -audiopath simaudio_input/3-bats_trajectory_simulation_1-order-reflections.wav -arraygeompath simaudio_input/mic_xyz_multibatsim_noisy0.01m.csv -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name '1cmxyzerror' -step-size 1e-3 -num-jobs 5 -dest-folder simtests

# 2.5 cm xyz error
python batsin_simaudio_tests.py -audiopath simaudio_input/3-bats_trajectory_simulation_1-order-reflections.wav -arraygeompath simaudio_input/mic_xyz_multibatsim_noisy0.025m.csv -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name '2.5cmxyzerror' -step-size 1e-3 -num-jobs 5 -dest-folder simtests


# 5 cm xyz error in mic array
python batsin_simaudio_tests.py -audiopath simaudio_input/3-bats_trajectory_simulation_1-order-reflections.wav -arraygeompath simaudio_input/mic_xyz_multibatsim_noisy0.05m.csv -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name '5cmxyzerror' -step-size 1e-3 -num-jobs 5 -dest-folder simtests


# 10 cm xyz error in mic array
python batsin_simaudio_tests.py -audiopath simaudio_input/3-bats_trajectory_simulation_1-order-reflections.wav -arraygeompath simaudio_input/mic_xyz_multibatsim_noisy0.1m.csv -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name '10cmxyzerror' -step-size 1e-3 -num-jobs 5 -dest-folder simtests


# run the localisation 
for errorrate in 0mm 5mm 1cm 2.5cm 5cm 10cm
do 
	echo Error rate: $errorrate being run now
	
	python -m pydatemm -paramfile simtests/paramset_${errorrate}xyzerror_$SLURM_ARRAY_TASK_ID.yaml
	echo Error rate: $errorrate done with run now 
done




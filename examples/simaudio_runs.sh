#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time 00:15:00
#SBATCH --mem 4G
#SBATCH --array=0-4
#SBATCH -o simaudio_%a.out
#SBATCH -e simaudio_%a.err
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
# Run the simulated audio first
python  multibatsimulation.py
# setup the parameter files 

# original xyz
python .\batsin_simaudio_tests.py -audiopath .\simaudio_input\3-bats_trajectory_simulation_1-order-reflections.wav -arraygeompath .\simaudio_input\mic_xyz_multibatsim.csv -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name 'origxyz' -step-size 1e-3 -num-jobs 5 -dest-folder simtests

# 5 cm xyz error in mic array
python .\batsin_simaudio_tests.py -audiopath .\simaudio_input\3-bats_trajectory_simulation_1-order-reflections.wav -arraygeompath .\simaudio_input\mic_xyz_multibatsim_noisy0.05m.csv -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name '5cmxyzerror' -step-size 1e-3 -num-jobs 5 -dest-folder simtests


# 10 cm xyz error in mic array
python .\batsin_simaudio_tests.py -audiopath .\simaudio_input\3-bats_trajectory_simulation_1-order-reflections.wav -arraygeompath .\simaudio_input\mic_xyz_multibatsim_noisy0.1m.csv -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name '10cmxyzerror' -step-size 1e-3 -num-jobs 5 -dest-folder simtests



# run the localisation 
python -m pydatemm -paramfile multibatsim_results/paramset_origxyz_$SLURM_ARRAY_TASK_ID.yaml

python -m pydatemm -paramfile multibatsim_results/paramset_5cmxyzerror_$SLURM_ARRAY_TASK_ID.yaml

python -m pydatemm -paramfile multibatsim_results/paramset_10cmxyzerror_$SLURM_ARRAY_TASK_ID.yaml


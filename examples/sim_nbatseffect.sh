#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time 00:20:00
#SBATCH --mem 4G
#SBATCH --array=0-4
#SBATCH -o simaudio_%A_%a.out
#SBATCH -e simaudio_%A_%a.err
#SBATCH --job-name=nbats-effect
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

mkdir multibat_stresstests

# Run the simulated audio first and generate the ground truth audio, arraygeom (true and noisy xyz) and flightpath files
for numbats in 2 4 8
do
    python  multibatsimulation.py -nbats $numbats -ncalls 5 -room-dim '4,9,3' -input-folder 'multibat_stresstests/nbat${numbats}/' -all-calls-before 0.1
    python batsin_simaudio_tests.py -audiopath multibat_stresstests/nbat${numbats}/${numbats}-bats_trajectory_simulation_1-order-reflections.wav -arraygeompath simaudio_input/mic_xyz_multibatsim.csv -K 3 -maxloopres 1e-4 -thresh-tdoaresidual 1e-8 -remove-lastchannel 'False' -min-peak-dist 0.25e-3 -window-size 0.01 -run-name 'nbats-${numbats}' -step-size 1e-3 -num-jobs 5 -dest-folder nbats_${numbats}
    #python -m pydatemm -paramfile nbats_${numbats}/paramset_nbats-${numbats}_${SLURM_ARRAY_TASK_ID}.yaml
done


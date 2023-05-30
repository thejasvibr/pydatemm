#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time 00:10:00
#SBATCH --mem 2G
#SBATCH --array=0-9
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
python batsin_simaudio_tests.py
# run the parameter files
python -m pydatemm -paramfile multibatsim_results/paramset_simaudio_$SLURM_ARRAY_TASK_ID.yaml



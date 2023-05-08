#!/bin/bash 
#SBATCH -n 2
#SBATCH -t 0-00:10
#SBATCH --mem 5G
#SBATCH -o batchlocalise_%A_%a.out
#SBATCH -e batchlocalise_%A_%a.err
#SBATCH --job-name=batchlocalise
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thejasvi.beleyur@bi.mpg.de

module load anaconda/3/2020.02
export LIBIOMP5_PATH=$ANACONDA_HOME/lib/libiomp5.so

# activate the environment 
conda init bash

source ~/.bashrc
conda activate /u/tbeleyur/conda-envs/fresh/

# and now run the file 
cd /u/tbeleyur/pydatemm/examples/
# setup the parameter files 
python sourcelocalising_1529543496_8000TMC.py
# run one of the parameter files
python -m pydatemm -paramfile paramset_1529543496_$SLURM_ARRAY_TASK_ID.yaml


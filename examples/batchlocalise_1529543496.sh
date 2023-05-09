#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time 00:20:00
#SBATCH --mem 20G
#SBATCH --array=7,11,13,14,16,18,19,22,24,25,29,32,36,39,40,41,43
#SBATCH -o output_batchlocalise-rd3_%a.out
#SBATCH -e error_batchlocalise-rd3_%a.err
#SBATCH --job-name=batchlocalise_rd3
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
python -m pydatemm -paramfile 1529543496_output/paramset_1529543496_$SLURM_ARRAY_TASK_ID.yaml


#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time 00:10:00
#SBATCH --mem 10G
#SBATCH --array=0-49
#SBATCH -o noisyxyz_%a.out
#SBATCH -e noisyxyz_%a.err
#SBATCH --job-name=noisyxyz
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
python preparing_parametersets_1529543496_8000TMC.py
# run one of the parameter files
python -m pydatemm -paramfile 1529543496_output/params_noisymicxyz_1529543496_$SLURM_ARRAY_TASK_ID.yaml


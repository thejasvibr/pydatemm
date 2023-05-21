#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --mem=10G
#SBATCH --array=0-24
#SBATCH -o 46136_%a.out
#SBATCH -e 46136_%a.err
#SBATCH --job-name=46136
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
python preparing_parametersets_1529546136_22000TMC.py
# run one of the parameter files
python -m pydatemm -paramfile 1529546136_output/origxyz__1529546136_$SLURM_ARRAY_TASK_ID.yaml


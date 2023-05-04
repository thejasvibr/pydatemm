#!/bin/bash 
#SBATCH -n 2
#SBATCH -t 0-00:2 
#SBATCH --mem 2000
#SBATCH -o output_miaow.txt
#SBATCH -e error_miaow.txt
#SBATCH --job-name=tng1
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
python batsin_simaudio_tests.py
# run one of the parameter files
python -m pydatemm -paramfile paramset_multibatsim_0.yaml


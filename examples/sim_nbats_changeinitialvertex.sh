#!/bin/bash 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time 01:00:00
#SBATCH --mem 8G
#SBATCH -o initialvertex.out
#SBATCH -e initialvertex.err
#SBATCH --job-name=initialvertex
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thejasvi.beleyur@ab.mpg.de

# JOB DESCRIPTION
# ~~~~~~~~~~~~~~~
# This script is a copy of sim_nbats_raytracing.sh
# Seeing that 8 and 16 bats somehow fail - but only partially 
# forced me to read through Kreissig's thesis. The effect of changing
# the intiial vertex to generate the Fundamental Loops is tested - and
# shown to be effective. Here I will do the same - except only run a few
# iterations to see if the effect it has is noticeable for our type of data. 
# The initial vertex is altered indirectly here by changing the channel order, 
# while keeping all of the channels in the array. 

module load anaconda/3/2020.02
export LIBIOMP5_PATH=$ANACONDA_HOME/lib/libiomp5.so

# activate the environment 
conda init bash

source ~/.bashrc
conda activate /u/tbeleyur/conda-envs/fresh/


# and now run the file 
cd /u/tbeleyur/pydatemm/examples/

main_out=initialvertex_tests/
for numbats in 8
do
	echo "Now creating sim data and params for $numbats"
	if mkdir -p "$main_out/nbat${numbats}/"; then
	  # Do stuff with new directory
	  echo "Made new directory $main_out/nbat${numbats}"
	else
		:
	fi
done
# Run the simulated audio first and generate the ground truth audio and flight trajectories

python initial_vertex_tests.py

k=6	
for ind in $(seq 0 7)
do 
	echo "Now running channel combi ${ind}"
	python -m pydatemm -paramfile "initialvertex_tests\\nbat8\\nbats8outdata\\paramset_K${k}-startch_${ind}.yaml"
	echo "Done running channel combi ${ind}"
done

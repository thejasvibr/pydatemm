'''
Localising overlapping calls: simulated audio case
==================================================
Here we'll run the 


'''
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
try:
    import pyvista as pv
except:
    print('Cant import pyvista!')
import soundfile as sf
from scipy.spatial import distance_matrix, distance
euclidean = distance.euclidean
import os
import subprocess
import time
import tqdm
import yaml

common_parameters = {}
common_parameters['audiopath'] = '3-bats_trajectory_simulation_1-order-reflections.wav'
#common_parameters['arraygeompath'] = 'mic_xyz_multibatsim.csv'
common_parameters['arraygeompath'] = 'mic_xyz_multibatsim_noisy0.05m.csv'
common_parameters['dest_folder'] = 'multibatsim_results'
common_parameters['K'] = 3
common_parameters['maxloopres'] = 1e-4
common_parameters['thresh_tdoaresidual'] = 1e-8 # s
common_parameters['remove_lastchannel'] = "False"
common_parameters['min_peak_dist'] = 0.25e-3 # s
common_parameters['window_size'] = 0.010 # s
common_parameters['run_name'] = 'simaudio'# the posix timestamp will be added later!

simdata = pd.read_csv('multibatsim_xyz_calling.csv').loc[:,'x':]
array_geom = pd.read_csv(common_parameters['arraygeompath']).loc[:,'x':'z'].to_numpy()

#%% Make the yaml file for the various time points
step_size = 0.001
window_size = 0.010
time_starts = np.arange(0, 0.5, step_size)

if not os.path.exists(common_parameters['dest_folder']):
    os.mkdir(common_parameters['dest_folder'])

# split the time_windows according to the total number of cores to be used.
split_timepoints = np.array_split(time_starts, 10)
#%%
for i, each in enumerate(split_timepoints):
    common_parameters['start_time'] = str(each.tolist())[1:-1]
    
    fname = os.path.join(common_parameters['dest_folder'], 
                         f'paramset_{common_parameters["run_name"]}_{i}.yaml')
    ff = open(fname, 'w+')
    yaml.dump(common_parameters, ff)


'''
Localising overlapping calls: 2018-06-21
========================================
21st June 2018 is special as the microphone positions were exactly
measured using a TotalStation. Here let's try to localise sources in the audio file 
with POSIX timestamp 1529543496. This audio file corresponds to P00/8000 TMC of
the thermal cameras. 


Let's also create a new mic xyz set with <= 5cm positional error. 

'''
import glob
from natsort import natsorted
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
#import pyvista as pv
import soundfile as sf
from scipy.spatial import distance_matrix, distance
euclidean = distance.euclidean
import os
import subprocess
import time
import tqdm
import yaml

common_parameters = {}
common_parameters['audiopath'] = '1529543496_input/video_synced10channel_first15sec_1529543496.WAV'
common_parameters['arraygeompath'] = '1529543496_input/Sanken9_centred_mic_totalstationxyz.csv'
common_parameters['dest_folder'] = '1529543496_output'
common_parameters['K'] = 3
common_parameters['maxloopres'] = 1e-4
common_parameters['min_peak_dist'] = 0.35e-4 # s
common_parameters['thresh_tdoaresidual'] = 1e-8 # s
common_parameters['remove_lastchannel'] = "False"
common_parameters['highpass_order'] = "2,20000"
common_parameters['run_name'] = 'params_originalmicxyz'

array_geom = pd.read_csv(common_parameters['arraygeompath']).loc[:,'x':'z'].to_numpy()

#%% Make the yaml file for the various time points
step_size = 0.003
window_size = 0.016
common_parameters['window_size'] = window_size
time_starts = np.arange(12.4, 13.4, step_size)

if not os.path.exists(common_parameters['dest_folder']):
    os.mkdir(common_parameters['dest_folder'])

# incoporate the time windows into the parameter file
relevant_time_windows = np.around(time_starts, 3)
# split the time_windows according to the total number of paramsets to be generated
split_timepoints = np.array_split(relevant_time_windows, 50)
#%%
for i, each in enumerate(split_timepoints):
    common_parameters['start_time'] = str(each.tolist())[1:-1]
    
    fname = os.path.join(common_parameters['dest_folder'], 
                         f'{common_parameters["run_name"]}_1529543496_{i}.yaml')
    ff = open(fname, 'w+')
    yaml.dump(common_parameters, ff)


#%% Noisy array geometry data
#   ~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we'll now add noise to the originally measured mic xyz so that the 
# overall euclidean error is <= 5 cm. 
np.random.seed(78464)
noise_max = 0.05
noisy_xyz = []
for xyz in array_geom:
    noisy_mic = xyz.copy()
    target_achieved = False
    while not target_achieved:
        noisy_mic = xyz + np.around(np.random.normal(0,noise_max,3),3)
        if euclidean(noisy_mic, xyz)<=noise_max:
            noisy_xyz.append(noisy_mic)
            target_achieved = True
noisy_micxyz = np.array(noisy_xyz)
noisy_micxyz = noisy_micxyz - noisy_micxyz[0,:]
noisy_arraygeom = pd.read_csv(common_parameters['arraygeompath'])
noisy_arraygeom.loc[:,'x':'z'] = noisy_micxyz
noisy_arraygeomfile = '1529543496_input/noisy-leq5cm_Sanken9_centred_mic_totalstationxyz.csv'
noisy_arraygeom.to_csv(noisy_arraygeomfile)


#%% 
# Change parameter values

common_parameters['arraygeompath'] = noisy_arraygeomfile 
common_parameters['run_name'] = 'params_noisymicxyz'

for i, each in enumerate(split_timepoints):
    common_parameters['start_time'] = str(each.tolist())[1:-1]
    
    fname = os.path.join(common_parameters['dest_folder'], 
                         f'{common_parameters["run_name"]}_1529543496_{i}.yaml')
    ff = open(fname, 'w+')
    yaml.dump(common_parameters, ff)




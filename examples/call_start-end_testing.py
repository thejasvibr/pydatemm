#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Presence-absence call emission detection
========================================
The guiding logic behind having DATEMM style algorithms is that I can detect
the start and end of a call just based on whether I can find a matching localised
source with a constant position over 3-5 ms (typical bat call duration). I've never
tested this though, and this module will check it using speaker playbacks from 
the Ushichka dataset. 

First we will create the bash script - and then call the bash script to run
parallel jobs for us. 


"""
import glob
import natsort
import os
import numpy as np 
import pandas as pd
from scipy import spatial 
euclidean = spatial.distance.euclidean

step_size = 2e-3
start_times = np.arange(4.38, 4.39, step_size)
window_size = 8e-3 # seconds
end_times = start_times + window_size

audio_file = 'ushichka_data/frame-synced_multichirp_2018-08-18_09-15-06_spkrplayback_ff802_10mic_snkn9_origin.wav'
arraygeom_file = 'ushichka_data/2018-08-17_ff802_10mic_xyz.csv'
common_command_LHS = 'python ushichka_2018-08-17_speakerplayback.py'
common_command_RHS = f' -K 3 -audiopath {audio_file} -arraygeompath {arraygeom_file} -dest_folder presence-absence'

with open ('presence-absence-testing.sh', 'w') as rsh:
    for start, end in zip(start_times, end_times):
        rsh.writelines(common_command_LHS + f'  -timewindow {start:.3f},{end:.3f}' + common_command_RHS+ "& \n")
#%%
# Run the bash script
os.system("bash presence-absence-testing.sh")

# kill any long-runnign processes manually...


#%%
# Load all the candidate files 
def keyfunc(X):
    ff = os.path.split(X)[-1]
    ff_brackets  = (ff.index('['), ff.index(']'))
    substr = ff[ff_brackets[0]+1:ff_brackets[1]]
    start = substr.split(',')[0]
    return float(start)

csvfiles = glob.glob('presence-absence/*.csv')
sortfiles = natsort.natsorted(csvfiles, key=keyfunc)
sources = [pd.read_csv(each) for each in sortfiles]
all_sources = pd.concat(sources).reset_index(drop=True).loc[:,['x','y','z','tdoa_res','label','t_start','t_end']]
sources_by_chunk = all_sources.groupby('t_end')
#%% Run the bash script and now collect the results
# First load the interpolated XYZ speaker trajectory from camera tracking. 
# 1) How muchh is the TOF range?
mic_xyz = pd.read_csv('ushichka_data/2018-08-17_ff802_10mic_xyz.csv').loc[:,'x':'z'].to_numpy()
interp_xyz = pd.read_csv('ushichka_data/interpolated_1000TMCxyzpoints.csv')

for t_end, sub_df in sources_by_chunk:
    best_fit = (interp_xyz['t']-t_end-0.01).abs().idxmin()
    camera_location  = interp_xyz.loc[best_fit,'x':'z']
    # REAL sources don't have crazy small TDOA residuals - remove anything that has TDOA res of ~1e-16 or lower
    real_sources = sub_df
    if real_sources.shape[0]>0:    
        source_to_groundtruth = real_sources.apply(lambda X: euclidean(camera_location, X['x':'z']), 1)
        
        if source_to_groundtruth.min() <= 0.3:
            source = real_sources.loc[source_to_groundtruth.idxmin(),'x':'z'].to_numpy()
            mic_source_distances = spatial.distance_matrix(source.reshape(-1,3), mic_xyz)
            print(f'{t_end-window_size},{t_end} has call', source)
            print(f'tdoa res: {real_sources.loc[source_to_groundtruth.idxmin(),"tdoa_res"]}')
            print(f'camera-based location: {camera_location.to_numpy()}\n')
            


    #%% 
# How much is the TOF range? 
fs = 192000
audio, fs = sf.read('ushichka_data/frame-synced_multichirp_2018-08-18_09-15-06_spkrplayback_ff802_10mic_snkn9_origin.wav',
                    start=int(fs*0.186), stop=int(fs*0.196))
#%%
plt.figure()
plt.specgram(audio[:,-2], Fs=fs)



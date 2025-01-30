# -*- coding: utf-8 -*-
"""
TDE map based CCG implementation 
================================

Created on Sat May 11 08:59:30 2024

@author: theja
"""
import matplotlib.pyplot as plt
import numpy as np 
import os 
from pydatemm.tde_map import compute_gcc_scores_overlapping, range_difference
import tqdm
import scipy.signal as signal 
import pandas as pd
from scipy.spatial import distance_matrix
import soundfile as sf
from itertools import combinations
from skimage.filters import rank
from skimage.morphology import footprints
import tqdm

# audio data
min_time, max_time = 1.7, 2.2
bataudio_path = os.path.join('batsim')
audiofile_path = os.path.join(bataudio_path, '5bat_5-bats_trajectory_simulation_1-order-reflections.wav')
fs = sf.info(audiofile_path).samplerate
# flight paths
xyz_filepath = os.path.join(bataudio_path,'5bat_multibatsim_xyz_calling.csv' )
xyz = pd.read_csv(xyz_filepath)
xyz = xyz.rename(columns={'batid':'batnum'})
all_flight_path = xyz.copy()
by_batnum = all_flight_path.groupby('batnum')

# get flight path of one of the bats
focal_batnum = 1
senders = all_flight_path.groupby('batnum').get_group(focal_batnum).loc[:,'x':'z'].to_numpy()
sender_times = all_flight_path.groupby('batnum').get_group(1)['t'].to_numpy()
rows_within_timerange = np.logical_and(sender_times>=min_time, sender_times<=max_time)
sender_times = sender_times[rows_within_timerange]
trajectory = senders[rows_within_timerange,:]
# load audio and scale to 16 bits
audio, fs = sf.read(audiofile_path, start=int(fs*min_time), stop=int(max_time*fs))
audio *= -1 + 2**15
b,a = signal.butter(2, np.array([20e3, 90e3])/fs, 'bandpass')
audio = np.apply_along_axis(lambda X: signal.filtfilt(b,a,X), 0, audio)

micxyz = pd.read_csv(os.path.join(bataudio_path, '5batmic_xyz_multibatsim.csv')).loc[:,'x':'z']
micxyz = micxyz.to_numpy()

#%%
# Generate GCC maps 
v_sound = 343 # m/s
max_intermic_delay = distance_matrix(micxyz, micxyz).max()/v_sound
chunk_length = int(fs*max_intermic_delay) + int(fs*3e-3) # max intermic delay + some extra
target_overlap = 2e-3 # amount you want the audio chunks to jump
target_overlap = 1 - int(fs*target_overlap)/chunk_length
gcc_map, t_chunks = compute_gcc_scores_overlapping(audio, fs, chunk_length, overlap=target_overlap)
t_chunks += min_time

chpairs = [sorted(each, reverse=True) for each in combinations(range(micxyz.shape[0]), 2)]

#%%
# Pass an entropy filter to extract regions with more structure.
# Each call's GCC 'footprint' looks like a horizontal bar.
entropy_footprint = footprints.ellipse(3,1) 
gcc_map_entropy = np.zeros(gcc_map.shape)
for chpair in tqdm.tqdm(chpairs):
    ch1, ch2 = chpair
    gcc_map_entropy[ch1, ch2,:,:] = rank.entropy(gcc_map[ch1,ch2,:,:], footprint=entropy_footprint)
    gcc_map_entropy[ch2, ch1,:,:] = np.flipud(gcc_map_entropy[ch1, ch2,:,:])


#%%
# Calculate predicted range-difference across all mic pairs

rangediffs = {}
for chpair in chpairs:
    rangediffs[tuple(chpair)] = range_difference(chpair, micxyz, trajectory)
    rangediffs[tuple(chpair[::-1])] = -rangediffs[tuple(chpair)]

#%% 
rangediff_min, rangediff_max = -(chunk_length/fs)*v_sound*0.5, (chunk_length/fs)*v_sound*0.5
ch1, ch2 = 1,0
plt.figure()
a = plt.subplot(211)
plt.imshow(gcc_map[ch1,ch2,:,:], aspect='auto',cmap="Set2",
           extent=[t_chunks[0], t_chunks[-1], rangediff_max, rangediff_min], interpolation='none')

plt.plot(sender_times, rangediffs[(ch1,ch2)])
plt.subplot(212, sharex=a, sharey=a)
plt.imshow(gcc_map_entropy[ch1,ch2,:,:], aspect='auto',cmap="Set2",
           extent=[t_chunks[0], t_chunks[-1], rangediff_max, rangediff_min], interpolation='none')

plt.plot(sender_times, rangediffs[(ch1,ch2)])


#%%
rangediff_tolerance = 0.15 # m
within_rangediff_tolerance = np.zeros(gcc_map.shape)
delay_values = np.linspace(rangediff_min, rangediff_max, gcc_map.shape[2])
# get all rows that fall within predicted rangediff and +/- tolerance and make a mask
valid_rows = []
for i, t in enumerate(t_chunks):
    for chpair in chpairs:
        ch1, ch2 = chpair
        valid_rows = np.where( abs(delay_values-rangediffs[(ch1,ch2)][i])<=rangediff_tolerance)
        within_rangediff_tolerance[ch1,ch2,valid_rows,i] = 1
        within_rangediff_tolerance[ch2,ch1,:,i] = within_rangediff_tolerance[ch1,ch2,:,i][::-1]
within_rangediff_tolerance = np.array(within_rangediff_tolerance, dtype=np.bool_)
#%%
# Segment and get all valid regions that are at least X ms long. 

ch1, ch2 = 1,0
plt.figure()
a = plt.subplot(211)
plt.imshow(gcc_map[ch1,ch2,:,:], aspect='auto',cmap="Set2",
           extent=[t_chunks[0], t_chunks[-1], rangediff_max, rangediff_min], interpolation='none')

plt.plot(sender_times, rangediffs[(ch1,ch2)])
plt.subplot(212, sharex=a, sharey=a)
plt.imshow(within_rangediff_tolerance[ch1,ch2,:,:], aspect='auto',cmap="Set2",
           extent=[t_chunks[0], t_chunks[-1], rangediff_max, rangediff_min], interpolation='none')

plt.plot(sender_times, rangediffs[(ch1,ch2)])


#%% Now 



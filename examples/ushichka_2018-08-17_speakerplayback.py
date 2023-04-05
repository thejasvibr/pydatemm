#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracking a moving speaker in a reverberant cave
===============================================
Here we will try to track a moving speaker playing back the '9-chirp' playback
sequence. The playback consists of 3 sweep types (linear, logarithmic, bidirectional)
, of 3 different durations (6, 12, 24 ms). Each sweep occurs at the end of a 
200 ms audio chunk - i.e. the 6 ms sweep has 194 ms silence with the sweep at the
end, and so on. After a set of 9 chirps are played, there is a 200 ms silence, 
followed again by a repeat of the 9 chirp sequence and so on. 

Only the first chirp has a chunk size of 180 ms, which means a 6ms sweep with 
174 ms silence preceding it. This size difference is because half of a frame 
(at 40 ms inter-frame-interval) is cut while aligning the audio with the video
frame points. 

The speaker was moved around in the cave as it played the chirps. 3 thermal
cameras were used to track the position of a hand-warmer pack stuck above the 
speaker. The microphone and speaker xyz positions were obtained by camera
triangulation. 


"""
import argparse
import datetime as dt
import numpy as np 
import os
import pathlib
import pandas as pd
from pydatemm import generate_candidate_sources
import scipy.signal as signal 
import matplotlib.pyplot as plt 
plt.rcParams['agg.path.chunksize'] = 10000
import soundfile as sf
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN



parser = argparse.ArgumentParser(description="Run generate_candidate_sources for a given time snippet of an audio file",
                                 )
parser.add_argument('-audiopath', type=pathlib.Path,
                    help="Path to the multichannel audio wav file",
                    default="ushichka_data/frame-synced_multichirp_2018-08-18_09-15-06_spkrplayback_ff802_10mic_snkn9_origin.wav")
parser.add_argument('-arraygeompath', type=pathlib.Path,
                    help="Path to array xyz csv file",
                    default="ushichka_data/2018-08-17_ff802_10mic_xyz.csv")
parser.add_argument('-timewindow', type=str,
                    default="0.18,0.20",
                    help="start and end time in seconds separated by a commat e.g. -timewindow 0.1,0.2")
parser.add_argument('-remove_lastchannel', type=bool, default=True, help="Whether to remove the last channel or not. The last channel in the Ushichka data is the camera sync\
square wave")
parser.add_argument('-thresh_tdoaresidual', type=float, default=0.1e-3, help="Threshold in seconds to filter out positions with higher TDOA residual. Defaults to 0.1 ms")
parser.add_argument('-dest_folder', type=pathlib.Path, default='./', help='Destination folder. Even if folder doesnt exist, it will be created.')
parser.add_argument('-K', type=int, default=4, help="Number of peaks to choose per channel pair. ")

args = parser.parse_args()

if not os.path.exists(args.dest_folder):
    os.mkdir(args.dest_folder)

now = dt.datetime.now()
#%%
# Utility functions
def conv_to_numpy(pydatemm_out):
    return np.array([np.array(each) for each in pydatemm_out]).reshape(-1,4)
# Run DBSCAN to simplify these datapoints!
def get_3d_median_location(xyz):
    return np.apply_along_axis(np.median, 0, xyz)
#%%
# Setting up tracking
vsound = 343.0 # m/s
timewindow = list(map(lambda X: float(X), args.timewindow.split(",")))
fs = sf.info(args.audiopath).samplerate
start_ind, stop_ind = int(timewindow[0]*fs), int(timewindow[1]*fs)
audio, fs = sf.read(args.audiopath, start=start_ind, stop=stop_ind+1)
sync = audio[:,-1] 
if args.remove_lastchannel:
	mic_audio = audio[:,:-1]
else:
    mic_audio = audio.copy()
b,a = signal.butter(2, 10000/(2*fs), 'highpass')
mic_audio = np.apply_along_axis(lambda X: signal.filtfilt(b,a,X), 1, mic_audio)
array_geom = pd.read_csv(args.arraygeompath).loc[:,'x':'z'].to_numpy()

#%%
# 
nchannels = mic_audio.shape[1]
kwargs = {'nchannels':nchannels,
          'fs':fs,
          'array_geom':array_geom,
          'pctile_thresh': 95,
          'use_gcc':True,
          'gcc_variant':'phat', 
          'min_peak_diff':0.35e-4, 
          'vsound' : 343.0}
kwargs['max_loop_residual'] = 0.5e-4
tdoa_resid_threshold = args.thresh_tdoaresidual

max_delay = np.max(distance_matrix(array_geom, array_geom))/kwargs['vsound']  
kwargs['K'] = args.K
kwargs['num_cores'] = 2
#%%
# Keep things simple for now, save some time and check only specific chunks of 
# audio that have the playbacks. 

print(f'Generating candidate sources for {timewindow}...')
output = generate_candidate_sources(audio, **kwargs)
print(f'Done generating candidate sources for {timewindow}...')

wavfilename = os.path.split(args.audiopath)[-1].split('.')[0]

csv_fname = os.path.join(args.dest_folder, f'{wavfilename}_{timewindow}.csv')

if len(output.sources)>0:
    posns = conv_to_numpy(output.sources)
    # get rid of -999 entries
    no_999 = np.logical_and(posns[:,0]!=-999, posns[:,1]!=-999)
    posns_filt = posns[no_999,:]
    posns_filt = posns_filt[posns_filt[:,-1]<tdoa_resid_threshold]
else:
    posns_filt = np.array([])
    
if posns_filt.shape[0]>0:
    print(f'Starting clustering for {timewindow}...Num candidates sources: {posns_filt.shape[0]}')
    # There're a lot of 'repeat' localisations - simplify and bring down the massive 
    # localisations to the minimum unique set. 

    posns_filt_str = np.char.array(posns_filt)
    spacer = np.char.array(np.tile(-999,posns_filt.shape[0]))
    all_rows_combined = posns_filt_str[:,0] +spacer+ posns_filt_str[:,1] + spacer+posns_filt_str[:,2] + spacer+posns_filt_str[:,3]

    unique_elements, unique_inds, counts= np.unique(all_rows_combined, return_index=True, return_counts=True)
    unique_posns_filt = posns_filt[unique_inds,:]
    print(f'Num unique candidates sources: {unique_posns_filt.shape[0]}')
    # perform DBSCAN only if the # of particles is very high (e.g. > 20000)
    if unique_posns_filt.shape[0]>20000:
        
        clustered = DBSCAN(eps=0.1).fit(unique_posns_filt)
        # clustered = DBSCAN(eps=0.3, min_samples=20, algorithm='ball_tree').fit(posns_filt)
        print(f'Done running DBSCAN for {timewindow}...')
        labels = clustered.labels_
        valid_labels = np.unique(labels)
        all_centres = []
        for each in valid_labels :
            rows = clustered.labels_ == each
            centre = get_3d_median_location(unique_posns_filt[rows,:])
            all_centres.append(np.append(centre, each))
        df = pd.DataFrame(all_centres)
        dbscan_timepoint = dt.datetime.now()
        print(f'Time taken with DBSCAN included: {dbscan_timepoint-now}')
    else:
        df = pd.DataFrame(unique_posns_filt)
        df['label'] = np.nan
    df['t_start'] = timewindow[0]
    df['t_end'] = timewindow[1]
    
    df.columns = ['x','y','z','tdoa_res','label','t_start','t_end']
    df.to_csv(csv_fname)

    print(f'COMPLETED RUN FOR {timewindow}')
else:
    df = pd.DataFrame(data={}, columns = ['x','y','z','tdoa_res','label','t_start','t_end'])
    
    print('No sources found in {timewindow}...')
df.to_csv(csv_fname)
stop = dt.datetime.now()
print(f'Time taken start-stop: {stop-now}')
#%% 
# import scipy.interpolate as interp
# xyz_camera = pd.read_csv('ushichka_data/DLTdv7_data_2018-08-17_P03_1000TMCxyzpts.csv')
# xyz_camera.columns = ['x','y','z']
# xyz_camera['t'] = xyz_camera.index/25
# xyz_camera = xyz_camera.dropna()
# t_interp = np.arange(0,4.75,0.04)
# xyz_interp_fn = {axis : interp.interp1d(xyz_camera['t'], xyz_camera[axis], kind='cubic') for axis in ['x','y','z'] }
# xyz_interp = pd.DataFrame(data={axis: xyz_interp_fn[axis](t_interp) for axis in ['x','y','z']})
# xyz_interp['t'] = t_interp
# xyz_interp['frame'] = np.int64(t_interp/0.04)
# #%%
# plt.figure()
# a0 = plt.subplot(111, projection='3d')
# plt.plot(xyz_interp['x'], xyz_interp['y'], xyz_interp['z'], '*')
# plt.plot(xyz_interp['x'][0], xyz_interp['y'][0], xyz_interp['z'][0], 'p')
# plt.plot(array_geom[:,0], array_geom[:,1], array_geom[:,2], 'g*')


# #%%
# import scipy.spatial as spatial
# euclid = spatial.distance.euclidean
# t_emission = timewindow[0]-0.01
# cam_tem_xyz = xyz_interp.loc[(xyz_interp['t']-t_emission).abs().idxmin(),:].loc['x':'z'].to_numpy()
# closest_emission_point = np.apply_along_axis(lambda X: euclid(X[:3],cam_tem_xyz),1,unique_posns_filt)
# print(unique_posns_filt[np.argmin(closest_emission_point),:3],
#       cam_tem_xyz,
#       np.min(closest_emission_point))


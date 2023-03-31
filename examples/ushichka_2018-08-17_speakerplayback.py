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
                    help="Path to the multichannel audio wav file")
parser.add_argument('-arraygeompath', type=pathlib.Path,
                    help="Path to array xyz csv file")
parser.add_argument('-timewindow', type=str,
                    help="start and end time in seconds separated by a commat e.g. -timewindow 0.1,0.2")
parser.add_argument('-thresh_tdoaresidual', type=float, default=0.1e-3, help="Threshold in seconds to filter out positions with higher TDOA residual. Defaults to 0.1 ms")

args = parser.parse_args()
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
mic_audio = audio[:,:-1]
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
kwargs['K'] = 4
kwargs['num_cores'] = 2
#%%
# Keep things simple for now, save some time and check only specific chunks of 
# audio that have the playbacks. 

print(f'Generating candidate sources for {timewindow}...')
output = generate_candidate_sources(audio, **kwargs)
print(f'Done generating candidate sources for {timewindow}...')

if len(output.sources)>0:
    posns = conv_to_numpy(output.sources)
    # get rid of -999 entries
    no_999 = np.logical_and(posns[:,0]!=-999, posns[:,1]!=-999)
    posns_filt = posns[no_999,:]
    posns_filt = posns_filt[posns_filt[:,-1]<tdoa_resid_threshold]

if posns_filt.shape[0]>0:
    print(f'Running DBSCAN for {timewindow}...Num candidates sources: {posns_filt.shape[0]}')
    clustered = DBSCAN(eps=0.3, min_samples=20).fit(posns_filt)
    print('Done running DBSCAN for {timewindow}...')
    labels = clustered.labels_
    valid_labels = np.unique(labels[labels!=-1])
    all_centres = []
    for each in valid_labels :
        rows = clustered.labels_ == each
        centre = get_3d_median_location(posns_filt[rows,:])
        all_centres.append(np.append(centre, each))
    df = pd.DataFrame(all_centres)
    df['t_start'] = timewindow[0]
    df['t_end'] = timewindow[1]
    wavfilename = os.path.split(args.audiopath)[-1].split('.')[0]
    csv_fname = f'{wavfilename}_{timewindow}.csv'
    df.columns = ['x','y','z','tdoa_res','label','t_start','t_end']
    df.to_csv(csv_fname)
    print(f'COMPLETED RUN FOR {timewindow}')
else:
    print('No sources found in {timewindow}...')
#%% 
# 

    
    
    




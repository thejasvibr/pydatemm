'''
Localising overlapping calls: simulated audio case
==================================================

'''
import numpy as np 
import pandas as pd
import soundfile as sf
from scipy.spatial import distance_matrix, distance
euclidean = distance.euclidean
import time
import tqdm
from pydatemm import generate_candidate_sources

filename = '3-bats_trajectory_simulation_1-order-reflections.wav'
try:
    fs = sf.info(filename).samplerate
    array_audio, fs = sf.read(filename, stop=int(0.2*fs))
except:
    import multibatsimulation as multibat
    fs = sf.info(filename).samplerate
    array_audio, fs = sf.read(filename, stop=int(0.2*fs))

array_geom = pd.read_csv('multibat_sim_micarray.csv').to_numpy()[:,1:]
simdata = pd.read_csv('multibat_xyz_emissiontime.csv')

nchannels = array_audio.shape[1]
kwargs = {'nchannels':nchannels,
          'fs':fs,
          'array_geom':array_geom,
          'pctile_thresh': 95,
          'use_gcc':True,
          'gcc_variant':'phat', 
          'min_peak_diff':0.35e-4, 
          'vsound' : 343.0}
kwargs['max_loop_residual'] = 0.5e-4

max_delay = np.max(distance_matrix(array_geom, array_geom))/343  
kwargs['K'] = 4
kwargs['num_cores'] = 2

#%%
# i = 110 -- tricky one , 120 even worse
# start_time = 0.030

time_pts = np.arange(0.0015, 0.020, 0.5e-3)

results = {}
for start_time in tqdm.tqdm(time_pts):    
    end_time = start_time  + max_delay
    start_sample, end_sample = int(fs*start_time), int(fs*end_time)  
    sim_audio = array_audio[start_sample:end_sample]
    output = generate_candidate_sources(sim_audio, **kwargs)
    results[start_time] = output.sources
    
#%%

first_call = simdata[simdata['t']<0.005]
emission_point = first_call.loc[:, 'x':'z'].to_numpy().flatten()

source_present = np.zeros(time_pts.size)    
for i,t in enumerate(time_pts):
    for each in results[t]:
        if euclidean(np.array(each[:-1]), emission_point)< 1e-2:
            print(np.array(each))
            source_present[i] += 1
            break
            
            
    
#%% When can we expect to detect a call emitted at 0.002 s
source_to_mic = []
for each in array_geom:
    source_to_mic.append(euclidean(each, emission_point))

arrival_times = 0.007 + np.array(source_to_mic)/343.0

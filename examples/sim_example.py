'''
Localising overlapping calls: simulated audio case
==================================================

'''
import numpy as np 
import pandas as pd
import soundfile as sf
from scipy.spatial import distance_matrix
import time
from pydatemm.source_generation import generate_candidate_sources_v2

#import multibatsimulation as multibat

filename = '3-bats_trajectory_simulation_1-order-reflections.wav'
try:
    fs = sf.info(filename).samplerate
    array_audio, fs = sf.read(filename, stop=int(0.2*fs))
except:
    import multibatsimulation as multibat
    fs = sf.info(filename).samplerate
    array_audio, fs = sf.read(filename, stop=int(0.2*fs))

array_geom = pd.read_csv('multibat_sim_micarray.csv').to_numpy()[:,1:]

nchannels = array_audio.shape[1]
kwargs = {'nchannels':nchannels,
          'fs':fs,
          'array_geom':array_geom,
          'pctile_thresh': 95,
          'use_gcc':True,
          'gcc_variant':'phat', 
          'min_peak_diff':0.35e-4, 
          'vsound' : 343.0, 
          'no_neg':False}
kwargs['max_loop_residual'] = 0.5e-4
kwargs['K'] = 5
dd = np.max(distance_matrix(array_geom, array_geom))/343  
dd_samples = int(kwargs['fs']*dd)

ignorable_start = int(0.01*fs)
shift_samples = 96
start_samples = np.arange(ignorable_start,array_audio.shape[0], shift_samples)
end_samples = start_samples+dd_samples
max_ind = int(0.010*1e3*2)
max_time = max_ind*(shift_samples/fs)+ignorable_start/fs
#%%
# i = 110 -- tricky one , 120 even worse
i = 20
sta = time.perf_counter()
audio_chunk = array_audio[start_samples[i]:end_samples[i]]
position_data, cfl_ids, tdedata = generate_candidate_sources_v2(audio_chunk, **kwargs)
sto = time.perf_counter()
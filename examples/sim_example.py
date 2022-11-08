'''
Localising overlapping calls: simulated audio case
==================================================

'''
import numpy as np 
import pandas as pd
import soundfile as sf
from scipy.spatial import distance_matrix
import time
from pydatemm.source_generation import generate_candidate_sources

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
#%%
# i = 110 -- tricky one , 120 even worse
start_time = 0.030
end_time = start_time  + max_delay
start_sample, end_sample = int(fs*start_time), int(fs*end_time)

kwargs['K'] = 4
kwargs['num_cores'] = 4


sim_audio = array_audio[start_sample:end_sample]
sim_audio = np.zeros(sim_audio.shape)

a = time.perf_counter_ns()/1e9    
generate_candidate_sources(sim_audio, **kwargs)
b = time.perf_counter_ns()/1e9
print(f'{num_cores} cores: {b-a} s')

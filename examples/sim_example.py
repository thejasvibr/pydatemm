'''
Localising overlapping calls: simulated audio case
==================================================

'''
import numpy as np 
import pandas as pd
import soundfile as sf
from scipy.spatial import distance_matrix
import time
from pydatemm.source_generation import generate_candidate_sources_v2, localise_sounds_v2, create_tde_data
from pydatemm.source_generation import chunk_create_tde_data, pll_create_tde_data, get_tde
#import multibatsimulation as multibat
#from memory_profiler import profile

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
          'vsound' : 343.0, 
          'no_neg':False}
kwargs['max_loop_residual'] = 0.5e-4
kwargs['K'] = 5
max_delay = np.max(distance_matrix(array_geom, array_geom))/343  
#%%
# i = 110 -- tricky one , 120 even worse
start_time = 0.030
end_time = start_time  + max_delay
start_sample, end_sample = int(fs*start_time), int(fs*end_time)

kwargs['num_cores'] = 8

# for num_cores in [8]:
    
#     sta = time.perf_counter()
    
#     sto = time.perf_counter()
#     print(f'{num_cores} Cores: {sto-sta} s taken')

#%%

audio_chunk = array_audio[start_sample:end_sample]
#position_data, cfl_ids, tdedata = generate_candidate_sources_v2(audio_chunk, **kwargs)
%load_ext line_profiler
%lprun -f pll_create_tde_data generate_candidate_sources_v2(audio_chunk, **kwargs)

#%% Write the data out into a txt file
# import csv

# with open("out.csv", "w") as f:
#     wr = csv.writer(f)
#     wr.writerows(tdedata[-500000:])


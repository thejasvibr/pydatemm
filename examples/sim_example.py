'''
Localising overlapping calls: simulated audio case
==================================================

'''
import matplotlib.pyplot as plt
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

array_geom = pd.read_csv('mic_xyz_multibatsim.csv').to_numpy()[:,1:]
simdata = pd.read_csv('multibatsim_xyz_calling.csv').loc[:,'x':]
simdata_callpoints = simdata[simdata['emission_point']]


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

time_pts = np.arange(0, 0.095, 10e-3)

results = {}
for start_time in tqdm.tqdm(time_pts):    
    end_time = start_time  + max_delay
    print(f'Now handling audio between: {(start_time,end_time)}')
    start_sample, end_sample = int(fs*start_time), int(fs*end_time)  
    sim_audio = array_audio[start_sample:end_sample]
    output = generate_candidate_sources(sim_audio, **kwargs)
    results[start_time] = output.sources

#%%
def conv_to_numpy(pydatemm_out):
    return np.array([np.array(each) for each in pydatemm_out]).reshape(-1,4)
xmax, ymax, zmax = [4,9,3]
for key, entry in results.items():
    print(key, len(entry))
    if len(entry)>0:
        posns = conv_to_numpy(entry)
        # get rid of -999 entries
        no_999 = np.logical_and(posns[:,0]!=-999, posns[:,1]!=-999)
        geq_0 = np.logical_and(posns[:,0]>=0, posns[:,1]>=0)
        geq_0 = np.logical_and(geq_0, posns[:,2]>=0)
        leq_xymax = np.logical_and(posns[:,0]<=xmax, posns[:,1]<=ymax)
        leq_xyzmax = np.logical_and(leq_xymax, posns[:,2]<=zmax)
        no999_geq0 = np.logical_and(no_999, geq_0)
        no999geq0_leqxyz = np.logical_and(no999_geq0, leq_xyzmax)
        posns_filt = posns[no999geq0_leqxyz,:]
        posns_filt = posns_filt[posns_filt[:,-1]<1e-3]
        
#%%
plt.figure()
plt.subplot(311)            
plt.violinplot(posns_filt[:,0], vert=False)
plt.subplot(312)            
plt.violinplot(posns_filt[:,1], vert=False)
plt.subplot(313)            
plt.violinplot(posns_filt[:,2], vert=False)

plt.figure()
a0 = plt.subplot(111, projection='3d')
plt.plot(posns_filt[:,0], posns_filt[:,1], posns_filt[:,2],'*')
a0.set_xlim(0,4); a0.set_ylim(0,9); a0.set_zlim(0,3)
a0.set_xlabel('x'); a0.set_ylabel('y'); a0.set_zlabel('z')
    
#%% When can we expect to detect a call emitted at 0.002 s
source_to_mic = []
for each in array_geom:
    source_to_mic.append(euclidean(each, emission_point))

arrival_times = 0.007 + np.array(source_to_mic)/343.0

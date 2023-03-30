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


max_delay = np.max(distance_matrix(array_geom, array_geom))/kwargs['vsound']  
kwargs['K'] = 4
kwargs['num_cores'] = 2

#%%
# i = 110 -- tricky one , 120 even worse
# start_time = 0.030

time_pts = np.arange(0, 0.21, 10e-3)

results = {}
for start_time in tqdm.tqdm(time_pts):    
    end_time = start_time  + max_delay
    print(f'Now handling audio between: {(start_time,end_time)}')
    start_sample, end_sample = int(fs*start_time), int(fs*end_time)  
    try:
        sim_audio = array_audio[start_sample:end_sample]
        output = generate_candidate_sources(sim_audio, **kwargs)
        results[start_time] = output.sources
    except:
        pass

#%%
def conv_to_numpy(pydatemm_out):
    return np.array([np.array(each) for each in pydatemm_out]).reshape(-1,4)

tdoa_resid_threshold = 0.1e-3
filtered_results = {}

for key, entry in results.items():
    print(key, len(entry))
    if len(entry)>0:
        posns = conv_to_numpy(entry)
        # get rid of -999 entries
        no_999 = np.logical_and(posns[:,0]!=-999, posns[:,1]!=-999)
        posns_filt = posns[no_999,:]
        posns_filt = posns_filt[posns_filt[:,-1]<tdoa_resid_threshold]
        filtered_results[key] = posns_filt
    else:
        filtered_results[key] = np.array([])
        
#%% 
# Account for time-of-flight
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# The sound received at the microphones is a signal from the 'past'. The mock video
# trajectory data is 'real-time' data. Choose one of the mics are the reference, and 
# calculate an estimated time-of-flight, to get the ~ time of emission. 
import scipy.spatial as spatial

max_tof = 15/kwargs['vsound']

for t_focal, tfocal_sources in filtered_results.items():
    print(t_focal)
    if len(tfocal_sources)>0:
        potential_sources = tfocal_sources[:,:3]
        
        tof = spatial.distance_matrix(potential_sources, array_geom[0,:].reshape(-1,3))/kwargs['vsound']
        
        
        tof = tof[tof<=max_tof]
        potential_toe = t_focal - tof
        potential_toe = potential_toe[np.logical_and(potential_toe>=0, potential_toe<=t_focal)]

        toe_range = np.percentile(potential_toe, [0,100])
        timelim_xyz = simdata[np.logical_and(simdata['t']>=toe_range[0], simdata['t']<=toe_range[1])].reset_index(drop=True)
        if timelim_xyz.shape[0]>0:
            traj_xyz = timelim_xyz.loc[:,'x':'z'].to_numpy()
            
            
            dist_mat_tfocal = spatial.distance_matrix(potential_sources, traj_xyz)
            # set a video-audio mismatch threshold
            mismatch_threshold = 0.3
            good_matches = np.argwhere(dist_mat_tfocal<=mismatch_threshold)
            
            best_traj_inds = np.unique(good_matches[:,1])
            best_traj_posns = traj_xyz[best_traj_inds,:]
            
            print('Best positions for call: ',best_traj_posns)
            #print(timelim_xyz.loc[best_traj_inds,:'batnum'])
            
        else:
            pass
    else:
        pass
        
    
    
    



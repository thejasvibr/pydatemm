# -*- coding: utf-8 -*-
"""
Localising a multi-bat simulation
=================================
Here we'll run a multi-bat simulation run in a shoe-box room. 3 bats emitting
3 calls at around 100 ms IPI. 


Created on Tue Sep 13 22:04:44 2022

@author: theja
"""

import soundfile as sf
from ky2013_fullsim_chain import *

#%%
filename = '3-bats_trajectory_simulation_raytracing-2.wav'
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
kwargs['max_loop_residual'] = 0.25e-4
kwargs['K'] = 4
dd = np.max(distance_matrix(array_geom, array_geom))/343  
dd_samples = int(kwargs['fs']*dd)

start_samples = np.arange(1920,array_audio.shape[0], 96)
end_samples = start_samples+dd_samples
max_inds = 90
start = time.perf_counter_ns()
all_candidates = []

for (st, en) in tqdm.tqdm(zip(start_samples[:max_inds], end_samples[:max_inds])):
    audio_chunk = array_audio[st:en,:]
    candidates = generate_candidate_sources(audio_chunk, **kwargs)
    #refined = refine_candidates_to_room_dims(candidates, 0.5e-4, room_dim)
    all_candidates.append(candidates)

for frame, df in enumerate(all_candidates):
    df['frame'] = frame


stop = time.perf_counter_ns()
durn_s = (stop - start)/1e9
print(f'Time for {max_inds} ms of audio analysis: {durn_s} s')
#%%
import trackpy as tp
all_frames = pd.concat(all_candidates).reset_index(drop=True)
#coarse_good_positions = all_frames[abs(all_frames['x'])<20]
valid_rows = np.tile(True, all_frames.shape[0])
for axis in ['x','y','z']:
    satisfying_rows = np.logical_and(all_frames[axis]<=4,
                                             all_frames[axis]>0)
    valid_rows *= satisfying_rows
coarse_positions = all_frames[valid_rows]
# also filter by good tdoa residual
tdoa_filtered = coarse_positions[coarse_positions['tdoa_resid_s']<1e-3].reset_index(drop=True)

#%% 
# Run DBSCAN to reduce the number of tracked points per frame!
filtered_by_frame = tdoa_filtered.groupby('frame')

all_dbscanned = []
for framenum, subdf in filtered_by_frame:
    dbscanned, std = dbscan_cluster(subdf, dbscan_eps=0.2, n_points=1)
    dbscanned_points = pd.DataFrame(data=[], index=range(len(dbscanned)),
                                                         columns=['x','y','z','frame'])
    dbscanned_points.loc[:,'x':'z'] = np.array(dbscanned).reshape(-1,3)
    dbscanned_points['frame'] = framenum
    all_dbscanned.append(dbscanned_points)

all_dbscanned = pd.concat(all_dbscanned).reset_index(drop=True)
linked = tp.link_df(all_dbscanned, search_range=0.2, pos_columns=['x','y','z'], memory=2)
# Keep only those particles that have been seen in at least 3 frames:
persistent_particles = []
for p_id, subdf in linked.groupby('particle'):
    if subdf.shape[0] >=2:
        persistent_particles.append(subdf)
all_persistent = pd.concat(persistent_particles).reset_index(drop=True)
all_persistent['t_sec'] = all_persistent['frame']*0.5e-3
avged_positions = []
for particle, subdf in all_persistent.groupby('particle'):
    xyz = subdf.loc[:,'x':'z'].to_numpy(dtype=np.float64)
    xyz_avg = np.mean(xyz, 0)
    avged_positions.append(xyz_avg)
#%%
plt.figure()
a0 = plt.subplot(111, projection='3d')
for framenum, subdf in all_persistent.groupby('frame'):
    a0.clear()
    a0.plot(array_geom[:,0],array_geom[:,1], array_geom[:,2],'^')
    #plt.plot(subdf['x'], subdf['y'], subdf['z'],'*')
    for particle_id, subsubdf in subdf.groupby('particle'):
        x,y,z = subsubdf.loc[:,'x':'z'].to_numpy(dtype=np.float64).flatten()
        a0.text(x, y, z, str(particle_id))
    a0.set_xlim(0, 4)
    a0.set_ylim(0, 9)
    a0.set_zlim(0, 3)
    a0.view_init(18,-56)
    plt.tight_layout()
    plt.title(f'frame: {framenum}', y=0.85)
    #plt.savefig(f'sources_by_frames_{framenum}.png')
    plt.pause(0.2)


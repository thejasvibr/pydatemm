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
kwargs['K'] = 10
dd = np.max(distance_matrix(array_geom, array_geom))/343  
dd_samples = int(kwargs['fs']*dd)

ignorable_start = int(0.01*fs)
shift_samples = 96
start_samples = np.arange(ignorable_start,array_audio.shape[0], shift_samples)
end_samples = start_samples+dd_samples
max_inds = int(0.2*fs/shift_samples)
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

room_dim = [4, 9, 3]
for ax_lim, axis in zip(room_dim, ['x','y','z']):
    satisfying_rows = np.logical_and(all_frames[axis]<=ax_lim,
                                             all_frames[axis]>0)
    valid_rows *= satisfying_rows
coarse_positions = all_frames[valid_rows]
# also filter by good tdoa residual
tdoa_filtered = coarse_positions[coarse_positions['tdoa_resid_s']<1.5e-3].reset_index(drop=True)

#%% 
# Run DBSCAN to reduce the number of tracked points per frame!
filtered_by_frame = tdoa_filtered.groupby('frame')

all_dbscanned = []
for framenum, subdf in tqdm.tqdm(filtered_by_frame):
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
    if subdf.shape[0] >=int(fs*0.004/shift_samples):
        persistent_particles.append(subdf)
all_persistent = pd.concat(persistent_particles).reset_index(drop=True)
all_persistent['t_sec'] = ignorable_start/fs + all_persistent['frame']*0.5e-3
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
    time_s = np.around((framenum*shift_samples+ignorable_start)/fs, 4)
    plt.title(f'frame: {framenum}, time: {time_s}', y=0.85)
    #plt.savefig(f'sources_by_frames_{framenum}.png')
    plt.pause(0.2)

#%% Load the flight trajectory (in principle obtained from another sensor, e.g. thermal video)
from scipy.interpolate import interp1d 

bat_xyz = pd.read_csv('multibat_xyz_emissiontime.csv')
# extra and intrapolate the data a bit. 
interp_by_batnum = {}
for batnum, subdf in bat_xyz.groupby('batnum'):
    interp_by_batnum[batnum] = [interp1d(subdf['t'],subdf[axis]) for axis in ['x','y','z']]

def interpolate_by_batnum(t, batnum, interp_dict):
    x, y, z = [interp_dict[batnum][i](t) for i in range(3)]
    return np.array([x,y,z])
# interpolate at every 0.5 ms between start and end time of trajectory
interp_trajectories = []
for batnum, subdf in bat_xyz.groupby('batnum'):
    t = subdf['t'].tolist()
    t_extended = np.arange(t[0], t[-1], 0.5e-3)
    interp_xyz = np.array([interpolate_by_batnum(step, batnum, interp_by_batnum) for step in t_extended])
    # add some noise already here
    interp_xyz += np.random.normal(0,0.05,interp_xyz.size).reshape(interp_xyz.shape)
    interp_traj_batnum = pd.DataFrame(data=interp_xyz, columns=['x','y','z'])
    interp_traj_batnum['t'] = t_extended
    interp_traj_batnum['batnum'] = batnum
    interp_trajectories.append(interp_traj_batnum)
interp_trajectories = pd.concat(interp_trajectories).reset_index(drop=True)

#%%
# Now let's try to align the video interpolated trajectories with the unlabelled
# acoustic source coordinates. We can simplify the problem by assuming that the
# call reaches within Delta time of the emission time. This means, broadly calculating
# the expected bat-array distance and then filtering out all sources that were detected 
# after that time point. 

particle_to_traj = {}
for particle, subdf in all_persistent.groupby('particle'):
    emission_xyz = subdf.loc[:,['x','y','z']].to_numpy(dtype=np.float64)
    average_emission = np.median(emission_xyz, 0)
    t_detection = np.min(subdf['t_sec'])
    # find all potential trajectories that are within 30 cm of the source, if
    # and -30 ms of emission detection time. 
    valid_window = np.logical_and(interp_trajectories['t'] < t_detection,
                                  interp_trajectories['t'] >= t_detection - 0.03)
    close_in_time = interp_trajectories[valid_window].reset_index(drop=True)
    # find all points that are close-by in space - within ~0.3 m
    traj_xyz = close_in_time.loc[:,['x','y','z']].to_numpy(dtype=np.float64)
    dist_mat = distance_matrix(traj_xyz,
                               average_emission.reshape(1,3))
    space_close = close_in_time[dist_mat < 0.3]
    particle_to_traj[particle] = []
    for batnum, subdf in space_close.groupby('batnum'):
        # append the first trajectory point as the start of the call emission.
        particle_to_traj[particle].append(subdf.loc[np.min(subdf.index),:])
# keep only those particles with a trajectory associated to them. 
matched_particle_trajs = {}
for particle, trajs in particle_to_traj.items():
    if len(trajs)>0:
        matched_particle_trajs[particle] = trajs

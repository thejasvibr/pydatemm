"""
Trying out acoustic tracking of multi-bat audio recorded in Orlova Chuka
========================================================================

Source audio file timestamp: 1529434881 (pre-synchronised, see 'audio_prep.py')
12 microphones split over a Fireface UC and Fireface UCX.

Microphones laid out in the following order

Channels 0-5: SMP9, SMP10, SMP11, SMP12, SMP1, SMP2
Channel 6: empty
Channel 7: SYNC
Channel 8-13: SMP3, SMP4, SMP5, SMP6, SMP7, SMP8
Channel 14: empty
Channel 15: SYNC

Microphone positions 
~~~~~~~~~~~~~~~~~~~~
Microphone positions given in XYZ as measured by Asparuh Kamburov on 2018-06-20
morning with a Total Station. The Total Station measurements have an accuracy
of a few mm. 

Recording
~~~~~~~~~
The current audio recording has what seems to be multiple bats in it!
"""
import soundfile as sf
import time, tqdm
from ccg_localiser import *
from scipy.spatial import distance_matrix

array_geometry = pd.read_csv('../examples/array_geom_2018-06-19_onlymics.csv')
array_geom = array_geometry.loc[:,'x':'z'].to_numpy()
multich_audio, fs = sf.read('../examples/first_twoseconds_1529434881_2018-06-19_synced.wav')
# take out only the mic relevant channels 
array_audio = multich_audio[:,[0,1,2,3,4,5,8,9,10,11,12,13]]

#%%
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
kwargs['K'] = 7
dd = np.max(distance_matrix(array_geom, array_geom))/343  
dd *= 0.8
dd_samples = int(kwargs['fs']*dd)

start_samples = np.arange(0,array_audio.shape[0], 96)
end_samples = start_samples+dd_samples
max_inds = 40
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
coarse_positions = all_frames[np.logical_and(np.abs(all_frames['x'])<10,
                                             np.abs(all_frames['y'])<10)]
coarse_positions = coarse_positions[np.abs(coarse_positions['z'])<3]
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
    if subdf.shape[0] >=4:
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
    a0.set_xlim(-7, 1)
    a0.set_ylim(-1, 4)
    a0.set_zlim(-3, 3)
    a0.view_init(28,-26)
    plt.tight_layout()
    plt.title(f'frame: {framenum}', y=0.85)
    plt.savefig(f'sources_by_frames_{framenum}.png')
    plt.pause(0.2)
    
#%%
# plt.figure()
# a1 = plt.subplot(111, projection='3d')
# for i, data in enumerate(all_candidates):
#     x,y,z = [data.loc[:,ax]  for ax in ['x','y','z']]
#     a1.plot(x,y,z,'*')
#     a1.plot(array_geom[:,0],array_geom[:,1], array_geom[:,2],'^')
#     a1.set_xlim(-7, 1)
#     a1.set_ylim(-1, 4)
#     a1.set_zlim(-3, 3)
#     plt.savefig(f'miaow{i}.png')
#     a1.clear()


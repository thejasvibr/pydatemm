#!/usr/bin/env pyth

# -*- coding: utf-8 -*-
"""
Estimating call emission time and source from simulated data
============================================================

"""
import glob
import matplotlib
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import numpy as np 
import pandas as pd
import soundfile as sf
from scipy.spatial import distance_matrix, distance
import scipy.signal as signal 
euclidean = distance.euclidean
import scipy.interpolate as si
import tqdm
from source_traj_aligner import generate_proximity_profile

NBAT=8
output_data_pattern = '823'
output_folder = f'multibat_stresstests/nbat{NBAT}'
arraygeom_file = output_folder+'/mic_xyz_multibatsim.csv'
audiofile = output_folder+f'/{NBAT}-bats_trajectory_simulation_1-order-reflections.WAV'
array_geom = pd.read_csv(arraygeom_file).loc[:,'x':'z'].to_numpy()
vsound = 340.0 # m/s
#%%
# load all the results into a dictionary
result_files = natsorted(glob.glob(output_folder+f'/nbats{NBAT}outdata/*{output_data_pattern}*.csv'))
# keep only those with the relevant time-window size
def get_start_stop_times(file_name):
    times = file_name.split('_')[-1].split('.csv')[0]
    start_t, stop_t = [float(each) for each in times.split('-')]
    durn = stop_t - start_t
    return durn
def window_length_is_correct(file_name, expected, tolerance=1e-15):
    durn = get_start_stop_times(file_name)
    if abs(durn-expected)<tolerance:
        return True
    else:
        return False
    
all_results = []
for i,fname in enumerate(result_files):
    if window_length_is_correct(fname, 0.010):
        all_results.append(pd.read_csv(fname))

all_sources = pd.concat(all_results).reset_index(drop=True)
all_posns = all_sources.loc[:,['x','y','z','tdoa_res','t_start','t_end']].to_numpy()

#%%
# Now load the video flight trajectories and transform them from the 
# camera to TotalStation coordinate system
start_time = 0
end_time = sf.info(audiofile).duration
flight_traj = pd.read_csv(output_folder+'/multibatsim_xyz_calling.csv')
flight_traj = flight_traj.rename(columns={'batnum':'batid'})
#%%
cmap = matplotlib.cm.get_cmap('viridis')
fractions = np.linspace(0,1,np.unique(flight_traj['batid']).size)
colors = [cmap(frac)[:-1] for frac in fractions]
#%% Now upsample the trajectories
upsampled_flighttraj = []
for batid, batdf in flight_traj.groupby('batid'):
    upsampled_df = pd.DataFrame(data=[], columns=['t','batid','x','y','z'])    
    t_raw = batdf['t'].to_numpy()
    t_highres = np.arange(t_raw.min(), t_raw.max(), 1e-3)
    upsampled_df['t'] = t_highres
    upsampled_df['batid'] = batid
    traj_splines = {i : si.interp1d(t_raw, batdf.loc[:,i], 'quadratic') for i in ['x','y','z']}
    traj_interp = { i: traj_splines[i](t_highres) for i in ['x','y','z']}

    for axis in ['x','y','z']:
        upsampled_df[axis] = traj_interp[axis]
    upsampled_flighttraj.append(upsampled_df)
upsampled_flighttraj = pd.concat(upsampled_flighttraj).reset_index(drop=True)

#%% Keep only those that are within a few meters of any bat trajectory positions
distmat = distance_matrix(upsampled_flighttraj.loc[:,'x':'z'].to_numpy(), all_posns[:,:3])
nearish_posns = np.where(distmat<0.75) # all points that are at most 1 metres from any mic
sources_nearish = all_posns[np.unique(nearish_posns[1]),:]

mic_video_xyz = pd.read_csv(arraygeom_file)


#%%
coarse_threshold = 0.5
fine_threshold = 0.3
topx = 10

counts_by_batid = {}
for batid, batdf in tqdm.tqdm(upsampled_flighttraj.groupby('batid')):
    prox_profile = generate_proximity_profile(batid, batdf, 
                                              sources_nearish, coarse_threshold,
                                              fine_threshold ,
                                              array_geom, vsound, topx)
    counts_by_batid[batid]= prox_profile

#%%
fs, duration = sf.info(audiofile).samplerate,  sf.info(audiofile).duration
audio, fs = sf.read(audiofile)
num_bats = len(counts_by_batid.keys())

#%%
from source_traj_aligner import plot_diagnostic

proximity_peaks = {}
for i,(batid, source_prof) in enumerate(counts_by_batid.items()):
    t_batid = upsampled_flighttraj.groupby('batid').get_group(batid)['t'].to_numpy()
    pks, _ = signal.find_peaks(source_prof, distance=15,  height=4)
    proximity_peaks[batid] = t_batid[pks]
proximity_data = (counts_by_batid, proximity_peaks,)

#%%
from scipy.signal import find_peaks_cwt

cwt_peaks = {}
for i,(batid, source_prof) in enumerate(counts_by_batid.items()):
    t_batid = upsampled_flighttraj.groupby('batid').get_group(batid)['t'].to_numpy()
    pks = find_peaks_cwt(source_prof, widths=np.arange(2,7),
                                      max_distances=np.tile(2,5))
    cwt_peaks[batid] = t_batid[pks ]

proximity_data1 = (counts_by_batid, cwt_peaks,)

#%%
fig, axs = plot_diagnostic((audio, fs, duration), proximity_data, array_geom, upsampled_flighttraj,
                 vis_channels=[0,-1], sim_traj=flight_traj)

fig1, axs1 = plot_diagnostic((audio, fs, duration), proximity_data1, array_geom, upsampled_flighttraj,
                 vis_channels=[0,1], sim_traj=flight_traj)
axs1[0].set_title('With CWT peak finding')
#%%
# Now let's analyse the results from DBSCAN
plt.figure()
a0 = plt.subplot(111, projection='3d')
by_batid = flight_traj.groupby('batid')
focal_batid = 3
# plot the array
plt.plot(array_geom[:,0],array_geom[:,1],array_geom[:,2],'k')
x,y,z = [by_batid.get_group(focal_batid).loc[:,ax] for ax in ['x','y','z']]
plt.plot(x,y,z, '*')
plt.plot(x[x.index[0]],y[y.index[0]],z[z.index[0]],'r^')
a0.set_xlim(0,4);a0.set_ylim(0,9);a0.set_zlim(0,3)
# plot all emission points 
subdf = by_batid.get_group(focal_batid)
call_points = subdf[subdf['emission_point']]
a0.plot(call_points.loc[:,'x'],call_points.loc[:,'y'],call_points.loc[:,'z'],'o')

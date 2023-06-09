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

NBAT=4
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
topx = 5

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
# def calculate_toa_across_mics(time_click, array_geom):
#     closest_traj_point = 
#     tof_mat = distance_matric(time_cli)
    

audio_channels = [0,1,]

fig, axs = plt.subplots(ncols=1, nrows=num_bats+len(audio_channels),
                        figsize=(5, 20.0),
                        layout="tight", sharex=True)
traj_data = upsampled_flighttraj.groupby('batid')

# here I'll also establish an interactive workflow to get the bat calls 

def get_clicked_point_location(event):
    if event.button is MouseButton.LEFT:
        event_axes = [ax.in_axes(event) for ax in axs ]
        if sum(event_axes) == 0:
            return None, None
        else:
            axes_id = int(np.where(event_axes)[0])
            
            tclick, _ =  event.xdata, event.ydata
            curr_plot = axs[axes_id]
            num_lines = len(curr_plot.lines)
            plt.sca(curr_plot)
            if num_lines >2:
                for i in range(2,num_lines)[::-1]:
                    curr_plot.lines[i].remove()
            
            plt.plot(event.xdata, event.ydata,'r*')
#            plt.vlines(event.xdata-0.008, 0, 50, colors=colors[axes_id])
            return tclick, int(axes_id)
    else:
        return None, None

def calculate_toa_channels(t_source, sourceid, arraygeom):
    flight_traj = traj_data.get_group(sourceid).reset_index(drop=True)
    # get closest point in time 
    nearest_ind = abs(flight_traj['t']-t_source).argmin()
    emission_point = flight_traj.loc[nearest_ind, 'x':'z'].to_numpy().reshape(-1,3)
    tof = distance_matrix(emission_point, arraygeom)/343.0 
    toa = t_source + tof
    return toa

def draw_expected_toa(event, target_channels, window_halfwidth):
    t_emission, ax_id = get_clicked_point_location(event)
    if t_emission is not None:
        batid = ax_id + 1 
        toa = calculate_toa_channels(t_emission, batid, 
                                     array_geom[target_channels,:]).flatten()
        toa_2 = calculate_toa_channels(t_emission-window_halfwidth, batid, 
                                     array_geom[target_channels,:]).flatten()


        # get last few axes - that have specgrams
        specgram_axes = axs[-len(target_channels):]
        for channel_toa,channel_toa2, axid in zip(toa, toa_2, specgram_axes):
            plt.sca(axid)
            plt.vlines(channel_toa, 20000,96000, linestyle='dotted', color=colors[ax_id])
            #plt.vlines(channel_toa2, 20000,96000, linestyle='dotted', color=colors[ax_id])
            
        fig.canvas.draw()
        #fig.canvas.draw_idle()

proximity_peaks = {}

for i,(batid, source_prof) in enumerate(counts_by_batid.items()):
    t_batid = upsampled_flighttraj.groupby('batid').get_group(batid)['t'].to_numpy()
    
    plt.sca(axs[i])
    plt.plot(t_batid, source_prof, label='bat id ' + str(batid), color=colors[int(batid)-1])
    plt.xticks([])
    plt.legend()
    
    pks, _ = signal.find_peaks(source_prof, distance=15,  height=4)
    proximity_peaks[batid] = t_batid[pks]
    plt.plot(t_batid[pks], source_prof[pks],'g*')


for i, ch in enumerate(audio_channels):
    plt.sca(axs[num_bats+i])
    plt.specgram(audio[:,ch], Fs=fs, xextent=[0, sf.info(audiofile).duration], cmap='cividis')


def plotted_toa_from_peaks(specgram_axes, batid, peak_times, target_channels, window_halfwidth):
    for t_emission in peak_times:
        toa = calculate_toa_channels(t_emission, batid, 
                                     array_geom[target_channels,:]).flatten()
        toa_2 = calculate_toa_channels(t_emission-window_halfwidth, batid, 
                                     array_geom[target_channels,:]).flatten()
    

        for channel_toa,channel_toa2, axid in zip(toa, toa_2, specgram_axes):
            plt.sca(axid)
            plt.vlines(channel_toa, 20000,96000, linestyle='dotted', color=colors[batid-1])
            #plt.vlines(channel_toa2, 20000,96000, linestyle='dashed', color=colors[batid-1])
    fig.canvas.draw()

nchannels = len(audio_channels)
for batid in list(counts_by_batid.keys()):
    plotted_toa_from_peaks(axs[-nchannels:], batid, proximity_peaks[batid], audio_channels, 16e-3)

plt.gca().set_xticks(np.arange(0, duration, .05))
plt.gca().set_xticks(np.linspace(0, duration, 100), minor=True)

from matplotlib.lines import Line2D
actual_emission_points = flight_traj[flight_traj['emission_point']].groupby('batid')
for batid, subdf in actual_emission_points:
    plt.sca(axs[batid-1])
    for i, row in subdf.iterrows():
        _,x,y,z,t,*uu = row
        plt.vlines(t, 0, np.max(counts_by_batid[batid]), color='r')

# access legend objects automatically created from data
plt.sca(axs[0])
handles, labels = axs[0].get_legend_handles_labels()
line = Line2D([0],[0],label='original emission times', color='r')
handles.extend([line])
plt.legend(handles=handles)

plt.savefig(f'{NBAT}_{output_data_pattern}_run.png')
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

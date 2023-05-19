#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparing effect of mic-geometry noise
======================================
Here I compare the effect of adding noise to the original microphone xyz coordinates,
such than there is a <= 5cm euclidean error to the original coordinates. 

@author: thejasvi
"""

import glob
import matplotlib
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np 
import pandas as pd
import pyvista as pv
import soundfile as sf
import vtk
from scipy.spatial import distance_matrix, distance
euclidean = distance.euclidean
import scipy.interpolate as si
from sklearn.cluster import DBSCAN

output_folder = '1529543496_output/'
arraygeom_file = '1529543496_input/Sanken9_centred_mic_totalstationxyz.csv'
audiofile = '1529543496_input/video_synced10channel_first15sec_1529543496.WAV'
array_geom = pd.read_csv(arraygeom_file).loc[:,'x':'z'].to_numpy()
vsound = 340.0 # m/s
#%%
# load all the results into a dictionary
result_orig_files = natsorted(glob.glob(output_folder+'/*4K*.csv'))
result_noisy_files = natsorted(glob.glob(output_folder+'/*noisy*.csv'))

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

orig_sources = pd.concat([pd.read_csv(each) for each in result_orig_files]).reset_index(drop=True)
noisy_sources = pd.concat([pd.read_csv(each) for each in result_noisy_files]).reset_index(drop=True)

orig_posns = orig_sources.loc[:,['x','y','z','tdoa_res','t_start','t_end']].to_numpy()
noisy_posns = noisy_sources.loc[:,['x','y','z','tdoa_res','t_start','t_end']].to_numpy() 
#%%
# Now load the video flight trajectories and transform them from the 
# camera to TotalStation coordinate system
start_time = 12.4
end_time = 13.4
flight_traj = pd.read_csv('1529543496_input/xyz_data/bat_trajs_round1_sanken9_centred.csv')
transform_matrix = pd.read_csv('1529543496_input/xyz_data/Sanken9_centred-video-to-TotalStation_transform.csv')
transform = transform_matrix.to_numpy()[:,1:]
bring_to_totalstation = lambda X: np.matmul(transform, X)
flight_traj_conv = flight_traj.loc[:,'x':'z'].to_numpy()
flight_traj_conv = np.apply_along_axis(bring_to_totalstation, 1, flight_traj_conv)
flight_traj_conv = pd.DataFrame(flight_traj_conv, columns=['x','y','z'])
flight_traj_conv['batid'] = flight_traj['batid'].copy()
flight_traj_conv['frame'] = flight_traj['frame'].copy()

# fill the gaps in traj 6:
traj6_data = flight_traj_conv.groupby('batid').get_group(6).dropna()
spline_fitxyz = [ si.interp1d(traj6_data.loc[:,'frame'], traj6_data.loc[:,ax],'quadratic') for ax in ['x','y','z']]
missing_frames = [15,16]


interp_xyz = []
for frame in missing_frames:
    interp_df = pd.DataFrame()
    x,y,z = np.array([ spline_fitxyz[i](frame) for i,ax in enumerate(['x','y','z'])]).flatten()
    maxind = flight_traj_conv.index.max()
    flight_traj_conv.loc[maxind+1] = [x,y,z,6,frame]

flight_traj_conv['time'] = flight_traj_conv['frame']/25 + 310/25
# filter out the relevant flight trajectory for the window
timewindow_rows = np.logical_and(flight_traj_conv['time'] >= start_time, 
                                        flight_traj_conv['time'] <= end_time) 
flighttraj_conv_window = flight_traj_conv.loc[timewindow_rows,:].dropna()

# Also upsample the flight trajectory data to millisecond resolution 

highres_xyz = []
for batid, subdf in flighttraj_conv_window.groupby('batid'):
    highres_subdf = pd.DataFrame()
    t = subdf['time'].to_numpy()
    t_interp = np.arange(t.min(), t.max(), 1e-3)    
    highres_spline = {ax: si.interp1d(t, subdf.loc[:,ax],'linear') for ax in ['x','y','z']}
    highres_subdf['t'] = t_interp
    for ax in ['x','y','z']:
        highres_subdf[ax] = highres_spline[ax](t_interp)
    highres_subdf['batid'] = batid
    highres_xyz.append(highres_subdf)
highres_flighttraj = pd.concat(highres_xyz).reset_index(drop=True)

#%% Filtering out nearby sources
#   ============================
# Keep only those that are within a few meters of any bat trajectory positions - and those that make sense
# time-wise 

orig_distmat = distance_matrix(highres_flighttraj.loc[:,'x':'z'].to_numpy(), orig_posns[:,:3])
orig_nearish_posns = np.where(orig_distmat<0.3) # 
orig_posns_nearish = orig_posns[np.unique(orig_nearish_posns[1]),:]

#%%
# Run a DBSCAN on the nearish sources to get call centres

clusters = DBSCAN(eps=0.05).fit(orig_posns_nearish[:,:3])

# function to calculate a weighted average based on the TDOA residual
def residual_weighted_average(xyz_tdoares):
    '''
    Assumes the output to be a Npointsx4 np.array
    with x,y,z,tdoares format
    '''
    weighted_centre = np.zeros(3)
    for i in range(3):
        weighted_centre[i] = np.average(xyz_tdoares[:,i], weights=xyz_tdoares[:,-1])
    return weighted_centre
        

cluster_centres = []
for lab in np.unique(clusters.labels_):
    if not lab == -1:
        inds = np.where(clusters.labels_==lab)
        cluster_points = orig_posns_nearish[inds, :].reshape(-1,6)
        centroid = np.median(cluster_points[:,:3],axis=0)
        #centroid = residual_weighted_average(cluster_points[:,:4])
        # get the min-max time-window for all of the cluster_points
        time_window = np.percentile(cluster_points[:,-2:].flatten(), [0,100], axis=0)
        centroid = np.append(centroid, time_window)
        print(f'Label: {lab}, {time_window}')
        cluster_centres.append(centroid)
cluster_centres = np.array(cluster_centres)

#%%
# filter out cluster centres based on how far they are from a trajectory and how close
# they can be time-wise
max_time_diff = 0.05
max_dist_from_traj = 0.25

valid_centres = []
potential_emission_points = []
for cluster in cluster_centres:
    x, y, z, tmin, tmax = cluster
    valid_trajs = np.logical_and(highres_flighttraj['t']>=tmin,
                                highres_flighttraj.loc[:,'t']<=tmax)
    subset_trajs = highres_flighttraj.loc[valid_trajs,:]
    # now check that the proposed source is close enough to a trajectory
    distmat = distance_matrix(subset_trajs.loc[:,'x':'z'].to_numpy(), np.array([x,y,z]).reshape(1,3))
    
    if sum(distmat<=max_dist_from_traj) >0:
        idxmin = np.argmin(distmat)
        nearest_trajpoint = subset_trajs.loc[subset_trajs.index[idxmin],:]
        cluster_and_trajpoint = np.append(nearest_trajpoint, cluster)
        valid_centres.append(cluster)
        potential_emission_points.append(cluster_and_trajpoint)

emission_points = pd.DataFrame(potential_emission_points)
emission_points.columns = ['t','x','y','z','batid','ac_x','ac_y','ac_z','tmin','tmax']
emission_points = emission_points.loc[:,:].sort_values(['batid','t'])

#%%
fs = sf.info(audiofile).samplerate
audio_clip, _ = sf.read(audiofile, start=int(12.4*fs), stop=int(13.4*fs))

fig, axes = plt.subplots(nrows=10, ncols=1,sharex=True, sharey=True, figsize=(10,20))
for i in range(10):
    plt.sca(axes[i])
    plt.specgram(audio_clip[:,i], Fs=fs, NFFT=256, noverlap=128, xextent=[12.4,13.4])
    axes[i].set_ylim(20000,96000)
    if i!=9:
        plt.xticks([])
    else:
        axes[i].set_xticks(np.arange(12.4,13.4,5e-3), minor=True)
        axes[i].set_xticks(np.arange(12.4,13.4,0.1), minor=False)

for batid, subdf in emission_points.groupby('batid'):
    print(batid)
    subdf = subdf.reset_index(drop=True)
    for i, callrow in subdf.iterrows():
        mic_callpoint_dist = distance_matrix(callrow['ac_x':'ac_z'].to_numpy().reshape(-1,3),
                                             array_geom)
        tof = mic_callpoint_dist/vsound
        toa = (callrow['t']+tof).flatten()
        for j in range(10):
            plt.sca(axes[j])
            plt.vlines(toa[j], 20000,96000)
            plt.text(toa[j], 50000+np.random.choice(np.arange(1000,3000),1), str(int(batid))+'-'+str(i))
#%%

#%%
fs = sf.info(audiofile).samplerate
audio_clip, _ = sf.read(audiofile, start=int(12.4*fs), stop=int(13.4*fs))

fig, axes = plt.subplots(nrows=10, ncols=1,sharex=True, sharey=True, figsize=(10,20))
for i in range(10):
    plt.sca(axes[i])
    plt.specgram(audio_clip[:,i], Fs=fs, NFFT=256, noverlap=128, xextent=[12.4,13.4])
    axes[i].set_ylim(20000,96000)
    if i!=9:
        plt.xticks([])
    else:
        axes[i].set_xticks(np.arange(12.4,13.4,5e-3), minor=True)
        axes[i].set_xticks(np.arange(12.4,13.4,0.1), minor=False)

#%%
# A zoomed in version

fs = sf.info(audiofile).samplerate
tstart, tstop = 13.0, 13.23
audio_clip, _ = sf.read(audiofile, start=int(tstart*fs), stop=int(tstop*fs))

fig, axes = plt.subplots(nrows=10, ncols=1,sharex=True, sharey=True, figsize=(10,20))
for i in range(10):
    plt.sca(axes[i])
    plt.specgram(audio_clip[:,i], Fs=fs, NFFT=192, noverlap=128, xextent=[tstart,tstop],)
    axes[i].set_ylim(20000,96000)
    if i!=9:
        plt.xticks([])
    else:
        axes[i].set_xticks(np.arange(tstart,tstop,5e-3), minor=True)
        axes[i].set_xticks(np.arange(tstart,tstop,0.1), minor=False)
plt.xlim(tstart, tstop)
for batid, subdf in emission_points.groupby('batid'):
    print(batid)
    valid_rows = np.logical_and(subdf['t']>=tstart, subdf['t']<=tstop)
    subdf = subdf.loc[valid_rows,:].reset_index(drop=True)
    for i, callrow in subdf.iterrows():
        mic_callpoint_dist = distance_matrix(callrow['ac_x':'ac_z'].to_numpy().reshape(-1,3),
                                             array_geom)
        tof = mic_callpoint_dist/vsound
        toa = (callrow['t']+tof).flatten()
        for j in range(10):
            plt.sca(axes[j])
            plt.vlines(toa[j], 20000,96000)
            plt.text(toa[j]+np.random.choice(np.linspace(0,2e-4,20), 1),
                     20000+np.random.choice(np.arange(10000,50000,4e3),1), str(int(batid))+'-'+str(i))
#%%
#
plt.figure()
ax = plt.subplot(111)
ax.scatter(emission_points['t'],emission_points['batid'],
           s=np.tile(40, emission_points.shape[0]), label='Call emission')
ax.set_yticks(np.arange(1,7))
ax.set_xticks(np.arange(12.4,13.4,0.1),  fontsize=18, minor=False)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_xticks(np.arange(12.4,13.4,0.05), minor=True)
plt.legend(loc='lower right', fontsize=15)
#ax.set_xticklabels(np.arange(12.4,13.4,0.05))
ax.grid(True, which='minor', axis='x', linestyle='--')
ax.grid(True, which='major', axis='y', linestyle='--')
plt.ylabel('Bat trajectory #', fontsize=22)
plt.xlabel('Audio time-stamp (s)', fontsize=22)
plt.tight_layout()
plt.savefig('call_emission_rough.png')


#%%
import matplotlib.pyplot as plt
all_ipis = []
for batid, subdf in emission_points.groupby('batid'):
    ipi = np.diff(subdf['t'])*1e3
    print(batid, ipi)
    all_ipis.append(ipi)
    print(subdf['t'].to_numpy())

plt.figure()
plt.hist(np.concatenate(all_ipis), bins=np.arange(0,500,10))

#%%
mic_video_xyz = pd.read_csv('1529543496_input/xyz_data/Sanken9_centred_mic_videoxyz.csv')
mic_video_xyz.columns = [0,'x','y','z','micname']

#%%
# Camera trajectories
camera_positions = np.array([
                    [-9.34, -4.0 , 2.11],
                    [-1.0, -8.18, 2.08],
                    [-5.27, -7.03, 2.24],
                    [-9.12, -3.12, 2.03],
                    ], dtype=np.float64)
t = np.linspace(0,1,camera_positions.shape[0])
t_interp = np.linspace(0,1,30)
az_values = []
elev_values = []
roll_values = [-31, 50, 76, 84]
roll_spline = si.interp1d(t,roll_values)
xyz_splines = [ si.interp1d(t, camera_positions[:,i], 'cubic') for i in range(3)]
xyz_interp = []
for i in range(3):
    xyz_interp.append(xyz_splines[i](t_interp))
roll_interp = roll_spline(t_interp)
camera_traj = np.column_stack(xyz_interp)

#%%

box = pv.Box(bounds=(0,5,0,9,0,3))
plotter = pv.Plotter()


def my_cpos_callback(*args):
    plotter.add_text(str(plotter.camera_position), name="cpos")
    print(plotter.camera_position)
    return

plotter.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, my_cpos_callback)

plotter.camera.position = (-2.9018132687083975, -5.786645457239613, 2.9734278298254244)
orientation = np.degrees([0.0221664793458192, -0.013873840146214714, 0.9996580234025079])
plotter.camera.azimuth, plotter.camera.elevation, plotter.camera.roll = orientation
plotter.camera.focal_point = (-1.12, 0.52, 0.46)
plotter.camera.view_angle = 30.

# campos2_params = {}
# campos2_params['position'] = (-2.95, -6.08, 1.73)
# campos2_params['focal_point'] = (-1.12, 0.52, 0.46)
# campos2_params['azimuth'] = np.degrees(0.197)
# campos2_params['elevation'] = np.degrees(0.31)
# campos2_params['roll'] =  np.degrees(0.98)

# for key, value in campos2_params.items():
#     setattr(plotter.camera, key, value)

# def mouse_callback(x):
#     print(f'camera position: {plotter.camera.position})')
#     print(f'\n az, roll, el: {plotter.camera.azimuth, plotter.camera.roll, plotter.camera.elevation}')
#     print(f'\n view angle, focal point: {plotter.camera.view_angle, plotter.camera.focal_point}')
#plotter.track_click_position(mouse_callback)

# include the mic array
for each in array_geom:
    plotter.add_mesh(pv.Sphere(0.03, center=each), color='g')

for i in [1,2,3]:
    plotter.add_mesh(pv.lines_from_points(array_geom[[0,i],:]), line_width=5)
plotter.add_mesh(pv.lines_from_points(array_geom[4:,:]), line_width=5)



cmap = matplotlib.cm.get_cmap('Spectral')
fractions = np.linspace(0,1,np.unique(flight_traj_conv['batid']).size)
colors = [cmap(frac)[:-1] for frac in fractions]


plotter.save_graphic('array_and_vid_trajs.pdf')

# plotter.add_points(orig_posns[:,:3], style='points_gaussian', point_size=1)
# plotter.save_graphic('sources_withtraj.pdf')

a1 = plotter.add_points(orig_posns_nearish[:,:3], style='points_gaussian', point_size=1,)
#plotter.save_graphic('sources_withtraj.pdf')

raw_clusts = []
for each in cluster_centres:
    raw_clusts.append(plotter.add_mesh(pv.Sphere(0.1, center=each[:3]), color='w', opacity=0.85))


plotter.save_graphic('raw_clusters_wtraj.pdf')

for each in raw_clusts:
    plotter.remove_actor(each)


# plot the flight trajectories and call emission points
for key, subdf in highres_flighttraj.groupby('batid'):
    traj_line = pv.lines_from_points(subdf.loc[:,'x':'z'].to_numpy())
    plotter.add_mesh(traj_line, line_width=7, color=colors[int(key)-1])

for potential_source in valid_centres:
    plotter.add_mesh(pv.Sphere(0.1, center=potential_source[:3]), color='w', opacity=0.85)


plotter.remove_actor(a1)
#plotter.add_points(orig_posns_nearish[:,:3], style='points_gaussian', point_size=1,)
plotter.save_graphic('clustered_sources.pdf')

# plotter.open_gif(f'1529543496_{int(max_dist_from_traj*100)}cmmaxdist-origmic.gif', fps=5)

# for roll,cam_posn in zip(roll_interp, camera_traj):
#     plotter.camera.position = tuple(cam_posn)
#     plotter.camera.roll = roll
#     plotter.write_frame()
# plotter.show(auto_close=True)  
# plotter.close()    

plotter.show()
#%%
# For each cluster centre - let's just say they are all correct. 
# If a call had been emitted from that point - using the expected TOFs we should
# recover bat calls at the calculated TOAs... 
batxyz = flighttraj_conv_window.loc[:,'x':'z'].dropna().to_numpy()
emission_data = pd.DataFrame(data=[], columns=['batid', 't_emission','toa_min','toa_max'])
index = 0 
timewindows = []
for each in cluster_centres:
  
    xyz  = each.reshape(1,3)
    tof = distance_matrix(xyz, array_geom)/340.0
    # find closest video point
    distmat = distance_matrix(xyz, batxyz)
    if np.min(distmat)<0.3:
        closest_point = batxyz[np.argmin(distmat)]
        # get video timestamp of the closest_point
        ind_closest_point = np.where(flighttraj_conv_window.loc[:,'x':'z']==closest_point)
        row = flighttraj_conv_window.index[ind_closest_point[0][0]]
        candidate_emission_time = flighttraj_conv_window.loc[row,'time']    
        batid = flighttraj_conv_window.loc[row,'batid']
        potential_toas = candidate_emission_time + tof
        toa_minmax = np.percentile(np.around(potential_toas,3), [0,100])
        emission_data.loc[index] = [batid, candidate_emission_time, toa_minmax[0], toa_minmax[1]]
        index += 1
        timewindows.append(toa_minmax)
        print(toa_minmax, np.around(distmat.min(),3), flighttraj_conv_window.loc[row, 'batid'])  
#%%
# import matplotlib.pyplot as plt
# import soundfile as sf
# fs = sf.info(audiofile).samplerate

# for j, row in emission_data.iterrows():
#     fig, axs = plt.subplots(10, 1, sharex='col', sharey='row', figsize=(2,6))
#     batid, t_emit, toa_min, toa_max = row
#     snippet, _ = sf.read(audiofile, start=int(toa_min*fs), stop=int(toa_max*fs))   
#     for i in range(10):
#         axs[i].specgram(snippet[:,i], Fs=fs, NFFT=192, noverlap=190)
#         axs[i].set_yticks([])
#         axs[i].set_xticks([0, toa_max-toa_min],[toa_min, toa_max])
#     axs[0].set_title(f'batid_{batid}_toa-min_{toa_min}')
#     plt.savefig(f'batid_{batid}_toa-min_{toa_min}_index{j}.png')
#     plt.close()     

#%%
# Now update the camera position 

box = pv.Box(bounds=(0,5,0,9,0,3))
plotter = pv.Plotter()

plotter.camera.position = tuple(camera_traj[0,:])
plotter.camera.azimuth = 0.0
plotter.camera.roll = roll_values[0]
plotter.camera.elevation = 0.0
plotter.camera.view_angle = 30.0
plotter.camera.focal_point = (-0.8600637284803319, 1.0764831143834046, 0.5677780189761863)
# include the mic array
for each in array_geom:
    plotter.add_mesh(pv.Sphere(0.03, center=each), color='g')

for i in [1,2,3]:
    plotter.add_mesh(pv.lines_from_points(array_geom[[0,i],:]), line_width=5)
plotter.add_mesh(pv.lines_from_points(array_geom[4:,:]), line_width=5)

# filter out cluster centres based on how far they are from a trajectory

for row,_ in zip(*valid_centres):
    plotter.add_mesh(pv.Sphere(0.1, center=cluster_centres[row,:]), color='w', opacity=0.9)

cmap = matplotlib.cm.get_cmap('Spectral')
fractions = np.linspace(0,1,np.unique(flight_traj_conv['batid']).size)
colors = [cmap(frac)[:-1] for frac in fractions]

# plot the flight trajectories and call emission points
for key, subdf in flighttraj_conv_window.groupby('batid'):
    traj_line = pv.lines_from_points(subdf.loc[:,'x':'z'].to_numpy())
    plotter.add_mesh(traj_line, line_width=7, color=colors[int(key)-1])

plotter.open_gif('1529543496-12.4-13.4.gif', fps=5, )

for roll,cam_posn in zip(roll_interp, camera_traj):
    plotter.camera.position = tuple(cam_posn)
    plotter.camera.roll = roll
    plotter.write_frame()
plotter.show(auto_close=True)  
plotter.close()    
    

#%%

viderr = 0.1 # +/- CI for video tracking in xyz
ranchoice = lambda : np.random.choice(np.arange(0,viderr,0.005), 3)
euc_dist = [np.linalg.norm(ranchoice()) for i in range(10000)]
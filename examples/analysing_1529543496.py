#!/usr/bin/env pyth

# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:33:20 2023

@author: thejasvi
"""
import glob
import matplotlib
from natsort import natsorted
import numpy as np 
import pandas as pd
import pyvista as pv
from scipy.spatial import distance_matrix, distance
euclidean = distance.euclidean
import scipy.interpolate as si

output_folder = '1529543496_output/'
arraygeom_file = '1529543496_input/Sanken9_centred_mic_totalstationxyz.csv'
audiofile = '1529543496_input/video_synced10channel_first15sec_1529543496.WAV'
array_geom = pd.read_csv(arraygeom_file).loc[:,'x':'z'].to_numpy()
#%%
# load all the results into a dictionary
result_files = natsorted(glob.glob(output_folder+'/*.csv'))
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
    if window_length_is_correct(fname, 0.016):
        all_results.append(pd.read_csv(fname))

all_sources = pd.concat(all_results).reset_index(drop=True)
all_posns = all_sources.loc[:,['x','y','z','tdoa_res']].to_numpy()

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

#%% Keep only those that are within a few meters of any bat trajectory positions
distmat = distance_matrix(flighttraj_conv_window.loc[:,'x':'z'].to_numpy(), all_posns[:,:-1])
nearish_posns = np.where(distmat<0.30) # all points that are at most 8 metres from any mic
sources_nearish = all_posns[np.unique(nearish_posns[1]),:]

# find the start and end times of the positions that are nearby. 
sources_w_time  = all_sources.loc[np.unique(nearish_posns[1]),['x','y','z','t_start','t_end']].reset_index(drop=True)

#%%
# Run a DBSCAN on the nearish sources to get call centres
from sklearn.cluster import DBSCAN
clusters = DBSCAN(eps=0.1).fit(sources_nearish[:,:-1])

cluster_centres = []
for lab in np.unique(clusters.labels_):
    if not lab == -1:
        inds = np.where(clusters.labels_==lab)
        cluster_points = sources_nearish[inds, :-1].reshape(-1,3)
        centroid = np.median(cluster_points,axis=0)
        #print(lab, centroid)
        cluster_centres.append(centroid)
cluster_centres = np.array(cluster_centres)

mic_video_xyz = pd.read_csv('1529543496_input/xyz_data/Sanken9_centred_mic_videoxyz.csv')
mic_video_xyz.columns = [0,'x','y','z','micname']

#%%
box = pv.Box(bounds=(0,5,0,9,0,3))
plotter = pv.Plotter()

def mouse_callback(x):
    print(f'camera position: {plotter.camera.position})')
    print(f'\n az, roll, el: {plotter.camera.azimuth, plotter.camera.roll, plotter.camera.elevation}')
    print(f'\n view angle, focal point: {plotter.camera.view_angle, plotter.camera.focal_point}')
plotter.track_click_position(mouse_callback)

#plotter.add_mesh(box, opacity=0.3)
# include the mic array
for each in array_geom:
    plotter.add_mesh(pv.Sphere(0.03, center=each), color='g')

for i in [1,2,3]:
    plotter.add_mesh(pv.lines_from_points(array_geom[[0,i],:]), line_width=5)
plotter.add_mesh(pv.lines_from_points(array_geom[4:,:]), line_width=5)

# filter out cluster centres based on how far they are from a trajectory
cluster_centre_traj = distance_matrix(cluster_centres, flighttraj_conv_window.loc[:,'x':'z'].to_numpy())
valid_centres = np.where(cluster_centre_traj<0.15)

for row,_ in zip(*valid_centres):
    plotter.add_mesh(pv.Sphere(0.1, center=cluster_centres[row,:]), color='w', opacity=0.9)

plotter.camera.position = (-2.29, -11.86, 1.50)
plotter.camera.azimuth = 0.0
plotter.camera.roll = 50
plotter.camera.elevation = 0.0
plotter.camera.view_angle = 30.0
plotter.camera.focal_point = (0.42, 1.59, 0.68)


cmap = matplotlib.cm.get_cmap('Spectral')
fractions = np.linspace(0,1,np.unique(flight_traj_conv['batid']).size)
colors = [cmap(frac)[:-1] for frac in fractions]

# plot the flight trajectories and call emission points
for key, subdf in flighttraj_conv_window.groupby('batid'):
    traj_line = pv.lines_from_points(subdf.loc[:,'x':'z'].to_numpy())
    plotter.add_mesh(traj_line, line_width=7, color=colors[int(key)-1])

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
import matplotlib.pyplot as plt
import soundfile as sf
fs = sf.info(audiofile).samplerate

for j, row in emission_data.iterrows():
    fig, axs = plt.subplots(10, 1, sharex='col', sharey='row', figsize=(2,6))
    batid, t_emit, toa_min, toa_max = row
    snippet, _ = sf.read(audiofile, start=int(toa_min*fs), stop=int(toa_max*fs))   
    for i in range(10):
        axs[i].specgram(snippet[:,i], Fs=fs, NFFT=192, noverlap=190)
        axs[i].set_yticks([])
        axs[i].set_xticks([0, toa_max-toa_min],[toa_min, toa_max])
    axs[0].set_title(f'batid_{batid}_toa-min_{toa_min}')
    plt.savefig(f'batid_{batid}_toa-min_{toa_min}_index{j}.png')
    plt.close()
        
#%%
camera_positions = np.array([[-1.0, -8.18, 2.08],
                    [-5.27, -7.03, 2.24],
                    [-9.12, -3.12, 2.03],
                    ])
t = np.linspace(0,1,camera_positions.shape[0])
t_interp = np.linspace(0,1,25)
roll_values = [50, 76, 84]
roll_spline = si.interp1d(t,roll_values)
xyz_splines = [ si.interp1d(t, camera_positions[:,i], 'quadratic') for i in range(3)]
xyz_interp = []
for i in range(3):
    xyz_interp.append(xyz_splines[i](t_interp))
roll_interp = roll_spline(t_interp)
camera_traj = np.column_stack(xyz_interp)
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

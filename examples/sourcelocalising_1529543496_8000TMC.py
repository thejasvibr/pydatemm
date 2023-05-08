'''
Localising overlapping calls: 2018-06-21
========================================
21st June 2018 is special as the microphone positions were exactly
measured using a TotalStation. Here let's try to localise sources in the audio file 
with POSIX timestamp 1529543496. This audio file corresponds to P00/8000 TMC of
the thermal cameras. 


'''
import glob
from natsort import natsorted
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
#import pyvista as pv
import soundfile as sf
from scipy.spatial import distance_matrix, distance
euclidean = distance.euclidean
import os
import subprocess
import time
import tqdm
import yaml

common_parameters = {}
common_parameters['audiopath'] = '1529543496_input/video_synced10channel_first15sec_1529543496.WAV'
common_parameters['arraygeompath'] = '1529543496_input/Sanken9_centred_mic_totalstationxyz.csv' #'1529543496_input/arraygeom_2018-06-21_1529543496.csv'
common_parameters['dest_folder'] = '1529543496_output'
common_parameters['K'] = 3
common_parameters['maxloopres'] = 1e-4
common_parameters['thresh_tdoaresidual'] = 1e-8 # s
common_parameters['remove_lastchannel'] = "False"
common_parameters['highpass_order'] = "2,20000"
#common_parameters['channels'] = "0,1,2,3,4,5,6"
# simdata = pd.read_csv('multibatsim_xyz_calling.csv').loc[:,'x':]
# simdata_callpoints = simdata[simdata['emission_point']]

array_geom = pd.read_csv(common_parameters['arraygeompath']).loc[:,'x':'z'].to_numpy()
#array_geom[:,:2] *= -1
#%% Make the yaml file for the various time points
step_size = 0.003
window_size = 0.016
time_starts = np.arange(12.5, 14.0, step_size)

if not os.path.exists(common_parameters['dest_folder']):
    os.mkdir(common_parameters['dest_folder'])

# incoporate the time windows into the parameter file
relevant_time_windows = np.around(time_starts, 3)
# split the time_windows according to the total number of paramsets to be generated
split_timepoints = np.array_split(relevant_time_windows, 50)
#%%
for i, each in enumerate(split_timepoints):
    common_parameters['start_time'] = str(each.tolist())[1:-1]
    
    fname = os.path.join(common_parameters['dest_folder'], 
                         f'paramset_1529543496_{i}.yaml')
    ff = open(fname, 'w+')
    yaml.dump(common_parameters, ff)

# #%%    
# # Now create a bash file which runs each of the parameter sets!
# all_param_files = natsorted(glob.glob(os.path.join(common_parameters['dest_folder'],'*.yaml')))
# common_command = "python -m pydatemm -paramfile"
# bash_filename = "1529543496_runs.sh"
# with open (bash_filename, 'w') as rsh:  
#     for paramfile in all_param_files:
#         rsh.writelines(common_command + f" {paramfile}" + " & \n")
# #%%
# os.system(f"bash {bash_filename}")
# #%%
# # load all the results into a dictionary
# result_files = natsorted(glob.glob(common_parameters['dest_folder']+'/*.csv'))
# param_files = glob.glob(common_parameters['dest_folder']+'/*.yaml')

# while len(result_files) !=len(param_files):
#     result_files = natsorted(glob.glob(common_parameters['dest_folder']+'/*.csv'))
#     param_files = glob.glob(common_parameters['dest_folder']+'/*.yaml')
#     time.sleep(5.0)
#     print(f'{len(result_files)} results of {len(param_files)} parameter files ready')
    
# all_results = []
# if len(result_files)==len(param_files):
#     results = {}
#     for i,_ in enumerate(param_files):
#         all_results.append(pd.read_csv(result_files[i]))
# else: 
#     raise IndexError('Num result files dont match the time poitns')
# all_sources = pd.concat(all_results).reset_index(drop=True)
# all_posns = all_sources.loc[:,['x','y','z','tdoa_res']].to_numpy()

# #%%
# # Now load the video flight trajectories and transform them from the 
# # camera to TotalStation coordinate system

# flight_traj = pd.read_csv('1529543496_input/xyz_data/bat_trajs_round1_sanken9_centred.csv')
# transform_matrix = pd.read_csv('1529543496_input/xyz_data/Sanken9_centred-video-to-TotalStation_transform.csv')
# transform = transform_matrix.to_numpy()[:,1:]
# bring_to_totalstation = lambda X: np.matmul(transform, X)
# flight_traj_conv = flight_traj.loc[:,'x':'z'].to_numpy()
# flight_traj_conv = np.apply_along_axis(bring_to_totalstation, 1, flight_traj_conv)
# flight_traj_conv = pd.DataFrame(flight_traj_conv, columns=['x','y','z'])
# flight_traj_conv['batid'] = flight_traj['batid'].copy()
# flight_traj_conv['frame'] = flight_traj['frame'].copy()
# flight_traj_conv['time'] = flight_traj_conv['frame']/25 + 310/25

# # filter out the relevant flight trajectory for the window
# timewindow_rows = np.logical_and(flight_traj_conv['time'] >= start_time, 
#                                         flight_traj_conv['time'] <= end_time) 
# flighttraj_conv_window = flight_traj_conv.loc[timewindow_rows,:]

# #%% Keep only those that are within a few meters of any bat trajectory positions
# distmat = distance_matrix(flighttraj_conv_window.loc[:,'x':'z'].to_numpy(), all_posns[:,:-1])
# nearish_posns = np.where(distmat<0.5) # all points that are at most 8 metres from any mic
# sources_nearish = all_posns[np.unique(nearish_posns[1]),:]

# # find the start and end times of the positions that are nearby. 
# sources_w_time  = all_sources.loc[np.unique(nearish_posns[1]),['x','y','z','t_start','t_end']].reset_index(drop=True)

# #%%
# # Run a DBSCAN on the nearish sources to get call centres
# from sklearn.cluster import DBSCAN
# clusters = DBSCAN(eps=0.1).fit(sources_nearish[:,:-1])

# cluster_centres = []
# for lab in np.unique(clusters.labels_):
#     if not lab == -1:
#         inds = np.where(clusters.labels_==lab)
#         cluster_points = sources_nearish[inds, :-1].reshape(-1,3)
#         centroid = np.median(cluster_points,axis=0)
#         print(lab, centroid)
#         cluster_centres.append(centroid)
# cluster_centres = np.array(cluster_centres)
# #%%


# # call_positions = flight_traj[flight_traj['emission_point']]
# mic_video_xyz = pd.read_csv('1529543496_input/xyz_data/Sanken9_centred_mic_videoxyz.csv')
# mic_video_xyz.columns = [0,'x','y','z','micname']
# # dist_mat = distance_matrix(all_posns[:,:3], call_positions.loc[:,'x':'z'].to_numpy())

# # #%% Filter out those points that are within ~2  meters of the known flight
# # # trajectories
# # distmat_flighttraj = distance_matrix(all_posns[:,:3], flight_traj.loc[:,'x':'z'].to_numpy())
# # sensible_posns = np.argwhere(distmat_flighttraj<=0.3)
# # all_sensible_posns = all_posns[sensible_posns[:,0],:3]
# #%%
# box = pv.Box(bounds=(0,5,0,9,0,3))
# plotter = pv.Plotter()

# def mouse_callback(x):
#     print(f'camera position: {plotter.camera.position})')
#     print(f'\n az, roll, el: {plotter.camera.azimuth, plotter.camera.roll, plotter.camera.elevation}')
#     print(f'\n view angle, focal point: {plotter.camera.view_angle, plotter.camera.focal_point}')
# plotter.track_click_position(mouse_callback)

# #plotter.add_mesh(box, opacity=0.3)
# # include the mic array
# for each in array_geom:
#     plotter.add_mesh(pv.Sphere(0.03, center=each), color='g')

# for i in [1,2,3]:
#     plotter.add_mesh(pv.lines_from_points(array_geom[[0,i],:]), line_width=5)
# plotter.add_mesh(pv.lines_from_points(array_geom[4:,:]), line_width=5)

# for every in sources_nearish[:,:]:
#     x,y,z = every[:-1]
#     plotter.add_mesh(pv.Sphere(0.02, center=[x,y,z]), color='r', opacity=0.5)

# for each in cluster_centres:
#     plotter.add_mesh(pv.Sphere(0.1, center=each), color='w', opacity=0.9)

# plotter.camera.position = (-2.29, -11.86, 1.50)
# plotter.camera.azimuth = 0.0
# plotter.camera.roll = 75
# plotter.camera.elevation = 0.0
# plotter.camera.view_angle = 30.0
# plotter.camera.focal_point = (0.42, 1.59, 0.68)


# cmap = matplotlib.cm.get_cmap('Spectral')
# fractions = np.linspace(0,1,np.unique(flight_traj_conv['batid']).size)
# colors = [cmap(frac)[:-1] for frac in fractions]

# # plot the flight trajectories and call emission points
# for key, subdf in flighttraj_conv_window.groupby('batid'):
#     traj_line = pv.lines_from_points(subdf.loc[:,'x':'z'].to_numpy())
#     plotter.add_mesh(traj_line, line_width=7, color=colors[key-1])

# plotter.show()
# #%%
# # For each cluster centre - let's just say they are all correct. 
# # If a call had been emitted from that point - using the expected TOFs we should
# # recover bat calls at the calculated TOAs... 
# batxyz = flighttraj_conv_window.loc[:,'x':'z'].dropna().to_numpy()
# timewindows = []
# for each in cluster_centres:
#     xyz  = each.reshape(1,3)
#     tof = distance_matrix(xyz, array_geom)/340.0
#     # find closest video point
#     distmat = distance_matrix(xyz, batxyz)
#     if np.min(distmat)<0.3:
#         closest_point = batxyz[np.argmin(distmat)]
#         # get video timestamp of the closest_point
#         ind_closest_point = np.where(flighttraj_conv_window.loc[:,'x':'z']==closest_point)
#         row = flighttraj_conv_window.index[ind_closest_point[0][0]]
#         candidate_emission_time = flighttraj_conv_window.loc[row,'time']    
#         potential_toas = candidate_emission_time + tof
#         toa_minmax = np.percentile(np.around(potential_toas,3), [0,100])
#         timewindows.append(toa_minmax)
#         print(toa_minmax, np.around(distmat.min(),3), flighttraj_conv_window.loc[row, 'batid'])
        
# #%%
# for each in timewindows:
#     nchannels = array_geom.shape[0]
#     plt.figure()
#     plt.subplot()

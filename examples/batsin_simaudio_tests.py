'''
Localising overlapping calls: simulated audio case
==================================================
Here we'll run the 


'''
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
try:
    import pyvista as pv
except:
    print('Cant import pyvista!')
import soundfile as sf
from scipy.spatial import distance_matrix, distance
euclidean = distance.euclidean
import os
import subprocess
import time
import tqdm
import yaml

common_parameters = {}
common_parameters['audiopath'] = '3-bats_trajectory_simulation_1-order-reflections.wav'
#common_parameters['arraygeompath'] = 'mic_xyz_multibatsim.csv'
common_parameters['arraygeompath'] = 'mic_xyz_multibatsim_noisy0.05m.csv'
common_parameters['dest_folder'] = 'multibatsim_results'
common_parameters['K'] = 2
common_parameters['maxloopres'] = 1e-4
common_parameters['thresh_tdoaresidual'] = 1e-10 # s
common_parameters['remove_lastchannel'] = "False"
common_parameters['min_peak_dist'] = 0.35e-4 # s
common_parameters['window_size'] = 0.010 # s

simdata = pd.read_csv('multibatsim_xyz_calling.csv').loc[:,'x':]
simdata_callpoints = simdata[simdata['emission_point']]

array_geom = pd.read_csv(common_parameters['arraygeompath']).loc[:,'x':'z'].to_numpy()

#%% Make the yaml file for the various time points
step_size = 0.003
window_size = 0.010
time_starts = np.arange(0, 0.5, step_size)

if not os.path.exists(common_parameters['dest_folder']):
    os.mkdir(common_parameters['dest_folder'])

# incoporate the time windows into the parameter file
relevant_time_windows = np.around(time_starts[time_starts<=0.249], 3)
# split the time_windows according to the total number of cores to be used.
split_timepoints = np.array_split(relevant_time_windows, 4)

for i, each in enumerate(split_timepoints):
    common_parameters['start_time'] = str(each.tolist())[1:-1]
    
    fname = os.path.join(common_parameters['dest_folder'], 
                         f'paramset_multibatsim_{i}.yaml')
    ff = open(fname, 'w+')
    yaml.dump(common_parameters, ff)


#%%    
# Now create a bash file which runs each of the parameter sets!
all_param_files = natsorted(glob.glob(os.path.join(common_parameters['dest_folder'],'*.yaml')))
common_command = "python -m pydatemm -paramfile"
with open ('bats_simaudio_runs.sh', 'w') as rsh:  
    for paramfile in all_param_files:
        rsh.writelines(common_command + f" {paramfile}" + " & \n")
# #%%
# os.system("bash bats_simaudio_runs.sh")
# #%%
# # load all the results into a dictionary
# result_files = natsorted(glob.glob(common_parameters['dest_folder']+'/*.csv'))
# all_results = []
# if len(result_files)==relevant_time_windows.size-1:
#     results = {}
#     for i in range(relevant_time_windows.size-1):
#         all_results.append(pd.read_csv(result_files[i]))
# else: 
#     raise IndexError('Num result files dont match the time poitns')
# all_sources = pd.concat(all_results).reset_index(drop=True)
# all_posns = all_sources.loc[:,['x','y','z','tdoa_res']].to_numpy()
                                                        
# #%%
# flight_traj = pd.read_csv('multibatsim_xyz_calling.csv')
# call_positions = flight_traj[flight_traj['emission_point']]

# dist_mat = distance_matrix(all_posns[:,:3], call_positions.loc[:,'x':'z'].to_numpy())

# #%% Filter out those points that are within ~2  meters of the known flight
# # trajectories
# distmat_flighttraj = distance_matrix(all_posns[:,:3], flight_traj.loc[:,'x':'z'].to_numpy())
# sensible_posns = np.argwhere(distmat_flighttraj<=0.3)
# all_sensible_posns = all_posns[sensible_posns[:,0],:3]
# #%%
# try:
#     box = pv.Box(bounds=(0,4,0,9,0,3))
#     plotter = pv.Plotter()
#     plotter.add_mesh(box, opacity=0.3)
#     colors = ['r', 'b', 'k']
    
#     # plot the flight trajectories and call emission points
#     for key, subdf in flight_traj.groupby('batnum'):
#         for each in subdf.loc[:,'x':'z'].to_numpy():
#             plotter.add_mesh(pv.Sphere(0.05, center=each), color=colors[key-1])
#     for key, subdf in call_positions.groupby('batnum'):
#         for each in subdf.loc[:,'x':'z'].to_numpy():
#             plotter.add_mesh(pv.Sphere(0.1, center=each), color=colors[key-1])
#     # include the mic array
#     for each in array_geom:
#         plotter.add_mesh(pv.Sphere(0.03, center=each), color='g')
    
#     for every in all_sensible_posns:
#         plotter.add_mesh(pv.Sphere(0.2, center=every[:3]), color='white', opacity=0.5)
    
#     plotter.show()
# except:
#     print('Cant execute pyvista visualisation')
# -*- coding: utf-8 -*-
"""
Run the same time window of simulated audio with different channel orders
=========================================================================

@author: theja
"""
import numpy as np 
import yaml 
import os 
import subprocess

#%% Make the simulated audio and groundtruth files 
roomdim = '4,9,3'
inputfolder = str(os.path.join('initialvertex_tests','nbat8'))
try:
    os.makedirs(inputfolder)
except:
    pass
try:
    os.makedirs(os.path.join(inputfolder,'nbats8outdata'))
except:
    pass

subprocess.run(f"python  multibatsimulation.py -nbats 8 -ncalls 5 -all-calls-before 0.1 -room-dim {roomdim} -seed 82319 -input-folder {inputfolder} -ray-tracing True",
               shell=True)


#%% Make a simple set of parameters where the first channels is changed. 
K = 7
paramset = {}
paramset['audiopath'] = os.path.join(inputfolder,'8-bats_trajectory_simulation_raytracing-1.wav')
paramset['arraygeompath'] =  os.path.join(inputfolder,'mic_xyz_multibatsim.csv')
paramset['dest_folder'] = os.path.join(inputfolder,'nbats8outdata')
paramset['maxloopres'] = 0.0005
paramset['min_peak_dist'] = 0.00025
paramset['num_jobs']: 1
paramset['remove_lastchannel'] = 'False'
paramset['K'] = K
times = np.arange(0.15,0.2,1e-3)
times_str = str(list(times))[1:-1]
paramset['start_time'] = times_str
paramset['step_size'] = 0.001
paramset['thresh_tdoaresidual'] = 1.0e-08
paramset['window_size'] = 0.01

commonparamfile_name = os.path.join(paramset['dest_folder'])

run_list = [os.path.join(commonparamfile_name,f'paramset_K{K}-startch_{each}.yaml') for each in range(8)]
channel_list = []
for each in range(8):
    allchannels = range(8)
    channelset = set(allchannels) - set([each])
    ordered_channel = [each] + list(channelset)
    channel_list.append(ordered_channel)
#%%
for run_name, firstchannel, channelset in zip(run_list, range(8), channel_list):
    paramset['channels'] = str(channelset)[1:-1]
    paramset['run_name'] = f'K{K}firstch_{firstchannel}'
    with open(run_name, 'w') as ff:
        yaml.dump(paramset, ff)

# #%%
# for ii in run_list:
#     print(f'Running {ii}')
#     #os.system(f'python -m pydatemm -paramfile {ii}')
#     output = subprocess.run(f'python -m pydatemm -paramfile {ii}')
#     print(output)



'''
Localising overlapping calls: simulated audio case
==================================================
Here we'll run the 


'''
import argparse
import numpy as np 
import pandas as pd
import os
import pathlib
import yaml
import soundfile as sf

# thanks Fransisco (https://stackoverflow.com/a/71845536/4955732)
def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


args = argparse.ArgumentParser()
args.add_argument('-audiopath',type=str, help='path to audio file')
args.add_argument('-arraygeompath', type=str, help='Path to array geometry csv file')
args.add_argument('-dest-folder', type=str, )
args.add_argument('-K', type=int)
args.add_argument('-maxloopres', type=float)
args.add_argument('-thresh-tdoaresidual', type=float)
args.add_argument('-remove-lastchannel', type=str)
args.add_argument('-min-peak-dist', type=float)
args.add_argument('-window-size', type=float)
args.add_argument('-run-name', type=str)
args.add_argument('-step-size', type=float, default=1e-3)
args.add_argument('-num-jobs', type=int, default=10)
args.add_argument('-channels', type=str, default=None)
common_parameters = args.parse_args()

for arg in vars(common_parameters):
    print(arg, getattr(common_parameters, arg))


# common_parameters = {}
# common_parameters['audiopath'] = '3-bats_trajectory_simulation_1-order-reflections.wav'
# common_parameters['arraygeompath'] = 'mic_xyz_multibatsim.csv'
# #common_parameters['arraygeompath'] = 'mic_xyz_multibatsim_noisy0.05m.csv'
# common_parameters['dest_folder'] = 'multibatsim_results_origxyz'
# common_parameters['K'] = 3
# common_parameters['maxloopres'] = 1e-4
# common_parameters['thresh_tdoaresidual'] = 1e-8 # s
# common_parameters['remove_lastchannel'] = "False"
# common_parameters['min_peak_dist'] = 0.25e-3 # s
# common_parameters['window_size'] = 0.010 # s
# common_parameters['run_name'] = 'simaudio'# the posix timestamp will be added later!

# simdata = pd.read_csv('multibatsim_xyz_calling.csv').loc[:,'x':]
# array_geom = pd.read_csv(common_parameters['arraygeompath']).loc[:,'x':'z'].to_numpy()

# #%% Make the yaml file for the various time points
step_size = common_parameters.step_size
window_size = common_parameters.window_size
audio_durn = sf.info(common_parameters.audiopath).duration
time_starts = np.arange(0, audio_durn, step_size)

if not os.path.exists(common_parameters.dest_folder):
    os.mkdir(common_parameters.dest_folder)

# split the time_windows according to the total number of cores to be used.
split_timepoints = np.array_split(time_starts, common_parameters.num_jobs)
#%%
common_parameters_yaml = namespace_to_dict(common_parameters)

for i, each in enumerate(split_timepoints):
    common_parameters_yaml['start_time'] = str(each.tolist())[1:-1]
    
    fname = os.path.join(common_parameters.dest_folder, 
                          f'paramset_{common_parameters_yaml["run_name"]}_{i}.yaml')
    ff = open(fname, 'w+')
    yaml.dump(common_parameters_yaml, ff)


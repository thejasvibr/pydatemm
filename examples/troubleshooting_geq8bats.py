# -*- coding: utf-8 -*-
"""
Are all the relevant TDOAs being picked up?
===========================================
Even after increasing the K from 3-6 I don't see a marked
increase in the detected calls for a given audio snippet. Why is this the case?



SOME PROBLEMS IN TOA CALCULATINO EVEN WITHOUT REVERB ALREADY!!

Created on Fri Jun 23 14:40:18 2023

@author: theja
"""
import numpy as np 
import os 
import pandas as pd
import scipy.signal as signal 
from scipy.spatial import distance_matrix
import soundfile as sf
import matplotlib.pyplot as plt 
import pydatemm.timediffestim as timediff
import subprocess
from source_traj_aligner import calculate_toa_channels
#%%
# Let's first create an audiofile with NO reflections of any kind to 
# test that our workflow is working out okay. 
roomdim = '4,9,3'
inputfolder = str(os.path.join('multibatno_reverb','nbat8'))
try:
    os.makedirs(inputfolder)
except:
    pass

subprocess.run(f"python  multibatsimulation.py -nbats 8 -ncalls 5 -all-calls-before 0.1 -room-dim {roomdim} -seed 82319 -input-folder {inputfolder} -ray-tracing False -ref-order 0",
               shell=True)

#%%
t_minemit, t_maxemit = 0.15, 0.160
sim_folder = os.path.join('initialvertex_tests',
                         'nbat8')
audiofile = os.path.join(sim_folder,
                         '8-bats_trajectory_simulation_raytracing-1.wav')

#audiofile = os.path.join('multibatno_reverb','nbat8','8-bats_trajectory_simulation_0-order-reflections.wav')
fs = sf.info(audiofile).samplerate
nchannels = sf.info(audiofile).channels
kwargs = {}
kwargs['fs'] = fs 
kwargs['min_peak_diff'] = 10e-6
kwargs['min_height'] = 0.01
kwargs['K'] = 20
#%%
micxyz = pd.read_csv(os.path.join(sim_folder,'mic_xyz_multibatsim.csv')).loc[:,'x':'z'].to_numpy()

# First load the relevant call emission point data between 0.15 - 0.182 
simdata = pd.read_csv(os.path.join(sim_folder, 'multibatsim_xyz_calling.csv'))
call_data = simdata[simdata['emission_point']==True]
call_data = call_data[call_data['t'].between(t_minemit, t_maxemit)].sort_values('t')
call_data = call_data.rename(columns={'batnum':'batid'})

#%% 
# Get expected TDOA for each call 
from itertools import combinations
unique_pairs = list(combinations(range(nchannels), 2))
unique_pairs = list(map(lambda X: sorted(X, reverse=True), unique_pairs))
expected_tdoa = {tuple(chpair) : [] for chpair in unique_pairs}

toa_min = []
toa_max = []
for idx, row in call_data.iterrows():
    batxyz = [row[ax] for ax in ['x','y','z']]
    toa_set = calculate_toa_channels(row['t'], call_data, row['batid'], micxyz).flatten()
    toa_min.append(toa_set.min())
    toa_max.append(toa_set.max())
    dist_mat = distance_matrix(np.array(batxyz).reshape(1,-1), 
                               micxyz)
    for chpair, tdoas in expected_tdoa.items():
        chb,cha = chpair
        rangediff = dist_mat[:,chb] - dist_mat[:,cha]
        tde = float(rangediff/343)
        tdoas.append(tde)
call_data['toa_min'] = toa_min
call_data['toa_max'] = toa_max

#%% Now load the audio based on the min-max TOA data
# so we know for SURE that the calls are in there. 
all_tmin = call_data['toa_min'].min()
all_tmax = call_data['toa_max'].max()+0.005

audio, _ = sf.read(audiofile, start=int(fs*all_tmin), stop=int(fs*all_tmax))


#%% Calculate TDOA at all channels

multich_cc = timediff.generate_multich_crosscorr(audio, **kwargs )

# Implement zero-ing of all beyond max-delay to focus attention 
# on the plausible range

mic2mic = distance_matrix(micxyz,micxyz)
for  chpair, cc in multich_cc.items():
    max_delay = mic2mic[chpair[0],chpair[1]]/343
    max_delay_samples = int(max_delay*fs)
    halfway_point = int(cc.shape[0]*0.5)
    left_edge = np.max([halfway_point-max_delay_samples,0])
    right_edge = np.min([halfway_point+max_delay_samples, cc.shape[0]])
    cc[:left_edge] *= 0
    cc[right_edge:] *= 0

kwargs['nchannels'] = audio.shape[1]
cc_peaks = timediff.get_multich_tdoas(multich_cc, **kwargs)

top_K_tdes = {}
for ch_pair, tdes in cc_peaks.items():
    descending_quality = sorted(tdes, key=lambda X: X[-1], reverse=True)
    top_K_tdes[ch_pair] = []
    for i in range(kwargs['K']):
        try:
            top_K_tdes[ch_pair].append(descending_quality[i])
        except:
            pass

#%%
fig,ax = plt.subplots(nrows=1,ncols=1)
chpair = (4,1)

tleft,tright = -(audio.shape[0])/fs, (audio.shape[0])/fs
t = np.linspace(tleft, tright, (audio.shape[0]*2)-1)
ax.plot(t, multich_cc[chpair])
#for each in 
samplepoints = list(map(lambda X: X[0], top_K_tdes[chpair]))
timepoints = list(map(lambda X: X[1], top_K_tdes[chpair]))
ax.plot(timepoints, multich_cc[chpair][samplepoints],'*')

ax.plot(expected_tdoa[chpair], np.tile(0.2, len(expected_tdoa[chpair])),'r*')

# total_resid = np.array(sorted(timepoints)) - np.array(sorted(expected_tdoa[chpair]))

# print(total_resid/(1/fs))








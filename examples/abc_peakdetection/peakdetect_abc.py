# -*- coding: utf-8 -*-
"""
Peak detection, with approximate Bayesian computation 
=====================================================
Created on Sun Jun 25 07:55:05 2023

@author: theja
"""

import numpy as np 
import os 
import pandas as pd
import scipy.signal as signal 
from scipy.spatial import distance_matrix, distance
from scipy.interpolate import interp1d
import soundfile as sf
import matplotlib.pyplot as plt 
if not 'timediff' in dir():
    import pydatemm.timediffestim as timediff

os.chdir('..\\')
import subprocess
from source_traj_aligner import calculate_toa_channels
#%%
# Let's first create an audiofile with NO reflections of any kind to 
# test that our workflow is working out okay. 

roomdim = '4,9,3'

inputfolder = os.path.join('abc_peakdetection','nbat8')
try:
    os.makedirs(inputfolder)
except:
    pass


# subprocess.run(f"python  multibatsimulation.py -nbats 8 -ncalls 5 -all-calls-before 0.1 -room-dim {roomdim} -seed 82319 -input-folder {inputfolder} -ray-tracing False -ref-order 0",
#                shell=True)

# subprocess.run(f"python  multibatsimulation.py -nbats 8 -ncalls 5 -all-calls-before 0.1 -room-dim {roomdim} -seed 82319 -input-folder {inputfolder} -ray-tracing True -ref-order 1",
#                shell=True)

# interpolate the 

#%%
t_minemit, t_maxemit = 0.0, 0.04

audiofile = os.path.join(inputfolder, '8-bats_trajectory_simulation_raytracing-1.wav')

#audiofile = os.path.join('multibatno_reverb','nbat8','8-bats_trajectory_simulation_0-order-reflections.wav')

fs = sf.info(audiofile).samplerate
nchannels = sf.info(audiofile).channels
kwargs = {}
kwargs['fs'] = fs 
kwargs['min_peak_diff'] = 10e-6
kwargs['min_height'] = 0.01
kwargs['K'] = 5
#%%
micxyz = pd.read_csv(os.path.join(inputfolder,'mic_xyz_multibatsim.csv')).loc[:,'x':'z'].to_numpy()

# First load the relevant call emission point data between 0.15 - 0.182 
simdata = pd.read_csv(os.path.join(inputfolder, 'multibatsim_xyz_calling.csv'))
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
all_tmax = call_data['toa_max'].max()+5e-3

print(all_tmin, all_tmax)

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

ccsize = multich_cc[chpair].size
expected_tdoa_ind = {}
for ch_pair, tdoas in expected_tdoa.items():
    tdoas_ind = np.array(tdoas)*fs
    tdoas_ind += ccsize*0.5
    tdoas_ind = np.int64(tdoas_ind)
    expected_tdoa_ind[ch_pair] = tdoas_ind

multich_cc_env = {}
for ch_pair, cc in multich_cc.items():
    envelope = abs(signal.hilbert(cc))
    multich_cc_env[ch_pair] = envelope

#%%
# What if we're more low-tech? Just amplify the regions of interest and see
# if there are peaks to be detected around there only. 

t_cc = lambda X,fs: np.linspace(-(int(X.size*0.5))/fs, ((int(X.size*0.5))+1)/fs, X.size)
t_chpair = t_cc(multich_cc[chpair], fs)

windurn = 0.15e-3
winsize = int(fs*windurn)
windows = {}
windowed_cc = {}
for ch_pair, cc in multich_cc.items():
    windows_ofinterest = np.zeros(multich_cc[ch_pair].size)
    all_windows = np.zeros((len(expected_tdoa[ch_pair]),windows_ofinterest.shape[0]))
    for i,exp_tdoa in enumerate(expected_tdoa_ind[ch_pair]):
        start = exp_tdoa - int(winsize*0.5)
        stop = start + winsize
        window = signal.windows.boxcar(winsize)
        this_window = windows_ofinterest.copy()
        all_windows[i,start:stop] += window
    consensus_window = np.sum(all_windows, axis=0)
    windows[ch_pair] = np.array(consensus_window>0, dtype=np.int16)
    windowed_cc[ch_pair] = windows[ch_pair]*cc
    
#%%
# What if we're more subtle. The windows peak at some value >= 1. 
amp_factor = 0.5
subtle_windows = {}
windowed_subtle_cc = {}
for ch_pair, window in windows.items():
    subtle_windows[ch_pair] = window*amp_factor + 1 
    windowed_subtle_cc[ch_pair] = multich_cc[ch_pair]*subtle_windows[ch_pair]

#%%
# Now perform the peak detection on the windowed ccs:
chpair= (4,3)
exp_tdoas = np.array(expected_tdoa[chpair]).reshape(-1,1)
tdoa_diff = distance_matrix(exp_tdoas, exp_tdoas)
min_peak_dist = tdoa_diff[tdoa_diff>0].min()*0.9
kwargs['min_peak_diff'] = min_peak_dist
kwargs['K'] = 7
wincc_peaks = timediff.get_multich_tdoas(windowed_cc, **kwargs)
subtle_peaks = timediff.get_multich_tdoas(windowed_subtle_cc, **kwargs)


def get_top_K_tdes(all_peaks, **kwargs):
    top_K_tdes = {}
    for ch_pair, tdes in all_peaks.items():
        descending_quality = sorted(tdes, key=lambda X: X[-1], reverse=True)
        top_K_tdes[ch_pair] = []
        for i in range(kwargs['K']):
            try:
                top_K_tdes[ch_pair].append(descending_quality[i])
            except:
                pass
    return top_K_tdes

# aggressive windowing peaks
aggressive_topK = get_top_K_tdes(wincc_peaks, **kwargs)
# subtle 
subtle_topK = get_top_K_tdes(subtle_peaks, **kwargs)


fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

plt.sca(ax[0])
t_peaks = list(map(lambda X: X[1], aggressive_topK[chpair]))
plt.plot(t_chpair, multich_cc[chpair])
plt.plot(t_peaks, np.tile(0.2,len(t_peaks)), '*')
ax[0].set_title('Aggressive windowing (binary)')
ax[0].plot(expected_tdoa[chpair], multich_cc[chpair][expected_tdoa_ind[chpair]],'r*')

plt.sca(ax[1])
t_peaks = list(map(lambda X: X[1], subtle_topK[chpair]))
plt.plot(t_chpair, multich_cc[chpair])
plt.plot(t_peaks, np.tile(0.2,len(t_peaks)), '*')
ax[1].set_title('Subtle windowing')
ax[1].plot(expected_tdoa[chpair], multich_cc[chpair][expected_tdoa_ind[chpair]],'r*')
#%%

f, aa = plt.subplots(nrows=2,ncols=1, sharex=True, sharey=True)
plt.sca(aa[0])
plt.specgram(audio[:,chpair[0]], Fs=fs,)
plt.sca(aa[1])
plt.specgram(audio[:,chpair[1]], Fs=fs,)


#%%
# Let's now 

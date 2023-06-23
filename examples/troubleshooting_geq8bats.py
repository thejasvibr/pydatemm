# -*- coding: utf-8 -*-
"""
Are all the relevant TDOAs being picked up?
===========================================
Even after increasing the K from 3-6 I don't see a marked
increase in the detected calls for a given audio snippet. Why is this the case?




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

#%%
t_start, t_stop = 0.15, 0.160
sim_folder = os.path.join('initialvertex_tests',
                         'nbat8')
audiofile = os.path.join(sim_folder,
                         '8-bats_trajectory_simulation_raytracing-1.wav')
fs = sf.info(audiofile).samplerate
audio, fs = sf.read(audiofile,
                    start=int(fs*t_start), stop=int(fs*t_stop))
kwargs = {}
kwargs['fs'] = fs 
kwargs['min_peak_diff'] = 0.0002
kwargs['min_height'] = 0.03
kwargs['K'] = 5
#%%
micxyz = pd.read_csv(os.path.join(sim_folder,'mic_xyz_multibatsim.csv')).loc[:,'x':'z'].to_numpy()

# First load the relevant call emission point data between 0.15 - 0.182 
simdata = pd.read_csv(os.path.join(sim_folder, 'multibatsim_xyz_calling.csv'))
call_data = simdata[simdata['emission_point']==True]
call_data = call_data[call_data['t'].between(t_start, t_stop)].sort_values('t')

#%% Calculate TDOA at all channels

multich_cc = timediff.generate_multich_crosscorr(audio, **kwargs )

# Implement zero-ing of all beyond max-delay to focus attention 
# on the plausible range

for  chpair, cc in multich_cc.items():
    max_delay = distance_matrix(micxyz[chpair,:],micxyz[chpair,:]).max()/340
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
# Get expected TDOA for each call 
from itertools import combinations
unique_pairs = list(combinations(range(audio.shape[1]), 2))
unique_pairs = list(map(lambda X: sorted(X, reverse=True), unique_pairs))
expected_tdoa = {tuple(chpair) : [] for chpair in unique_pairs}

for idx, row in call_data.iterrows():
    batxyz = [row[ax] for ax in ['x','y','z']]
    dist_mat = distance_matrix(np.array(batxyz).reshape(1,-1), 
                               micxyz)
    for chpair, tdoas in expected_tdoa.items():
        chb,cha = chpair
        rangediff = dist_mat[:,chb] - dist_mat[:,cha]
        tde = float(rangediff/340)
        tdoas.append(tde)


#%%
fig,ax = plt.subplots(nrows=1,ncols=1)
chpair = (7,5)

tleft,tright = -(audio.shape[0])/fs, (audio.shape[0])/fs
t = np.linspace(tleft, tright, (audio.shape[0]*2)-1)
ax.plot(t, multich_cc[chpair])
#for each in 
samplepoints = list(map(lambda X: X[0], top_K_tdes[chpair]))
timepoints = list(map(lambda X: X[1], top_K_tdes[chpair]))
ax.plot(timepoints, multich_cc[chpair][samplepoints],'*')

ax.plot(expected_tdoa[chpair], np.tile(0.2, len(expected_tdoa[chpair])),'r*')









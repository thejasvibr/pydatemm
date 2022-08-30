#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigating the pre-processing of cross-correlations for peak detection
=========================================================================
I've been observing that a lot of my bad source localisations are due to missing
TDEs already from the first stage itself. If the correct TDE peaks are not 
detected - of course the correct graphs can't be combined - and thus we 
get a poor localisation. 

Till now I've been implementing a simple peak detection routine - where the 
percentile of the relevant cc portion (+/- max delay) is used to set the thres-
hold for peak detection. A minimum distance is defined between detected peaks -
and this actually works much of the time. However, the details make all the
difference!

Here I'll be comparing TDE peak detection with the current implementation and 
the 'noneg' version. 

The 'noneg' peak detector
~~~~~~~~~~~~~~~~~~~~~~~~~
Nothing special here really. The CC output often has -ve values. The -ve values
are of no interest to us - as we're looking for +ve samples, and preferrably high
positive samples.

The idea is to do two things: 
    1. Set all -ve samples to 0
    2. See which threshold values typically make sense (I doubt that 95 %ile would
                                                        still work)
"""
#from ky2012_fullsim_chain import * 
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd 
import pyroomacoustics as pra
import numpy as np 
from pydatemm.timediffestim import get_peaks, generate_multich_crosscorr
from pydatemm.timediffestim import max_interch_delay as maxintch
import scipy.signal as signal 
from investigating_peakdetection_gccflavours import multich_expected_peaks

#%% Make the simulated audio

array_geom = pd.read_csv('../pydatemm/tests/scheuing-yang-2008_micpositions.csv').to_numpy()
# from the pra docs
room_dim = [9, 7.5, 3.5]  # meters
fs = 192000
ref_order = 2

reflection_max_order = ref_order

rt60_tgt = 0.3  # seconds
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

room = pra.ShoeBox(
    room_dim, fs=fs, materials=pra.Material(e_absorption),
    max_order=ref_order,
    ray_tracing=False,
    air_absorption=True)

call_durn = 7e-3
t_call = np.linspace(call_durn, int(fs*call_durn))
batcall = signal.chirp(t_call, 85000, t_call[-1], 9000,'linear')
batcall *= signal.hamming(batcall.size)
batcall *= 0.7

random = True
num_sources = int(np.random.choice(range(3,7),1)) # or overruled by the lines below.

xyzrange = [np.arange(0,dimension, 0.01) for dimension in room_dim]
if not random:
    sources = [[2.5, 1, 2.5],
               [4, 3, 1.5],
               [1, 4, 1.0],
               [8,7,0.5],
               ]
    num_sources = len(sources)
else:
    sources = []
    for each in range(num_sources):
        each_source = [float(np.random.choice(each,1)) for each in xyzrange]
        sources.append(each_source)

delay = np.linspace(0,0.050,len(sources))
for each, emission_delay in zip(sources, delay):
    room.add_source(position=each, signal=batcall, delay=emission_delay)

room.add_microphone_array(array_geom.T)
room.compute_rir()
print('room simultation started...')
room.simulate()
print('room simultation ended...')
# choose only the first 0.2 s 
sim_audio = room.mic_array.signals.T
if sim_audio.shape[0]>(int(fs*0.2)):
    sim_audio = sim_audio[:int(fs*0.2),:]
nchannels = array_geom.shape[0]

#%%
kwargs = {'nchannels':nchannels,
          'fs':fs,
          'array_geom':array_geom,
          'pctile_thresh': 75,
          'use_gcc':True,
          'gcc_variant':'phat', 
          'min_peak_diff':0.5e-4, 
          'vsound' : 343.0,}

multich_cc = generate_multich_crosscorr(sim_audio, **kwargs )
#%%
noneg_multich_peaks = {}
std_multich_peaks = {}

edges = list(map(lambda X: tuple(sorted(X, reverse=True)),
                  combinations(range(sim_audio.shape[1]),2)))

noneg_multich_peaks = {}

for chpair in edges:
    max_sample = int(maxintch(chpair, kwargs['array_geom'])*fs)
    minmaxsample =  np.int64(sim_audio.shape[0] + np.array([-max_sample, max_sample]))
    relevant_cc = multich_cc[chpair][minmaxsample[0]:minmaxsample[1]]
    # with no-neg preprocessing
    kwargs['no_neg'] = True 
    noneg_peaks = get_peaks(relevant_cc, **kwargs)
    noneg_peaks += minmaxsample[0]
    noneg_multich_peaks[chpair] = noneg_peaks
    # 'standard' peak detection
    kwargs['no_neg'] = False
    norm_peaks = get_peaks(relevant_cc, **kwargs)
    norm_peaks += minmaxsample[0]
    std_multich_peaks[chpair] = norm_peaks
    print(np.all(norm_peaks==noneg_peaks))
#%% Also compare what the residual is for 'noneg' method of peak detection. 
edges_str = list(map(lambda X:str(X), edges))
peak_detn_data = pd.DataFrame(data=[], index=range(len(sources)*2), 
                              columns=['source_no','no_neg']+edges_str)
num_sources = len(sources)
for i,s in enumerate(sources):
    exp_tdes_multich = multich_expected_peaks(sim_audio, [s], array_geom, fs=192000)
    for ch_pair, predicted_tde in exp_tdes_multich.items():
        for j, pks in enumerate([noneg_multich_peaks[ch_pair], std_multich_peaks[ch_pair]]):
            residual = np.min(np.abs(np.array(pks)-predicted_tde))
            if j==0:
                peak_detn_data.loc[i,'source_no'] = i
                peak_detn_data.loc[i,str(ch_pair)] = residual
                peak_detn_data.loc[i,'no_neg'] = True
            elif j==1:
                peak_detn_data.loc[i+num_sources,str(ch_pair)] = residual
                peak_detn_data.loc[i+num_sources,'no_neg'] = False
                peak_detn_data.loc[i+num_sources,'source_no'] = i

peak_detn_data['sum_resid'] = peak_detn_data.loc[:,'(1, 0)':'(7, 6)'].apply(np.sum, 1)
peak_detn_data['max_resid'] = peak_detn_data.loc[:,'(1, 0)':'(7, 6)'].apply(np.max, 1)
print(peak_detn_data)

qq = peak_detn_data.boxplot(column='sum_resid', by='no_neg')

# #%%
# from scipy.ndimage import convolve 

chpair = (3,2)
pk_pos = exp_tdes_multich[chpair]
max_sample = int(maxintch(chpair, kwargs['array_geom'])*fs)
minmaxsample =  np.int64(sim_audio.shape[0] + np.array([-max_sample, max_sample]))
plt.figure()
#plt.subplot(212, sharex=a0)
plt.plot(multich_cc[chpair])
plt.scatter(pk_pos, multich_cc[chpair][pk_pos], color='r', s=50)
for each in std_multich_peaks[chpair]:
    plt.plot(each, multich_cc[chpair][each], '^')
plt.xlim(minmaxsample[0], minmaxsample[1])

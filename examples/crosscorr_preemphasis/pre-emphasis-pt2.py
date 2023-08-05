# -*- coding: utf-8 -*-
"""
Testing the effect of cross-correlation pre-emphasis on peak-detection
======================================================================

Step 1: Create the pre-emphasis regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
I realised after seeing the actual time-delays, the time-delays can vary a lot
from point to point in the flight trajectory. Therefore, using just one TDE estimate
to pre-emphasise will therefore fail. Here I will use the total TDE min-max range
for a channel pair from all bats in the data to pre-emphasise the cross-correlation. 

Step 2: Effect of incoporating fine-scale TDE peak-to-peak distances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""


import numpy as np 
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt 
import os
from pydatemm import timediffestim as tdestim
from itertools import combinations
from scipy.spatial import distance_matrix as distmat
from scipy.interpolate import interp1d
import sys
sys.path.append('../')
from source_traj_aligner import calculate_toa_channels
from pydatemm.timediffestim import generate_multich_crosscorr, geometrically_valid
from pydatemm.timediffestim import get_multich_tdoas, get_topK_peaks
from pre_emph_utils import *
#%%
# Load raw data and upsample flight data. 
datafolder = '../multibat_stresstests/nbat8/'
arraygeomfile = os.path.join(datafolder,'mic_xyz_multibatsim.csv')
micxyz = pd.read_csv(arraygeomfile).loc[:,'x':].to_numpy()
audiofile = os.path.join(datafolder,'8-bats_trajectory_simulation_1-order-reflections.wav')
fs = sf.info(audiofile).samplerate
flightdata_file = os.path.join(datafolder,'multibatsim_xyz_calling.csv')
flighttraj = pd.read_csv(flightdata_file).rename({'batnum':'batid'},axis='columns')


upsampled_trajectories = upsample_xyz_to_ms_resolution(flighttraj,1e-3)
call_points = flighttraj[flighttraj['emission_point']].sort_values('t')


#%%
vsound = 343.0 #m/s
ideal_windowsize = np.round(distmat(micxyz, micxyz).max()/vsound,3)
approx_calldurn = 10e-3 
tstart, tstop = 10e-3, 20e-3
tstop += approx_calldurn

audio_snip, fs = sf.read(audiofile, start=int(fs*tstart), stop=int(fs*tstop))


# Get cross-corr
multich_crosscor = generate_multich_crosscorr(audio_snip)
#%%
# Take only flight traj within the tstart and tstop
flighttraj_window = upsampled_trajectories[upsampled_trajectories['t'].between(tstart, tstop)]

byid = flighttraj_window.groupby('batid')


tdoa_profiles_bybat = {}
channel_pairs = make_crosscor_chpairs(micxyz.shape[0])
for batid in byid.groups.keys():
    idx_range = byid.get_group(batid).index
    toas_over_time = np.zeros((len(idx_range),micxyz.shape[0]))
    tdoas_over_time = np.zeros((len(idx_range),len(channel_pairs)))

    for i,idx in enumerate(idx_range):
        row  = flighttraj_window.loc[idx,:]
        toas = calculate_toa_channels(row['t'], flighttraj_window, row['batid'], micxyz).flatten() 
        toas_over_time[i,:] = toas
        for col, chpair in enumerate(channel_pairs):
            chb, cha = chpair
            tdoas_over_time[i,col] = toas[chb] - toas[cha]
    tdoa_profiles_bybat[batid] = tdoas_over_time

# get the min-max tdoa range for each cross-cor  in channel 
tdoa_profile_minmax_bybat = {}
for batid, tdoa_over_time in tdoa_profiles_bybat.items():
    tdoa_profile_minmax_bybat[batid] = np.percentile(tdoa_over_time, [0,100], axis=0)    

#%%
# now assemble all the tdoa min-max ranges across bats
overall_minmax_tdoa_chwise = {}
for j,chpair in enumerate(channel_pairs):
    overall_minmax_tdoa_chwise[tuple(chpair)] = []
    for batid, tdoa_minmax in tdoa_profile_minmax_bybat.items():
        minmax_tdoa_chpair = np.array(tdoa_minmax[:,j]*fs, dtype=np.int64)
        overall_minmax_tdoa_chwise[tuple(chpair)].append(minmax_tdoa_chpair)

#%%
kwargs = {}
kwargs['fs'] = fs 
kwargs['min_height'] = 1e-2
kwargs['min_peak_diff'] = 50e-6
kwargs['array_geom'] = micxyz
kwargs['K'] = 10

# with pre-emphasis
preemph_multichcrosscor = {}
for chpair, crosscor in multich_crosscor.items():
    profile = np.zeros(crosscor.size)
    for ii,each in enumerate(overall_minmax_tdoa_chwise[chpair]):
        tdoa_min, tdoa_max = np.int64(each+crosscor.size*0.5)
        print(ii, tdoa_min, tdoa_max)
        profile[tdoa_min:tdoa_max] += 1 
    preemph_crosscor = np.zeros(crosscor.size)
    preemph_crosscor[np.nonzero(profile)[0]] = crosscor[np.nonzero(profile)]
    preemph_multichcrosscor[chpair] = preemph_crosscor
#%%
# Compare peaks from raw and pre-emphasised audio


# TDE peaks without pre-emphasis - raw cross-correlation
normal_tdoas = get_multich_tdoas(multich_crosscor, **kwargs)
geomvalid_multichcrosscor = geometrically_valid(normal_tdoas, **kwargs)
normal_tdes = get_topK_peaks(geomvalid_multichcrosscor, **kwargs)

# TDE peaks with pre-emphasis 
preemph_tdoas = get_multich_tdoas(preemph_multichcrosscor, **kwargs)
preemph_tdes = get_topK_peaks(preemph_tdoas, **kwargs)

#%%
# What are the ground-truth TDOAs?
actual_calls = call_points[call_points['t']<tstop].loc[:,'x':]

# only keep the calls that will arrive within tstop
toa_minmax = np.zeros((actual_calls.shape[0],2))
for i, (idx, row) in enumerate(actual_calls.iterrows()):
    toas = calculate_toa_channels(row['t'], flighttraj_window, row['batid'], micxyz)
    toamintoamax = np.percentile(toas, [0,100])
    toa_minmax[i,:] = toamintoamax
actual_calls.loc[:,'toa_min'] = toa_minmax[:,0]
actual_calls.loc[:,'toa_max'] = toa_minmax[:,1]
actual_calls = actual_calls[actual_calls.loc[:,'toa_max']+approx_calldurn*0.5<=tstop+1e-3]

# What are the expected TDOAs from the actual calls
expected_tdes = get_chpair_tdes_from_sources(actual_calls, flighttraj_window, micxyz)
expected_tde_samples = {}
for chpair, tdes in expected_tdes.items():
    samples = np.array(np.array(tdes)*fs+multich_crosscor[chpair].size*0.5, dtype=np.int64)
    expected_tde_samples[chpair] = samples.flatten()


#%%
# Quantifying error in peak-detection 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here let's quantify the peak detection error - and moe importantly compare the 
# errors between raw-CC and pre-emphasised CC. In addition we can also 
# check the effect of limiting peak-to-peak distances in an informed manner. 


#%% for plotting
chpair = (4,3)
raw_tdeinds, *_ = zip(*normal_tdes[chpair])
raw_tdeinds = list(raw_tdeinds)

pemp_tdeinds, *_ = zip(*preemph_tdes[chpair])
pemp_tdeinds = list(pemp_tdeinds)
this_crosscorr = multich_crosscor[chpair]

expected_tde = expected_tde_samples[chpair]

plt.figure()
plt.plot(this_crosscorr)
plt.plot(pemp_tdeinds, this_crosscorr[pemp_tdeinds],'g*')
plt.plot(raw_tdeinds, this_crosscorr[raw_tdeinds]+0.2,'k*')
plt.plot(expected_tde, this_crosscorr[expected_tde]+0.3,'r*')

raise ValueError('WHY ARE THERE PREDICTED TDE PEAKS WHERE THE CROSS-CORR IS SO FLAT?????')

# -*- coding: utf-8 -*-
"""
TDE pre-emphasis and its effects on peak-detection
==================================================
The central point of failure in the CCG workflow is the fact that true
peaks are not picked up when they need to be. Instead often irrelevant peaks
that are higher are picked up. 

Knowing the position of the bats I could use a kind of 'pre-emphasis' to eliminate
regions where TDE peaks are not possible. Here I will check to see if this 
TDE pre-emphasis results in any kind of improvement in the peak-detection of 
the pydatemm package as it stands. 

Created on Thu Aug  3 17:36:34 2023

@author: theja
"""
import numpy as np 
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt 
import os
from pydatemm import timediffestim as tdestim
from itertools import combinations
from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d
import sys
sys.path.append('../')
from source_traj_aligner import calculate_toa_channels

#%% Choose a chunk of audio where we know the 'correct' bat calls are there
datafolder = '../multibat_stresstests/nbat8/'
arraygeomfile = os.path.join(datafolder,'mic_xyz_multibatsim.csv')
micxyz = pd.read_csv(arraygeomfile).loc[:,'x':].to_numpy()
audiofile = os.path.join(datafolder,'8-bats_trajectory_simulation_1-order-reflections.wav')
fs = sf.info(audiofile).samplerate


callingfile = os.path.join(datafolder,'multibatsim_xyz_calling.csv')
df = pd.read_csv(callingfile)
call_data = df[df['emission_point']].sort_values('t')
call_data = call_data.rename({'batnum':'batid'}, axis='columns')
audio_tmax = 0.02
first_fewms = call_data[call_data['t']<audio_tmax].reset_index(drop=True)

#%% Upsample the known xyz positions of bats over time
def upsample_xyz_to_ms_resolution(traj_df, final_res=1e-3):
    bybat = traj_df.groupby('batid')
    all_bats_interp = []
    for batid, subdf in bybat:
        tmin,tmax = np.percentile(subdf['t'], [0,100])
        t_highres = np.arange(tmin,tmax+final_res,final_res)
        t_highres = t_highres[np.logical_and(t_highres>=tmin, t_highres<=tmax)]
        fitted_fn = {axis: interp1d(subdf['t'], subdf[axis])for axis in ['x','y','z']}
        interp_data = pd.DataFrame(data=[], columns=['batid','x','y','z','t'])
        interp_data['t'] = t_highres
        interp_data['batid'] = batid
        for each in ['x','y','z']:
            interp_data[each] = fitted_fn[each](t_highres)
        all_bats_interp.append(interp_data)
    return pd.concat(all_bats_interp, axis=0).reset_index(drop=True)

highres_flightdata = upsample_xyz_to_ms_resolution(call_data,1e-3)

def get_chpair_tdes_from_sources(call_pointsdf, highres_flighttraj, array_geom):
    '''
    Generates all expected TDEs from sources given a high-res flight trajectory 
    and call-point

    Parameters
    ----------
    call_pointsdf : pd.DataFrame
        With columns x,y,z,t,batid
    highres_flighttraj: pd.DataFrame
        With columns x,y,z,t,batid - and with high temporal resolution preferably
    array_geom : (Mmics,3) np.array
        xyz coordinates of the array
    
    Returns 
    -------
    expected_tdoas : dict
        Dictionary with channel pair as a tuple and np.array with expected time-delays
        in seconds. 
    '''
    nchannels = array_geom.shape[0]
    unique_pairs = list(combinations(range(nchannels), 2))
    unique_pairs = list(map(lambda X: sorted(X, reverse=True), unique_pairs))
    expected_tdoa = {tuple(chpair) : [] for chpair in unique_pairs}
    for idx, row in call_pointsdf.iterrows():
        toa_set = calculate_toa_channels(row['t'], highres_flighttraj, row['batid'], array_geom).flatten()
        for chpair, tdoas in expected_tdoa.items():
            chb,cha = chpair
            tde = toa_set[chb]-toa_set[cha] 
            tdoas.append(tde)
    return expected_tdoa

miaowmiaow = get_chpair_tdes_from_sources(first_fewms, highres_flightdata, micxyz )
#%%
# Generate the 'observed' TDOA peaks
first_fewms.loc[:,'toa_max'] = np.tile(np.nan,first_fewms.shape[0])
for idx, row in first_fewms.iterrows():
    toaset = calculate_toa_channels(row['t'], highres_flightdata, row['batid'],
                                    micxyz)
    first_fewms['toa_max'] = toaset.max()


max_time = np.round(first_fewms['toa_max'].max(),3)+6e-3
fs = sf.info(audiofile).samplerate
audio, fs = sf.read(audiofile, stop=int(fs*max_time))

kwargs = {}
kwargs['fs'] = fs
kwargs['min_peak_diff'] = 10e-6
kwargs['min_height'] = 0.01
kwargs['K'] = 10
kwargs['array_geom'] = micxyz

multich_crosscor = tdestim.generate_multich_crosscorr(audio)
multich_tdoa = tdestim.get_multich_tdoas(multich_crosscor, **kwargs)
multich_tdoa = tdestim.geometrically_valid(multich_tdoa, **kwargs)
top_K_tdoas = tdestim.get_topK_peaks(multich_tdoa, **kwargs)


#%% 
# What does the pre-emphasised cross-corr look like. 
# Let's create a bounding box that is +/- 2 ms off (so 4 ms wide)
# But here we need to choose bounding boxes based on ALL the 8 bats' positions
# at a given time-window. 
highres_tillmaxtime = highres_flightdata[highres_flightdata['t']<=max_time].sort_values('batid')

box_halfwidth = int(fs*1e-3)

all_expected_tdoas = {}

# generate expected TDOA peaks for all bats at the midpoint of the flight window
# (not only the calling bats!)
midpoint_data = pd.DataFrame(data=[], columns=call_data.columns)
all_midpoint_data = []
for i,(batid, subdf) in enumerate(highres_tillmaxtime.groupby('batid')):
    midpoint_data.loc[i,'x':'t']= np.median(subdf.loc[:,'x':'t'], axis=0)
    midpoint_data.loc[i,'batid'] = batid

allbats_tdes = get_chpair_tdes_from_sources(midpoint_data, highres_flightdata,micxyz)
allbats_tde_samples = {}
for chpair, tdes in allbats_tdes.items():
    allbats_tde_samples[chpair] = np.array(np.array(tdes)*fs + multich_crosscor[chpair].size*0.5,  dtype=np.int64).flatten()
allbats_tde_windows = {}
for chpair, tdes in allbats_tde_samples.items():
    allbats_tde_windows[chpair] = np.zeros((2,allbats_tde_samples[chpair].size),dtype=np.int64)
    allbats_tde_windows[chpair][0,:] = np.array(allbats_tde_samples[chpair] - box_halfwidth, dtype=np.int64).flatten()
    allbats_tde_windows[chpair][1,:] = np.array(allbats_tde_samples[chpair] + box_halfwidth, dtype=np.int64).flatten()

# Get an estimate of the minimum inter-peak interval to expect
chwise_min_interpeakdist = []
for chpair, tdes in allbats_tde_samples.items():
    inter_peak_dists = list(map(lambda X: abs(X[0]-X[1]),combinations(tdes,2)))
    min_inter_peakdist = min(inter_peak_dists)
    chwise_min_interpeakdist.append(min_inter_peakdist)
    if min_inter_peakdist==1:
        print(chpair, tdes)

informed_min_peakdist = np.percentile(chwise_min_interpeakdist, 10)
#%% And now pre-empasise only those sections 
preemp_multich_crosscor = {}
for chpair, cc in multich_crosscor.items():
    cc_preemp = np.zeros(cc.size)
    tde_regions = allbats_tde_windows[chpair].T
    for region in tde_regions:
        cc_preemp[region[0]:region[1]] = cc[region[0]:region[1]]
    preemp_multich_crosscor[chpair] = cc_preemp

#chpair = (3,1)
plt.figure()
plt.plot(multich_crosscor[chpair],'g')
plt.plot(preemp_multich_crosscor[chpair])

#%%
# Pick peaks from pre-emphasised cross-corr
kwargs['min_peak_diff'] = (informed_min_peakdist*0.9)/fs
preemph_multich_tdoa = tdestim.get_multich_tdoas(preemp_multich_crosscor, **kwargs)
preemph_top_K_tdoas = tdestim.get_topK_peaks(preemph_multich_tdoa, **kwargs)

#%%
chpair = (7,2)

plt.figure()
plt.plot(multich_crosscor[chpair])
inds = list(map(lambda X: X[0], top_K_tdoas[chpair]))
inds_preemph = list(map(lambda X: X[0], preemph_top_K_tdoas[chpair]))
plt.plot(inds, multich_crosscor[chpair][inds], 'r*',markersize=10, label='naive')
plt.plot(inds_preemph, preemp_multich_crosscor[chpair][inds_preemph]+2e-2,
         'k*',markersize=10, label='pre-emph')
plt.plot(preemp_multich_crosscor[chpair])
true_inds  = true_tdes_bychpair[chpair]
plt.plot(true_inds, multich_crosscor[chpair][true_inds]+0.03, 'g*',
                         markersize=10, label='true')
plt.xlim(min(inds)-10, max(inds)+10)
plt.vlines(multich_crosscor[chpair].size*0.5, 0,multich_crosscor[chpair].max(),'k')
plt.legend()
# Now run the raw peak detection





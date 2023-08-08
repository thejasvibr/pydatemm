# -*- coding: utf-8 -*-
"""
Pre-emphasis on CCG localisation
================================

Step 1: Create the pre-emphasis regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
I realised after seeing the actual time-delays, the time-delays can vary a lot
from point to point in the flight trajectory. Therefore, using just one TDE estimate
to pre-emphasise will therefore fail. Here I will use the total TDE min-max range
for a channel pair from all bats in the data to pre-emphasise the cross-correlation. 

Step 2: Effect on localisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""

import argparse
import glob
import natsort
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
from source_traj_aligner import calculate_toa_channels, generate_proximity_profile
from source_traj_aligner import get_close_points
from pydatemm.timediffestim import generate_multich_crosscorr, geometrically_valid
from pydatemm.timediffestim import get_multich_tdoas, get_topK_peaks
from pre_emph_utils import *
from dataclasses import dataclass
import pydatemm.graph_manip as gramanip
from pydatemm.source_generation import localiser_sounds_v3, cpp_make_array_geom
import pydatemm.localiser as lo
try:
    import cppyy as cpy
except:
    pass
from pydatemm.__main__ import conv_to_numpy
import tqdm

# parser = argparse.ArgumentParser()
# parser.add_argument('-tstart', dest='tstart', type=lambda X: float(X))
# parser.add_argument('-tstop', dest='tstop', type=lambda X: float(X))
# parser.add_argument('-runname', dest='runname', type=str, default=None)
# parser.add_argument('-minpeakdist', dest='minpeakdist', type=float, default=50e-6)
# args = parser.parse_args()

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

args = dataclass()

winstarts = np.arange(0,200,5e-3)
winstops = winstarts + 10e-3

for (tsta, tsto) in zip(winstarts, winstops):
    args.tstart = tsta
    args.tstop = tsto
    args.minpeakdist = 50e-6 # seconds
    
    
    
    
    #%%
    vsound = 343.0 #m/s
    ideal_windowsize = np.round(distmat(micxyz, micxyz).max()/vsound,3)
    approx_calldurn = 6e-3 
    # tstart, tstop = 115e-3, 125e-3
    tstart = args.tstart
    tstop = args.tstop
    tstop += approx_calldurn
    
    print('WHOO', tstart, tstop)
    
    audio_snip, fs = sf.read(audiofile, start=int(fs*tstart), stop=int(fs*tstop))
    
    
    # Get cross-corr
    multich_crosscor = generate_multich_crosscorr(audio_snip)
    #%%
    # Take only flight traj within the tstart- time-of-flight and tstop
    video_tstart = np.max([tstart, tstart-0.02])
    flighttraj_window = upsampled_trajectories[upsampled_trajectories['t'].between(video_tstart, tstop)]
    
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
    kwargs['min_peak_diff'] = args.minpeakdist
    kwargs['array_geom'] = micxyz
    kwargs['K'] = 20
    
    # with pre-emphasis
    preemph_multichcrosscor = {}
    for chpair, crosscor in multich_crosscor.items():
        profile = np.zeros(crosscor.size)
        for ii,each in enumerate(overall_minmax_tdoa_chwise[chpair]):
            tdoa_min, tdoa_max = np.int64(each+crosscor.size*0.5)
            #print(ii, tdoa_min, tdoa_max)
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
    # Which calls are expected to be there in the audio snippet
    
    
    actual_calls = call_points[call_points['t']<tstop].loc[:,'x':]
    
    # only keep the calls that will arrive within tstop
    toa_minmax = np.zeros((actual_calls.shape[0],2))
    for i, (idx, row) in enumerate(actual_calls.iterrows()):
        toas = calculate_toa_channels(row['t'], flighttraj_window, row['batid'], micxyz)
        toamintoamax = np.percentile(toas, [0,100])
        toa_minmax[i,:] = toamintoamax
    actual_calls.loc[:,'toa_min'] = toa_minmax[:,0]
    actual_calls.loc[:,'toa_max'] = toa_minmax[:,1]
    # get actual calls that are within the audio window 
    atleast_tstop = actual_calls['toa_min']>=tstart
    upto_tstop = actual_calls['toa_max']<=tstop
    within_timelimits = np.logical_and(atleast_tstop, upto_tstop)
    
    actual_calls = actual_calls[within_timelimits]
    
    
    #%%
    # Quantifying localisation error pre/post pre-emphasis
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Here I will run the CCG localisation algorithm and check the effect it has on 
    # how accurately the sources are detected with and without TDE pre-emphasis. 
    
    
    kwargs['num_cores'] = kwargs.get('num_cores', 2)
    kwargs['initial_vertex'] = kwargs.get('initial_vertex',0)
    kwargs['nchannels'] = audio_snip.shape[1]
    
    cfls_from_tdes = gramanip.make_consistent_fls_cpp(preemph_tdes, **kwargs)
    cfls_from_tdes_raw = gramanip.make_consistent_fls_cpp(normal_tdes, **kwargs)
    run_types = ['preemph', 'raw']
    
    for cfl_input, run_type  in zip([cfls_from_tdes, cfls_from_tdes_raw], run_types):
        print(f'Num CFLS found: {len(cfl_input)}')
        if len(cfl_input)>0:
            ccg_matrix = cpy.gbl.make_ccg_matrix(cfl_input)
            solns_cpp = lo.CCG_solutions_cpp(ccg_matrix)
            ag = cpp_make_array_geom(**kwargs)
            output = localiser_sounds_v3(kwargs['num_cores'], ag, solns_cpp, cfl_input)
            if len(output.sources)==0:
                pass
            else:
                #
                # Check if there are any points within 30 cm of the flight emission point
                posns = conv_to_numpy(output.sources)
                no_999 = np.logical_and(posns[:,0]!=-999, posns[:,1]!=-999)
                posns_filt = posns[no_999,:]
                posns_xyz = posns_filt[:,:3]
                # save to dataframe 
                df_sources = pd.DataFrame(data=posns_filt, columns=['x','y','z','tdoares'])
                df_sources['tstart'] = args.tstart
                df_sources['tstop'] = args.tstop
                csv_filename = f'{run_type}_{tstart}-{tstop}s_sources.csv'
                df_sources.to_csv(csv_filename)


#%%
# And now let's check the accuracy of tracking in 8 bats with and without pre-emphasis.

preemph_csv = natsort.realsorted(glob.glob('pree*.csv'))
all_sources = pd.concat([pd.read_csv(each) for each in preemph_csv]).reset_index(drop=True)
preemph_distmat = distmat(all_sources.loc[:,'x':'z'], upsampled_trajectories.loc[:,'x':'z'])
preemph_sources_nearish = all_sources[preemph_distmat[:,0]<=2].loc[:,['x','y','z','tdoares','tstart','tstop']]
preemph_sources_nearish = preemph_sources_nearish.to_numpy()


raw_csv = natsort.realsorted(glob.glob('raw*.csv'))
all_sources = pd.concat([pd.read_csv(each) for each in raw_csv]).reset_index(drop=True)
raw_distmat = distmat(all_sources.loc[:,'x':'z'], upsampled_trajectories.loc[:,'x':'z'])
raw_sources_nearish = all_sources[raw_distmat[:,0]<=2].loc[:,['x','y','z','tdoares','tstart','tstop']]
raw_sources_nearish = raw_sources_nearish.to_numpy()




coarse_threshold = 1
fine_threshold = 0.3
array_geom = micxyz.copy()
topx = 10

proxprofiles_bybatid = {}

for batid  in tqdm.tqdm(upsampled_trajectories['batid'].unique()):
   
    batdf = upsampled_trajectories.groupby('batid').get_group(batid)
    proxprofiles_bybatid[batid] = generate_proximity_profile(batid, batdf, preemph_sources_nearish, coarse_threshold,
                                              fine_threshold, array_geom, vsound, topx )


rawproxprofiles_bybatid = {}

for batid  in tqdm.tqdm(upsampled_trajectories['batid'].unique()):
   
    batdf = upsampled_trajectories.groupby('batid').get_group(batid)
    rawproxprofiles_bybatid[batid] = generate_proximity_profile(batid, batdf, raw_sources_nearish, coarse_threshold,
                                              fine_threshold, array_geom, vsound, topx )

#%%    
plt.figure()
for batid, profile in proxprofiles_bybatid.items():
    plt.plot(profile)
plt.vlines(call_points['t']*1e3,0,5,'k')

#%%
plt.figure()
for batid, profile in rawproxprofiles_bybatid.items():
    plt.plot(profile)
plt.vlines(call_points['t']*1e3,0,5,'k')
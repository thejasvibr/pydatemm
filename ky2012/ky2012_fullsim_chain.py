# -*- coding: utf-8 -*-
"""
Kreissig-Yang CombineAll simulated data with full chain
=======================================================

Created on Wed Aug  3 10:28:28 2022

@author: theja
"""
from synthetic_data_generation import * 
from pydatemm.tdoa_quality import residual_tdoa_error
from pydatemm.timediffestim import generate_multich_autocorr, generate_multich_crosscorr
from pydatemm.timediffestim import geometrically_valid, get_multich_tdoas
from pydatemm.timediffestim import get_multich_aa_tdes
from pydatemm.raster_matching import multichannel_raster_matcher
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyroomacoustics as pra
import scipy.signal as signal 
import time 
from pydatemm.localisation_mpr2003 import mellen_pachter_raquet_2003
from investigating_peakdetection_gccflavours import multich_expected_peaks
from copy import deepcopy
#np.random.seed(82319)
# %load_ext line_profiler
#%% Generate simulated audio
array_geom = pd.read_csv('../pydatemm/tests/scheuing-yang-2008_micpositions.csv').to_numpy()
# from the pra docs
room_dim = [9, 7.5, 3.5]  # meters
fs = 192000
ref_order = 1

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
batcall *= 0.5

num_sources = int(np.random.choice(range(5,7),1)) # or overruled by the lines below.
random = False

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

import soundfile as sf
sf.write(f'simaudio_reflection-order_{ref_order}.wav', sim_audio, samplerate=fs)

mic2sources = [mic2source(each, array_geom) for each in sources]    
delta_tdes = [np.zeros((nchannels, nchannels)) for each in range(len(mic2sources))]

for i,j in product(range(nchannels), range(nchannels)):
    for source_num, each in enumerate(delta_tdes):
        each[i,j] = mic2sources[source_num][i]-mic2sources[source_num][j] 
        each[i,j] /= vsound

#%%
paper_twrm = 16/fs
paper_twtm = 16/fs
kwargs = {'twrm': paper_twrm,
          'twtm': paper_twtm,
          'nchannels':nchannels,
          'fs':fs,
          'array_geom':array_geom,
          'pctile_thresh': 95,
          'use_gcc':True,
          'gcc_variant':'phat', 
          'min_peak_diff':0.35e-4, 
          'vsound' : 343.0, 
          'no_neg':False}
#%%
# Estimate inter-channel TDES
multich_cc = generate_multich_crosscorr(sim_audio, **kwargs )
#kwargs['use_gcc'] = False
#multich_ac = generate_multich_autocorr(sim_audio, **kwargs)
#%%
#multiaa = get_multich_aa_tdes(multich_ac, **kwargs) 
cc_peaks = get_multich_tdoas(multich_cc, **kwargs)
# valid_tdoas = multichannel_raster_matcher(cc_peaks, multiaa,
#                                        **kwargs)
valid_tdoas = deepcopy(cc_peaks)
#%%
# choose only K=5 (top 5)
K = 30
top_K_tdes = {}
for ch_pair, tdes in valid_tdoas.items():
    descending_quality = sorted(tdes, key=lambda X: X[-1], reverse=True)
    top_K_tdes[ch_pair] = []
    for i in range(K):
        try:
            top_K_tdes[ch_pair].append(descending_quality[i])
        except:
            pass

#%% Here let's check what the expected TDEs are for a given source and array_geom


# and check what the max error is across channels:
edges = map(lambda X: sorted(X, reverse=True), combinations(range(sim_audio.shape[1]),2))
edges = list(map(lambda X: str(tuple(X)), edges))
residual_chpairs = pd.DataFrame(data=[], index=range(len(sources)), columns=['source_no']+edges)


for i,s in enumerate(sources):
    exp_tdes_multich = multich_expected_peaks(sim_audio, [s], array_geom, fs=192000)
    residual_chpairs.loc[i,'source_no'] = i
    for ch_pair, predicted_tde in exp_tdes_multich.items():
        samples = list(map(lambda X: X[0], top_K_tdes[ch_pair]))
        # residual
        residual = np.min(np.abs(np.array(samples)-predicted_tde))
        residual_chpairs.loc[i,str(ch_pair)] = residual

# generate an overall report of fit - look at the mean residual:
residual_chpairs['max_resid'] = residual_chpairs.loc[:,'(1, 0)':'(7, 6)'].apply(np.max,1)
residual_chpairs['sum_resid'] = residual_chpairs.loc[:,'(1, 0)':'(7, 6)'].apply(np.sum,1)
print(residual_chpairs['max_resid'])

#%% create cFLs from TDES
# First get all Fundamental Loops

if __name__ == '__main__':

    print('making the cfls...')
    cfls_from_tdes = make_consistent_fls(top_K_tdes, nchannels,
                                         max_loop_residual=0.15e-4)
    cfls_from_tdes = list(set(cfls_from_tdes))
    all_fls = make_fundamental_loops(nchannels)
    cfls_by_fl = {}
    for fl in all_fls:
        cfls_by_fl[fl] = []
    for i,each in enumerate(cfls_from_tdes):
        for fl in all_fls:
            if set(each.nodes) == set(fl):
                cfls_by_fl[fl].append(i)
    # output = []
    # for fl, cfl_idxs in cfls_by_fl.items():
    #     idx = cfl_idxs[21]
    #     output.append(cfls_from_tdes[idx])
    #%%
    output = cfls_from_tdes[:]
    print(f'# of cfls in list: {len(output)}, starting to make CCG')
    start = time.perf_counter()
    if len(output) < 500:
        ccg_pll = make_ccg_matrix(output)
        #stop_time_normal = time.perf_counter()
    else:
        ccg_pll = make_ccg_pll(output)
        #stop_time_pll = time.perf_counter()
#        assert np.all(ccg_pll == ccg_matrix)
    print('done making the cfls...')
    #print(f'Normal run time: {stop_time_normal-start}, Pll run time: {stop_time_pll-stop_time_normal}')
    #%% generate CCG from cFLs
    # ccg_matrix = make_ccg_matrix(cfls_from_tdes)
    print('..making the ccg matrix')
    # smaller_cflset = cfls_from_tdes[::]
    # print(f'cflsets: {len(smaller_cflset)}')
    # smaller_ccg = make_ccg_matrix(smaller_cflset) 
    np.savetxt('flatA.txt',ccg_pll.flatten(),delimiter=',',fmt='%i')
    # print('..done making the ccg matrix')
    # #%%
    # Call the ui_combineall exe implemented in Cpp
    import os, platform
    if platform.system() == 'Windows':
        os.system('ui_combineall.exe flatA.txt')
    elif platform.system()=='Linux':
        os.system('./ui_combineall flatA.txt')
    
    print('Loading the solution txt file')
    # #%%
    # Load the 'jagged' csv file 
    output_file = 'combineall_solutions.csv'
    import csv
    comp_cfls = []
    with open(output_file, 'r') as ff:
        csvfile = csv.reader(ff, delimiter=',')
        for lines in csvfile:
            fmted_lines = [int(each) for each in lines if not each=='']
            comp_cfls.append(fmted_lines)
    print('...loading done...')
    #%% Parallelise the localisation code. 
    def localise_sounds(compatible_solutions, all_cfls, **kwargs):
        localised_geq4_out = pd.DataFrame(index=range(len(compatible_solutions)), 
                                     data=[], columns=['x','y','z','tdoa_resid_s','cfl_inds'])
        localised_4ch_out = pd.DataFrame(index=range(len(compatible_solutions)), 
                                     data=[], columns=['x','y','z','tdoa_resid_s','cfl_inds'])
        ii = 0
        for i, compat_cfl in enumerate(compatible_solutions):
            #print(f'i: {i}')
            if len(compat_cfl)>=2:
                source_graph = combine_compatible_triples([all_cfls[j] for j in compat_cfl])
                source_tde = nx.to_numpy_array(source_graph, weight='tde')
                d = source_tde[1:,0]*kwargs['vsound']
                channels = list(source_graph.nodes)
                #print('channels', channels)
                source_xyz = np.array([np.nan])
                if len(channels)>4:
                    try:
                        source_xyz = spiesberger_wahlberg_solution(kwargs['array_geom'][channels,:],
                                                               d)
                    except:
                        pass
                    if not np.sum(np.isnan(source_xyz))>0:
                        localised_geq4_out.loc[i,'x':'z'] = source_xyz
                        localised_geq4_out.loc[i,'tdoa_resid_s'] = residual_tdoa_error(source_graph,
                                                                            source_xyz,
                                                                            array_geom[channels,:])
                        localised_geq4_out.loc[i,'cfl_inds'] = str(compat_cfl)
                    
                elif len(channels)==4:
                    source_xyz  = mellen_pachter_raquet_2003(array_geom[channels,:], d)
                    if not np.sum(np.isnan(source_xyz))>0:
                        if np.logical_or(source_xyz.shape[0]==1,source_xyz.shape[0]==3):
                            localised_4ch_out.loc[ii,'x':'z'] = source_xyz
                            localised_4ch_out.loc[ii,'tdoa_resid_s'] = residual_tdoa_error(source_graph,
                                                                                source_xyz,
                                                                                array_geom[channels,:])
                            ii += 1

                        elif source_xyz.shape[0]==2:
                            for ss in range(2):
                                localised_4ch_out.loc[ii,'x':'z'] = source_xyz[ss,:]
                                localised_4ch_out.loc[ii,'tdoa_resid_s'] = residual_tdoa_error(source_graph,
                                                                                    source_xyz[ss,:],
                                                                                    array_geom[channels,:])
                                ii += 1                    
                        localised_4ch_out.loc[ii,'cfl_inds'] = str(compat_cfl)
            else:
                pass
        localised_combined = pd.concat([localised_geq4_out, localised_4ch_out]).reset_index(drop=True).dropna()
        return localised_combined
    #%%
    #%load_ext line_profiler
    #from pydatemm.localisation import choose_SW_valid_solution, make_rangediff_mat
    #%lprun -f localise_sounds localise_sounds(split_solns[0][::100], cfls_from_tdes, **kwargs)
    #%% 
    import warnings
    warnings.filterwarnings('ignore')
    print('localising solutions...')
    import tqdm
    print(f'...length of all solutions...{len(comp_cfls)}')
    parts = joblib.cpu_count()
    split_solns = [comp_cfls[i::parts] for i in range(parts)]
    out_dfs = Parallel(n_jobs=-1)(delayed(localise_sounds)(comp_subset, cfls_from_tdes, **kwargs) for comp_subset in split_solns)
    print('...Done localising solutions...')
    #%%
    # localise_sounds(split_solns[0], cfls_from_tdes, **kwargs)
    #%%
    print('...subsetting sensible localisations')
    all_locs = pd.concat(out_dfs).reset_index(drop=True).dropna()
    good_locs = all_locs[all_locs['tdoa_resid_s']<1e-4].reset_index(drop=True)
    valid_rows = np.logical_and(np.abs(good_locs['x'])<10, np.abs(good_locs['y'])<10,
                                np.abs(good_locs['z'])<10)
    sensible_locs = good_locs[valid_rows].reset_index(drop=True)
    print('...calculating error to known sounds')
    from scipy import spatial
    for i, ss in tqdm.tqdm(enumerate(sources)):
        sensible_locs.loc[:,f's_{i}'] = sensible_locs.apply(lambda X: spatial.distance.euclidean(X['x':'z'], ss),1)
    #%% Are the best candidates in here? 
    for k in range(len(sources)):
        idx = sensible_locs.loc[:,f's_{k}'].argmin()
        print('\n',np.around(sources[k],2), np.around(sensible_locs.loc[idx,'x':'z'].tolist(),2),
              np.around(sensible_locs.loc[idx,f's_{k}'],3), sensible_locs.loc[idx,'cfl_inds'])
    print('Done')

    #%% 
    # gr = combine_compatible_triples([cfls_from_tdes[each] for each in [62,136,511]])
    # plt.figure()
    # plot_graph_w_labels(gr, plt.gca())
    from pydatemm.timediffestim import max_interch_delay as maxintch
    exp_tdes_multich = multich_expected_peaks(sim_audio, [sources[3]], array_geom, fs=192000)
    edges = list(map(lambda X: sorted(X, reverse=True), combinations(range(sim_audio.shape[1]),2)))
    pk_lim = 20
    plt.figure()
    a0 = plt.subplot(111)
    for chpair in edges:
        chpair = tuple(chpair)
        exp_peak = exp_tdes_multich[chpair]
        plt.cla()
        a0.plot(multich_cc[chpair])
        
        for pk in top_K_tdes[chpair]:
            plt.plot(pk[0], pk[2],'*')
        a0.scatter(exp_peak, multich_cc[chpair][int(exp_peak)], color='r', s=80)
        max_delay = int(maxintch(chpair, kwargs['array_geom'])*fs)
        minmaxsample =  np.int64(sim_audio.shape[0] + np.array([-max_delay, max_delay]))
        plt.xlim(exp_peak-pk_lim, exp_peak+pk_lim)
        plt.vlines(minmaxsample, 0, np.max(multich_cc[chpair]), 'r')
        plt.title(f' Channel pair: {chpair}')
        plt.pause(5)
    #%% 
    # Source 3 is the problem fix. Is this caused by erroneous TDEs or something else? 
    def index_tde_to_sec(X, audio, fs):
        return (X - audio.shape[0])/fs
    exp_tde_s = {}
    for chpair, tde_inds in exp_tdes_multich.items():
        exp_tde_s[chpair] = index_tde_to_sec(tde_inds, sim_audio, fs)
    
    k = 3
    idx = sensible_locs.loc[:,f's_{k}'].argmin()
    compat_inds = sensible_locs.loc[idx,'cfl_inds']
    compat_inds_int = [int(each) for each in compat_inds[1:-1].split(',')]
    combined_graph = combine_compatible_triples([cfls_from_tdes[each] for each in compat_inds_int])
    mic_nodes = combined_graph.nodes
    mic_array = array_geom[mic_nodes,:]   
    tdemat = nx.to_numpy_array(combined_graph, weight='tde')
    tdemat[:,0]
    
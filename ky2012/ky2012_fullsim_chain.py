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
from sklearn import cluster
import time 
import tqdm
from pydatemm.localisation_mpr2003 import mellen_pachter_raquet_2003
from investigating_peakdetection_gccflavours import multich_expected_peaks
from copy import deepcopy
try:
    import cppyy 
    cppyy.include('./combineall_cpp/ui_combineall.cpp')
    vector_cpp = cppyy.gbl.std.vector
    set_cpp = cppyy.gbl.std.set
except ImportError:
    vector_cpp = cppyy.gbl.std.vector
    set_cpp = cppyy.gbl.std.set
    pass

#np.random.seed(82319)
# %load_ext line_profiler
#%% Generate simulated audio
array_geom = pd.read_csv('../pydatemm/tests/scheuing-yang-2008_micpositions.csv').to_numpy()
# from the pra docs
room_dim = [9, 7.5, 3.5]  # meters
fs = 192000
ref_order = 1

reflection_max_order = ref_order

rt60_tgt = 0.2  # seconds
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
               [8, 6, 0.7],
               [1, 4, 1.0],
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
# choose only the first X s
# durn = 0.03
sim_audio = room.mic_array.signals.T
#if sim_audio.shape[0]>(int(fs*durn)):
#    sim_audio = sim_audio[:int(fs*durn),:]
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

#%% Here let's check what the expected TDEs are for a given source and array_geom


# # and check what the max error is across channels:
# edges = map(lambda X: sorted(X, reverse=True), combinations(range(sim_audio.shape[1]),2))
# edges = list(map(lambda X: str(tuple(X)), edges))
# residual_chpairs = pd.DataFrame(data=[], index=range(len(sources)), columns=['source_no']+edges)


# for i,s in enumerate(sources):
#     exp_tdes_multich = multich_expected_peaks(sim_audio, [s], array_geom, fs=192000)
#     residual_chpairs.loc[i,'source_no'] = i
#     for ch_pair, predicted_tde in exp_tdes_multich.items():
#         samples = list(map(lambda X: X[0], top_K_tdes[ch_pair]))
#         # residual
#         residual = np.min(np.abs(np.array(samples)-predicted_tde))
#         residual_chpairs.loc[i,str(ch_pair)] = residual

# # generate an overall report of fit - look at the mean residual:
# residual_chpairs['max_resid'] = residual_chpairs.loc[:,'(1, 0)':'(7, 6)'].apply(np.max,1)
# residual_chpairs['sum_resid'] = residual_chpairs.loc[:,'(1, 0)':'(7, 6)'].apply(np.sum,1)
# print(residual_chpairs['max_resid'])

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


def generate_candidate_sources(sim_audio, **kwargs):
    multich_cc = generate_multich_crosscorr(sim_audio, **kwargs )
    cc_peaks = get_multich_tdoas(multich_cc, **kwargs)
    valid_tdoas = deepcopy(cc_peaks)
    
    K = kwargs.get('K',5)
    top_K_tdes = {}
    for ch_pair, tdes in valid_tdoas.items():
        descending_quality = sorted(tdes, key=lambda X: X[-1], reverse=True)
        top_K_tdes[ch_pair] = []
        for i in range(K):
            try:
                top_K_tdes[ch_pair].append(descending_quality[i])
            except:
                pass
    print('making the cfls...')
    cfls_from_tdes = make_consistent_fls(top_K_tdes, nchannels,
                                         max_loop_residual=kwargs.get('max_loop_residual',
                                                                      0.5e-4))
    cfls_from_tdes = list(set(cfls_from_tdes))
    all_fls = make_fundamental_loops(nchannels)
    cfls_by_fl = {}
    for fl in all_fls:
        cfls_by_fl[fl] = []
    for i,each in enumerate(cfls_from_tdes):
        for fl in all_fls:
            if set(each.nodes) == set(fl):
                cfls_by_fl[fl].append(i)
    # 
    output = cfls_from_tdes[:]
    print(f'# of cfls in list: {len(output)}, starting to make CCG')
    start = time.perf_counter()
    if len(output) < 500:
        ccg_pll = make_ccg_matrix(output)
    else:
        ccg_pll = make_ccg_pll(output)
    print('done making the cfls...')
    # 
    n_rows = ccg_pll.shape[0]
    ac_cpp = vector_cpp[vector_cpp[int]]([ccg_pll[i,:].tolist() for i in range(n_rows)])
    v_cpp = set_cpp[int](range(n_rows))
    l_cpp = set_cpp[int]([])
    x_cpp = set_cpp[int]([])
    print('Starting cppyy CombineAll run...')
    solns_cpp = cppyy.gbl.combine_all(ac_cpp, v_cpp, l_cpp, x_cpp)
    print('Done with CombineAll run')
    comp_cfls = list([]*len(solns_cpp))
    comp_cfls = [list(each) for each in solns_cpp]
    # 
    print('localising solutions...')
    print(f'...length of all solutions...{len(comp_cfls)}')
    parts = joblib.cpu_count()
    split_solns = [comp_cfls[i::parts] for i in range(parts)]
    out_dfs = Parallel(n_jobs=-1)(delayed(localise_sounds)(comp_subset, cfls_from_tdes, **kwargs) for comp_subset in split_solns)
    print('...Done localising solutions...')
    all_locs = pd.concat(out_dfs).reset_index(drop=True).dropna()
    return all_locs

def refine_candidates_to_room_dims(candidates, max_tdoa_res, room_dims):
    good_locs = candidates[candidates['tdoa_resid_s']<max_tdoa_res].reset_index(drop=True)
    
    # get rid of 'noisy' localisations. 
    np_and = np.logical_and
    valid_rows = np.logical_and(np_and(good_locs['x']<=room_dims[0], good_locs['x']>=0),
                                np_and(good_locs['y']<=room_dims[1], good_locs['y']>=0)
                                )
    valid_rows_w_z = np_and(valid_rows, np_and(good_locs['z']<=room_dims[2], good_locs['z']>=0))
    
    sensible_locs = good_locs.loc[valid_rows_w_z,:].sort_values(['tdoa_resid_s']).reset_index(drop=True)
    return sensible_locs

def dbscan_cluster(candidates, dbscan_eps):
    
    pred_posns = candidates.loc[:,'x':'z'].to_numpy(dtype='float64')
    output = cluster.DBSCAN(eps=dbscan_eps).fit(pred_posns)
    uniq_labels = np.unique(output.labels_)
    labels = output.labels_
    #% get mean positions
    cluster_locns = []
    for each in uniq_labels:
        idx = np.argwhere(each==labels)
        sub_cluster = pred_posns[idx,:]
        cluster_locn = np.mean(sub_cluster,0)
        cluster_varn = np.std(sub_cluster,0)
        cluster_locns.append(np.concatenate((cluster_locn, cluster_varn)))
    #print(np.around(cluster_locns,2))
    return cluster_locns



if __name__ == '__main__':
    kwargs['max_loop_residual'] = 0.25e-4
    kwargs['K'] = 5
    dd = 0.001 + np.max(distance_matrix(array_geom, array_geom))/343  
    dd_samples = int(kwargs['fs']*dd)

    start_samples = np.arange(0,sim_audio.shape[0], 192)
    end_samples = start_samples+dd_samples
    max_inds = 50    
    all_candidates = []
    for (st, en) in zip(start_samples[:max_inds], end_samples[:max_inds]):
        candidates = generate_candidate_sources(sim_audio[st:en,:], **kwargs)
        refined = refine_candidates_to_room_dims(candidates, 0.5e-4, room_dim)
        all_candidates.append(refined)

    #%%
    clustered_positions = []
    for each in all_candidates:
        try:
            clustered_positions.append(dbscan_cluster(each, 0.5))
        except ValueError:
            clustered_positions.append([])
    #%%
    print('...subsetting sensible localisations')
    
    
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
    from pydatemm.timediffestim import max_interch_delay as maxintch
    exp_tdes_multich = multich_expected_peaks(sim_audio, [sources[3]],
                                              array_geom, fs=192000)
    #%% 
    # Source 3 is the problem fix. Is this caused by erroneous TDEs or something else? 
    def index_tde_to_sec(X, audio, fs):
        return (X - audio.shape[0])/fs
    exp_tde_s = {}
    for chpair, tde_inds in exp_tdes_multich.items():
        exp_tde_s[chpair] = index_tde_to_sec(tde_inds, sim_audio, fs)
    k = 0
    idx = sensible_locs.loc[:,f's_{k}'].argmin()
    compat_inds = sensible_locs.loc[idx,'cfl_inds']
    compat_inds_int = [int(each) for each in compat_inds[1:-1].split(',')]
    combined_graph = combine_compatible_triples([cfls_from_tdes[each] for each in compat_inds_int])
    mic_nodes = combined_graph.nodes
    mic_array = array_geom[mic_nodes,:]   
    tdemat = nx.to_numpy_array(combined_graph, weight='tde')
    tdemat[:,0]
    #%%
    
    #%% 
    
        
    #%%
    
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(pred_posns[:,0], pred_posns[:,1], '*', alpha=0.5)
    # for s in sources:
    #     plt.scatter(s[0], s[1], color='r', s=100)
    
    # for each in cluster_locns:
    #     every = each[0]
    #     plt.plot(every[0], every[1],'g*')
    # plt.subplot(212)
    # plt.plot(pred_posns[:,1], pred_posns[:,2], '*')
    # for s in sources:
    #     plt.plot(s[1], s[2], 'r*')
    
    # for each in cluster_locns:
    #     every = each[0]
    #     plt.plot(every[1], every[2],'g*')
    # #%% 
    # from mpl_toolkits import mplot3d
    # plt.figure()
    # a0 = plt.subplot(111, projection='3d')
    # st = 200
    # plt.plot(pred_posns[:st,0], pred_posns[:st,1], pred_posns[:st,2], '*')
    # for each in sources:
    #     plt.plot(*[xx for xx in each], 'r*')
    # a0.set_xlim(0,room_dim[0])
    # a0.set_ylim(0,room_dim[1])
    # a0.set_zlim(0, room_dim[2])
    # # plot array
    # for every in array_geom:
    #     plt.plot(*[yy for yy in every], 'g^')
    
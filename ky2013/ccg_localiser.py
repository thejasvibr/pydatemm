#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCG Localiser
=============
Functions to localise sounds from CCG solutions. 

Created on Thu Sep 15 16:38:31 2022

@author: thejasvi
"""
from itertools import chain
import networkx as nx
import numpy as np 
from scipy.spatial import distance
euclidean = distance.euclidean
import pandas as pd
from build_ccg import *
from sklearn import cluster
from pydatemm.timediffestim import generate_multich_crosscorr, get_multich_tdoas
from pydatemm.localisation import spiesberger_wahlberg_solution, choose_SW_valid_solution_tau51
from pydatemm.localisation_mpr2003 import mellen_pachter_raquet_2003
from pydatemm.tdoa_quality import residual_tdoa_error_nongraph
from pydatemm.tdoa_quality import residual_tdoa_error
from joblib import wrap_non_picklable_objects
from joblib import Parallel, delayed
import time

import os 
try:
    os.environ['EXTRA_CLING_ARGS'] = '-fopenmp -O2'
    import cppyy 
    cppyy.load_library('C:\\Users\\theja\\anaconda3\\Library\\bin\\libiomp5md.dll')
    #cppyy.load_library("/home/thejasvi/anaconda3/lib/libiomp5.so")
    cppyy.add_include_path('./eigen/')
    cppyy.include('./sw2002_vectorbased.h')
    cppyy.include('./combineall_cpp/ui_combineall.cpp')
    vector_cpp = cppyy.gbl.std.vector
    set_cpp = cppyy.gbl.std.set
except ImportError:
    vector_cpp = cppyy.gbl.std.vector
    set_cpp = cppyy.gbl.std.set
    pass

get_nmics = lambda X: int((X.size+1)/4)

def cppyy_sw2002(micntde):
    as_Vxd = cppyy.gbl.sw_matrix_optim(vector_cpp['double'](micntde),
                                       )
    return np.array(as_Vxd, dtype=np.float64)

def pll_cppyy_sw2002(many_micntde, c):
    block_in = vector_cpp[vector_cpp['double']](many_micntde.shape[0])
    for i in range(many_micntde.shape[0]):
        block_in[i] = vector_cpp['double'](many_micntde[i,:])
    block_out = cppyy.gbl.pll_sw_optim(block_in, c)
    return np.array([each for each in block_out])

def row_based_mpr2003(tde_data):
    nmics = get_nmics(tde_data)
    sources = mellen_pachter_raquet_2003(tde_data[:nmics*3].reshape(-1,3), tde_data[-(nmics-1):])
    
    if sources.size ==3:
        output = sources.reshape(1,3)
        residual = np.zeros((1,1))
        residual[0] = residual_tdoa_error_nongraph(tde_data[-(nmics-1):], sources,
                                                tde_data[:nmics*3].reshape(-1,3))
        return np.column_stack((output, residual))
    elif sources.size==6:
        output = np.zeros((sources.shape[0], sources.shape[1]))
        output[:,:3] = sources
        residual = np.zeros((2,1))
        for i in range(2):
            residual[i,:] = residual_tdoa_error_nongraph(tde_data[-(nmics-1):], sources[i,:],
                                                tde_data[:nmics*3].reshape(-1,3))
        return np.column_stack((output, residual))
    else:
        return np.array([])
 
def create_tde_data(compatible_solutions, all_cfls, **kwargs):
    '''
    Wrapper to decide if the serial or parallel version is used. 
    
    See Also
    --------
    chunk_create_tde_data
    pll_create_tde_data
    '''
    if len(compatible_solutions) > 500:
        return pll_create_tde_data(compatible_solutions, all_cfls, **kwargs)
    else:
        return chunk_create_tde_data(compatible_solutions, all_cfls, **kwargs)

def chunk_create_tde_data(compatible_solutions, all_cfls, **kwargs):
    ''' Creates dictionary of 2D numpy arrays with 
    Parameters
    ----------
    compatible_solutions :  list with sub-lists
    all_cfls : list with nx.Graphs
        Containing all the consistent fundamental loops
    Keyword Args
    ------------
    vsound, array_geom
    
    Returns
    -------
    tde_by_channelnum : dict
        Keys are number of channels, entries hold np.arrays of size
        Mrows x (3*Nchannels + Nchannels-1)
    cfl_id : dict
        Keys are channel numbers, entries are cFL index numbers. 
        This dictionary helps with troubleshooting and error tracking.
    '''
    raw_tde_by_channelnum = {}
    cfl_ids = {} # for troubleshooting and error tracing
    for i, compat_cfl in enumerate(compatible_solutions):
        source_graph = combine_compatible_triples([all_cfls[j] for j in compat_cfl])
        source_tde = nx.to_numpy_array(source_graph, weight='tde')
        d = source_tde[1:,0]*kwargs['vsound']
        channels = list(source_graph.nodes)
        numchannels = len(channels)
        tde_data = np.concatenate((kwargs['array_geom'][channels,:].flatten(), d))
        if raw_tde_by_channelnum.get(numchannels) is None:
            raw_tde_by_channelnum[numchannels] = []
            raw_tde_by_channelnum[numchannels].append(tde_data)
            cfl_ids[numchannels] = []
            cfl_ids[numchannels].append(compat_cfl)
        else:
            raw_tde_by_channelnum[numchannels].append(tde_data)
            cfl_ids[numchannels].append(compat_cfl)
    tde_by_channelnum = {}
    for nchannels, tde_data in raw_tde_by_channelnum.items():
        tde_by_channelnum[nchannels] = np.row_stack(tde_data)
    return tde_by_channelnum, cfl_ids

def pll_create_tde_data(solns_cpp, all_cfls, **kwargs):
    parts = os.cpu_count()
    #split data into parts
    split_solns = (solns_cpp[i::parts] for i in range(parts))
    results = Parallel(n_jobs=parts)(delayed(chunk_create_tde_data)(chunk, all_cfls, **kwargs) for chunk in split_solns)
    # join split data into single dictionaries
    all_channel_keys = [list(tdedata_dict.keys()) for (tdedata_dict, _) in results]
    unique_num_channels = set(chain(*all_channel_keys))
    
    channelwise_tdedata = {}
    channelwise_cflid = {}
    for nchannels in unique_num_channels:
        channelwise_tdedata[nchannels] = []
        channelwise_cflid[nchannels] = []
        for (tde_data, cfl_id) in results:
            if tde_data.get(nchannels) is not None:
                channelwise_tdedata[nchannels].append(tde_data[nchannels])
                channelwise_cflid[nchannels].append(cfl_id[nchannels])
        channelwise_tdedata[nchannels] = np.row_stack(channelwise_tdedata[nchannels])
        channelwise_cflid[nchannels] = list(chain(*channelwise_cflid[nchannels]))
    return channelwise_tdedata, channelwise_cflid        

def localise_sounds_v2(compatible_solutions, all_cfls, **kwargs):
    '''
    '''
    tde_data, cfl_ids = create_tde_data(compatible_solutions, all_cfls, **kwargs)
    sources = []
    ncores = os.cpu_count()
    all_sources = []
    all_cfls = []
    for (nchannels, tde_input) in tde_data.items():
       
        if nchannels > 4:
            calc_sources = pll_cppyy_sw2002(tde_input, kwargs['vsound'])
            all_sources.append(calc_sources)
            all_cfls.append(cfl_ids[nchannels])
        elif nchannels == 4:
            fourchannel_cflids= []
            for i in range(tde_input.shape[0]):
                calc_sources = row_based_mpr2003(tde_input[i,:])
                if calc_sources.size==6:
                    all_sources.append(calc_sources[0,:])
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    all_sources.append(calc_sources[1,:])
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                elif calc_sources.size==3:
                    all_sources.append(calc_sources)
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                elif len(calc_sources) == 0:
                    pass
            all_cfls.append(fourchannel_cflids)
        else:
            print(f'{nchannels} channels encountered - Ignoring...')
    return np.row_stack(all_sources), list(chain(*all_cfls))

def generate_candidate_sources_v2(sim_audio, **kwargs):
    multich_cc = generate_multich_crosscorr(sim_audio, **kwargs )
    cc_peaks = get_multich_tdoas(multich_cc, **kwargs)

    K = kwargs.get('K',5)
    top_K_tdes = {}
    for ch_pair, tdes in cc_peaks.items():
        descending_quality = sorted(tdes, key=lambda X: X[-1], reverse=True)
        top_K_tdes[ch_pair] = []
        for i in range(K):
            try:
                top_K_tdes[ch_pair].append(descending_quality[i])
            except:
                pass
    print('making the cfls...')
    cfls_from_tdes = make_consistent_fls(top_K_tdes, **kwargs)
    print(f'len of cfls: {len(cfls_from_tdes)}')
    # put all consistent loops into fundamental loop 'bins'
    all_fls = make_fundamental_loops(kwargs['nchannels'])
    cfls_by_fl = {}
    for fl in all_fls:
        cfls_by_fl[fl] = []

    for i,each in enumerate(cfls_from_tdes):
        for fl in all_fls:
            if set(each.nodes) == set(fl):
                cfls_by_fl[fl].append(i)

    if len(cfls_from_tdes) < 500:
        ccg_matrix = make_ccg_matrix(cfls_from_tdes)
    else:
        ccg_matrix = make_ccg_pll(cfls_from_tdes)

    solns_cpp = CCG_solutions(ccg_matrix)
    sources, cfl_ids = localise_sounds_v2(solns_cpp, cfls_from_tdes, **kwargs)
    return sources, cfl_ids

def generate_candidate_sources(sim_audio, **kwargs):
    multich_cc = generate_multich_crosscorr(sim_audio, **kwargs )
    cc_peaks = get_multich_tdoas(multich_cc, **kwargs)

    K = kwargs.get('K',5)
    top_K_tdes = {}
    for ch_pair, tdes in cc_peaks.items():
        descending_quality = sorted(tdes, key=lambda X: X[-1], reverse=True)
        top_K_tdes[ch_pair] = []
        for i in range(K):
            try:
                top_K_tdes[ch_pair].append(descending_quality[i])
            except:
                pass
    print('making the cfls...')
    cfls_from_tdes = make_consistent_fls(top_K_tdes, **kwargs)
    print(f'len of cfls: {len(cfls_from_tdes)}')
    # put all consistent loops into fundamental loop 'bins'
    all_fls = make_fundamental_loops(kwargs['nchannels'])
    cfls_by_fl = {}
    for fl in all_fls:
        cfls_by_fl[fl] = []

    for i,each in enumerate(cfls_from_tdes):
        for fl in all_fls:
            if set(each.nodes) == set(fl):
                cfls_by_fl[fl].append(i)

    if len(cfls_from_tdes) < 500:
        ccg_matrix = make_ccg_matrix(cfls_from_tdes)
    else:
        ccg_matrix = make_ccg_pll(cfls_from_tdes)

    solns_cpp = CCG_solutions(ccg_matrix)
    parts = joblib.cpu_count()
    split_solns = (solns_cpp[i::parts] for i in range(parts))
    out_dfs = Parallel(n_jobs=1)(delayed(localise_sounds)(comp_subset, cfls_from_tdes, **kwargs) for comp_subset in split_solns)
    all_locs = pd.concat(out_dfs).reset_index(drop=True).dropna()
    return all_locs

def CCG_solutions(ccg_matrix):
    n_rows = ccg_matrix.shape[0]
    ac_cpp = vector_cpp[vector_cpp[int]]([ccg_matrix[i,:].tolist() for i in range(n_rows)])
    v_cpp = set_cpp[int](range(n_rows))
    l_cpp = set_cpp[int]([])
    x_cpp = set_cpp[int]([])
    solns_cpp = cppyy.gbl.combine_all(ac_cpp, v_cpp, l_cpp, x_cpp)
    comp_cfls = list([]*len(solns_cpp))
    comp_cfls = [list(each) for each in solns_cpp]
    return comp_cfls

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

def dbscan_cluster(candidates, dbscan_eps, n_points):
    
    pred_posns = candidates.loc[:,'x':'z'].to_numpy(dtype='float64')
    output = cluster.DBSCAN(eps=dbscan_eps, min_samples=n_points).fit(pred_posns)
    uniq_labels = np.unique(output.labels_)
    labels = output.labels_
    #% get mean positions
    cluster_locns_mean = []
    cluster_locns_std = []
    for each in uniq_labels:
        idx = np.argwhere(each==labels)
        sub_cluster = pred_posns[idx,:]
        cluster_locn = np.mean(sub_cluster,0)
        cluster_varn = np.std(sub_cluster,0)
        cluster_locns_mean.append(cluster_locn)
        cluster_locns_std.append(cluster_varn)
    return cluster_locns_mean, cluster_locns_std

if __name__ == "__main__":
    import soundfile as sf
    from scipy.spatial import distance_matrix

    
    filename = '3-bats_trajectory_simulation_raytracing-2.wav'
    fs = sf.info(filename).samplerate
    array_audio, fs = sf.read(filename, stop=int(0.2*fs))
    array_geom = pd.read_csv('multibat_sim_micarray.csv').to_numpy()[:,1:]
    
    nchannels = array_audio.shape[1]
    kwargs = {'nchannels':nchannels,
              'fs':fs,
              'array_geom':array_geom,
              'pctile_thresh': 95,
              'use_gcc':True,
              'gcc_variant':'phat', 
              'min_peak_diff':0.35e-4, 
              'vsound' : 343.0, 
              'no_neg':False}
    kwargs['max_loop_residual'] = 0.5e-4
    kwargs['K'] = 5
    dd = np.max(distance_matrix(array_geom, array_geom))/343  
    dd_samples = int(kwargs['fs']*dd)
    
    ignorable_start = int(0.01*fs)
    shift_samples = 96
    start_samples = np.arange(ignorable_start,array_audio.shape[0], shift_samples)
    end_samples = start_samples+dd_samples
    max_inds = int(0.2*fs/shift_samples)

    #%%
    import tqdm
    sta = time.perf_counter_ns()/1e9
    for i in tqdm.trange(350):
        audio_chunk = array_audio[start_samples[i]:end_samples[i]]
        aa,jj = generate_candidate_sources_v2(audio_chunk, **kwargs)
    sto = time.perf_counter_ns()/1e9
    print(f'{sto-sta} s time')    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCG Localiser
=============
Functions to localise sounds from CCG solutions. 

Created on Thu Sep 15 16:38:31 2022

@author: thejasvi
"""
import networkx as nx
import numpy as np 
import pandas as pd
from build_ccg import *
from pydatemm.timediffestim import generate_multich_crosscorr, get_multich_tdoas
from pydatemm.localisation import spiesberger_wahlberg_solution, choose_SW_valid_solution_tau51
from pydatemm.localisation_mpr2003 import mellen_pachter_raquet_2003
from pydatemm.tdoa_quality import residual_tdoa_error
try:
    import cppyy 
    cppyy.add_include_path('./eigen/')
    cppyy.include('./localisation.cpp')
    cppyy.include('./combineall_cpp/ui_combineall.cpp')
    MatrixXd = cppyy.gbl.Eigen.MatrixXd
    VectorXd = cppyy.gbl.Eigen.VectorXd
    MatrixXd = cppyy.gbl.Eigen.MatrixXd
    VectorXd = cppyy.gbl.Eigen.VectorXd
    
    vector_cpp = cppyy.gbl.std.vector
    set_cpp = cppyy.gbl.std.set
except ImportError:
    vector_cpp = cppyy.gbl.std.vector
    set_cpp = cppyy.gbl.std.set
    pass

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
                                                                        kwargs['array_geom'][channels,:])
                    localised_geq4_out.loc[i,'cfl_inds'] = str(compat_cfl)
                
            elif len(channels)==4:
                source_xyz  = mellen_pachter_raquet_2003(kwargs['array_geom'][channels,:], d)
                if not np.sum(np.isnan(source_xyz))>0:
                    if np.logical_or(source_xyz.shape[0]==1,source_xyz.shape[0]==3):
                        localised_4ch_out.loc[ii,'x':'z'] = source_xyz
                        localised_4ch_out.loc[ii,'tdoa_resid_s'] = residual_tdoa_error(source_graph,
                                                                            source_xyz,
                                                                            kwargs['array_geom'][channels,:])
                        ii += 1

                    elif source_xyz.shape[0]==2:
                        for ss in range(2):
                            localised_4ch_out.loc[ii,'x':'z'] = source_xyz[ss,:]
                            localised_4ch_out.loc[ii,'tdoa_resid_s'] = residual_tdoa_error(source_graph,
                                                                                source_xyz[ss,:],
                                                                                kwargs['array_geom'][channels,:])
                            ii += 1                    
                    localised_4ch_out.loc[ii,'cfl_inds'] = str(compat_cfl)
        else:
            pass
    localised_combined = pd.concat([localised_geq4_out, localised_4ch_out]).reset_index(drop=True).dropna()
    return localised_combined

def cpp_spiesberger_wahlberg(array_geom_np, d_np, **kwargs):
    mic0 = np.zeros((1,3))
    if np.sum(array_geom_np[0,:]) != 0:
        mic0 = array_geom_np[0,:]
        array_geom_np -= array_geom_np[0,:]
        
    rows, cols = array_geom_np.shape
    array_geom = MatrixXd(rows, cols)
    for i in range(rows):
        for j in range(cols):
            array_geom[i,j] = array_geom_np[i,j]
    
    d = VectorXd(rows-1)
    for i, value in enumerate(d_np):
        d[i] = value
    

    solutions = cppyy.gbl.spiesberger_wahlberg_solution(array_geom, d, 343.0)
    s = list(map(lambda X: np.array(list(X)), solutions))
    valid_solution = choose_SW_valid_solution_tau51(s, array_geometry+mic0, d_np,
                                                                      **kwargs)
    return valid_solution
    

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
    out_dfs = Parallel(n_jobs=-1)(delayed(localise_sounds)(comp_subset, cfls_from_tdes, **kwargs) for comp_subset in split_solns)
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

if __name__ == '__main__':
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
    kwargs['K'] = 7
    dd = np.max(distance_matrix(array_geom, array_geom))/343  
    dd_samples = int(kwargs['fs']*dd)
    
    ignorable_start = int(0.01*fs)
    shift_samples = 96
    start_samples = np.arange(ignorable_start,array_audio.shape[0], shift_samples)
    end_samples = start_samples+dd_samples
    max_inds = int(0.2*fs/shift_samples)
    
    all_candidates = []
    i = 5
    audio_chunk = array_audio[start_samples[i]:end_samples[i]]
    #for (st, en) in tqdm.tqdm(zip(start_samples[:max_inds], end_samples[:max_inds])):
        # audio_chunk = array_audio[st:en,:]
    import time
    start = time.perf_counter_ns()
    all_locs = generate_candidate_sources(audio_chunk, **kwargs)
    print((time.perf_counter_ns()-start)/1e9 , ' s')
    all_locs.to_csv('np_outputs.csv')
    # generate_candidate_sources(audio_chunk, **kwargs)
    #%% 
    # %load_ext line_profiler
    # %lprun -f generate_candidate_sources generate_candidate_sources(audio_chunk, **kwargs)

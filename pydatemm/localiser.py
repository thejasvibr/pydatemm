#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCG Localiser
=============
Functions to localise sounds from CCG solutions. 

Created on Thu Sep 15 16:38:31 2022

@author: thejasvi
"""
import numpy as np 
from scipy.spatial import distance
euclidean = distance.euclidean
from sklearn import cluster
from pydatemm.localisation_mpr2003 import mellen_pachter_raquet_2003
from pydatemm.tdoa_quality import residual_tdoa_error_nongraph
import cppyy

from pydatemm.compilation_utils import load_and_compile_cpp_code

load_and_compile_cpp_code()




try:
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
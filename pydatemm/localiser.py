#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCG Localiser
=============
Functions to localise sounds from CCG solutions. 

"""
import numpy as np 
from scipy.spatial import distance
euclidean = distance.euclidean
from sklearn import cluster
print('MIAOW........../')
try:
    from pydatemm.compilation_utils import load_and_compile_with_own_flags
    load_and_compile_with_own_flags()
except ImportError:
    pass  

from time import perf_counter_ns as pcn
time_s = lambda : pcn()/1e9

try:
    import cppyy
    vector_cpp = cppyy.gbl.std.vector
    set_cpp = cppyy.gbl.std.set
except ImportError:
    vector_cpp = cppyy.gbl.std.vector
    set_cpp = cppyy.gbl.std.set
    pass

def combine_all(ac_cpp, v_cpp, l_cpp, x_cpp):
    '''
    A thin wrapper around the C++ combine_all - only here for documentation
    purposes. 
    
    Parameters
    ----------
    ac_cpp : vector<vector<int>> 
        The compatibility-conflict graph.
    v_cpp : set<int>
        Set of available vertices. 
    l_cpp : set<int>
        The current solution set
    x_cpp : set<int>
        Already visited vertices. 
    
    Returns
    -------
    vector<set<int>> 
        All possible solutions of compatible graphs. 
   
    See Also
    --------
    cppyy.gbl.combine_all
    
    '''    
    return cppyy.gbl.combine_all(ac_cpp, v_cpp, l_cpp, x_cpp)
    

def CCG_solutions_cpp(eigen_ccg_matrix):
    '''
    Receives Eigen CCG matrix and outputs a vector<vector<int>> with
    solutions. 
    
    Parameters
    ----------
    eigen_ccg_matrix : Eigen::MatrixXd (Ncfls, Ncfls)  C++ object
        2D square matrix indicating compatibility of all formed
        cFLs. 

    Returns 
    -------
    solns_cpp : vector<set<int>> C++ object
        The indices for all the compatible cFLS that can be combined
        to make a bigger graph.
    '''
    n_rows = int(eigen_ccg_matrix.rows());
    ac_cpp = cppyy.gbl.mat2d_to_vector(eigen_ccg_matrix)
    v_cpp = set_cpp[int](list(range(n_rows)))
    l_cpp = set_cpp[int]([])
    x_cpp = set_cpp[int]([])
    solns_cpp = combine_all(ac_cpp, v_cpp, l_cpp, x_cpp)
    return solns_cpp

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

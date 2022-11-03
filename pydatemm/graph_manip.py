#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph manipulations
===================

Created on Tue Oct 18 14:06:06 2022

@author: thejasvi
"""
from itertools import product, combinations
import igraph as ig
import joblib
from joblib import Parallel, delayed
import numpy as np
try:
    import cppyy as cpy
except:
    pass


def make_fundamental_loops(nchannels):
    '''
    Parameters
    ----------
    nchannels : int
        Number of channels, and thus number of nodes in the TDE graph.
    Returns
    -------
    fundamental_loops : list
        List with tuples containing integer node numbers.
    '''
    G = ig.Graph.Full(nchannels)
    G.vs['name'] = range(nchannels)
    minspan_G = G.spanning_tree()
    main_node = 0
    co_tree = minspan_G.complementer().simplify()
    fundamental_loops = []
    for edge in co_tree.es:
        fl_nodes = tuple((main_node, edge.source, edge.target))
        fundamental_loops.append(fl_nodes)
    return fundamental_loops


def make_triple_pairs(triple):
    pairs = combinations(sorted(triple),2)
    # reverse order to make them into j,i pairs from i,j pairs
    ji_pairs = list(map(lambda X: X[::-1], pairs))
    return ji_pairs

def make_edges_for_fundamental_loops(nchannels):
    '''
    Parameters
    ----------
    nchannels : int
        Num. of channels.
    
    Returns
    -------
    triple_definition : dict
        Keys are fundamental loops as tuples. Values are lists with
        edges as tuples
    '''
    funda_loops = make_fundamental_loops(nchannels)
    triple_definition = {}
    for fun_loop in funda_loops:
        edges = make_triple_pairs(fun_loop)
        triple_definition[fun_loop] = []
        # if the edge (ab) is present but the 'other' way round (ba) - then 
        # reverse polarity. 
        for edge in edges:
            triple_definition[fun_loop].append(edge)
    return triple_definition

def make_consistent_fls(multich_tdes, **kwargs):
    '''
    Parameters
    ----------
    multich_tdes : dict
        Keys are 
    '''
    max_loop_residual = kwargs.get('max_loop_residual', 1e-6)
    all_edges_fls = make_edges_for_fundamental_loops(kwargs['nchannels'])
    all_cfls = []

    for fundaloop, edges in all_edges_fls.items():
        a,b,c = fundaloop
        ba_tdes = multich_tdes[(b,a)]
        ca_tdes = multich_tdes[(c,a)]
        cb_tdes = multich_tdes[(c,b)]
        abc_combinations = list(product(ba_tdes, ca_tdes, cb_tdes))
        node_to_index = {nodeid: index for index, nodeid in  zip(range(3), fundaloop)}
        for i, (tde1, tde2, tde3) in enumerate(abc_combinations):
            if abs(tde1[1]-tde2[1]+tde3[1]) < max_loop_residual:
                this_cfl = ig.Graph(3, directed=True)
                this_cfl.vs['name'] = fundaloop
                for e, tde in zip(edges, [tde1, tde2, tde3]):
                    this_cfl.add_edge(node_to_index[e[0]], node_to_index[e[1]],
                                      tde=tde[1])
                all_cfls.append(this_cfl)
    return all_cfls

def make_all_entries_nan(EigenXd):
    '''Assigns all values in an Eigen Matrix to NaN
    '''
    for i in range(EigenXd.rows()):
        for j in range(EigenXd.cols()):
            EigenXd[i,j] = np.nan
    return EigenXd

def make_consistent_fls_cpp(multich_tdes, **kwargs):
    '''The C++-Python hybrid version. 
    '''
    max_loop_residual = kwargs.get('max_loop_residual', 1e-6)
    all_edges_fls = make_edges_for_fundamental_loops(kwargs['nchannels'])
    all_cfls = []
    k = 0
    for fundaloop, edges in all_edges_fls.items():
        a,b,c = fundaloop
        ba_tdes = multich_tdes[(b,a)]
        ca_tdes = multich_tdes[(c,a)]
        cb_tdes = multich_tdes[(c,b)]
        abc_combinations = list(product(ba_tdes, ca_tdes, cb_tdes))
        #node_to_index = {nodeid: index for index, nodeid in  zip(range(3), fundaloop)}
        for i, (tde1, tde2, tde3) in enumerate(abc_combinations):
            if abs(tde1[1]-tde2[1]+tde3[1]) < max_loop_residual:
                this_cfl = cpy.gbl.Eigen.MatrixXd(kwargs['nchannels'], kwargs['nchannels'])
                make_all_entries_nan(this_cfl)
                for e, tde in zip(edges, [tde1, tde2, tde3]):
                    if e[0] != e[1]:
                        #print(i, e[0], e[1])
                        this_cfl[e[0],e[1]] = tde[1]
                        this_cfl[e[1],e[0]] = tde[1]
                k += 1 
                all_cfls.append(this_cfl)
    return all_cfls

def node_names(ind_tup,X):
    node_names = X.vs['name']
    return tuple(node_names[i] for i in ind_tup)

def check_for_one_common_edge(X,Y):
    '''
    Parameters
    ----------
    X,Y : ig.Graph
    Returns 
    -------
    int : (-1,0,1)
        -1 indicates incompatibility, 1 - compatibility and 0 indicates
        NA
    '''
    X_edge_weights = [ (node_names(i.tuple, X), i['tde']) for i in X.es]
    Y_edge_weights = [ (node_names(i.tuple, Y), i['tde']) for i in Y.es]
    common_edge = set(Y_edge_weights).intersection(set(X_edge_weights))
    if len(common_edge)==1:
        return 1
    else:
        return -1

def ccg_definer(X,Y):
    common_nodes = set(X.vs['name']).intersection(set(Y.vs['name']))
    if len(common_nodes) >= 2:
        if len(common_nodes) < 3:
            relation = check_for_one_common_edge(X, Y)
        else:
            # all nodes the same
            relation = -1
    else:
        relation = -1
    return relation


def make_ccg_matrix(cfls, **kwargs):
    '''
    Sped up version. Previous version had explicit assignment of i,j and j,i
    compatibilities.
    '''
        
    num_cfls = len(cfls)
    ccg = np.zeros((num_cfls, num_cfls), dtype='int32')
    cfl_ij = combinations(range(num_cfls), 2)
    for (i,j) in cfl_ij:
        trip1, trip2  = cfls[i], cfls[j]
        cc_out = ccg_definer(trip1, trip2)
        ccg[i,j] = cc_out
    ccg += ccg.T
    return ccg

def get_compatibility(cfls, ij_combis):
    output = []
    for (i,j) in ij_combis:
        trip1, trip2  = cfls[i], cfls[j]
        cc_out = ccg_definer(trip1, trip2)
        output.append(cc_out)
    return output
    
def make_ccg_pll(cfls, **kwargs):
    '''Parallel version of make_ccg_matrix'''
    num_cores = kwargs.get('num_cores', int(joblib.cpu_count()))
    print(f'num cores: {num_cores}')
    num_cfls = len(cfls)

    all_ij = list(combinations(range(num_cfls), 2))
    cfl_ij_parts = [all_ij[i::num_cores] for i in range(num_cores)]
    compatibility = Parallel(n_jobs=num_cores)(delayed(get_compatibility)(cfls, ij_parts)for ij_parts in cfl_ij_parts)
    ccg = np.zeros((num_cfls, num_cfls), dtype='int32')
    for (ij_parts, compat_ijparts) in zip(cfl_ij_parts, compatibility):
        for (i,j), (comp_val) in zip(ij_parts, compat_ijparts):
            ccg[i,j] = comp_val
    
    # make symmetric
    ccg += ccg.T
    return ccg

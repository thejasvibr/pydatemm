#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:35:54 2022

@author: thejasvi
"""

import igraph as ig
from itertools import product, combinations, chain
import networkx as nx
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
import pandas as pd
import build_ccg as bccg
import ccg_localiser as ccl
from build_ccg import make_fundamental_loops, make_triple_pairs
from pydatemm.timediffestim import generate_multich_crosscorr, get_multich_tdoas
import tqdm
import numpy as np
import soundfile as sf
import time 
ns_time = time.perf_counter_ns

def ig_make_fundamental_loops(nchannels):
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

def ig_make_edges_for_fundamental_loops(nchannels):
    funda_loops = ig_make_fundamental_loops(nchannels)
    triple_definition = {}
    for fun_loop in funda_loops:
        edges = make_triple_pairs(fun_loop)
        triple_definition[fun_loop] = []
        # if the edge (ab) is present but the 'other' way round (ba) - then 
        # reverse polarity. 
        for edge in edges:
            triple_definition[fun_loop].append(edge)
    return triple_definition

def ig_make_consistent_fls(multich_tdes, **kwargs):
    max_loop_residual = kwargs.get('max_loop_residual', 1e-6)
    all_edges_fls = ig_make_edges_for_fundamental_loops(kwargs['nchannels'])
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

audio, fs = sf.read('3-bats_trajectory_simulation_0-order-reflections.wav')
start, end = np.int64(fs*np.array([0.01, 0.075]))
sim_audio = audio[start:end,:]
nchannels = audio.shape[1]
kwargs = {'nchannels':nchannels,
          'fs':fs,
          'pctile_thresh': 95,
          'use_gcc':True,
          'gcc_variant':'phat', 
          'min_peak_diff':0.35e-4, 
          'vsound' : 343.0, 
          'no_neg':False}
kwargs['max_loop_residual'] = 0.5e-4
kwargs['K'] = 9
array_geom = pd.read_csv('multibat_sim_micarray.csv').loc[:,'x':'z'].to_numpy()
kwargs['array_geom'] = array_geom
multich_cc = generate_multich_crosscorr(sim_audio, **kwargs )
cc_peaks = get_multich_tdoas(multich_cc, **kwargs)

K = kwargs.get('K',3)
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

def node_names(ind_tup,X):
    node_names = X.vs['name']
    return tuple(node_names[i] for i in ind_tup)
    
    
def ig_check_for_one_common_edge(X,Y):
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

def ig_ccg_definer(X,Y):
    common_nodes = set(X.vs['name']).intersection(set(Y.vs['name']))
    if len(common_nodes) >= 2:
        if len(common_nodes) < 3:
            relation = ig_check_for_one_common_edge(X, Y)
        else:
            # all nodes the same
            relation = -1
    else:
        relation = -1
    return relation


def ig_make_ccg_matrix(cfls):
    '''
    Sped up version. Previous version had explicit assignment of i,j and j,i
    compatibilities.
    '''
        
    num_cfls = len(cfls)
    ccg = np.zeros((num_cfls, num_cfls), dtype='int32')
    cfl_ij = combinations(range(num_cfls), 2)
    for (i,j) in cfl_ij:
        trip1, trip2  = cfls[i], cfls[j]
        cc_out = ig_ccg_definer(trip1, trip2)
        ccg[i,j] = cc_out
    ccg += ccg.T
    return ccg


def ig_get_compatibility(cfls, ij_combis):
    output = []
    for (i,j) in ij_combis:
        trip1, trip2  = cfls[i], cfls[j]
        cc_out = ig_ccg_definer(trip1, trip2)
        output.append(cc_out)
    return output



def ig_make_ccg_pll(cfls, **kwargs):
    '''Parallel version of make_ccg_matrix'''
    num_cores = kwargs.get('num_cores', int(joblib.cpu_count()*0.5))
    num_cfls = len(cfls)

    all_ij = list(combinations(range(num_cfls), 2))
    cfl_ij_parts = [all_ij[i::num_cores] for i in range(num_cores)]
    compatibility = Parallel(n_jobs=num_cores)(delayed(ig_get_compatibility)(cfls, ij_parts)for ij_parts in cfl_ij_parts)
    ccg = np.zeros((num_cfls, num_cfls), dtype='int32')
    for (ij_parts, compat_ijparts) in zip(cfl_ij_parts, compatibility):
        for (i,j), (comp_val) in zip(ij_parts, compat_ijparts):
            ccg[i,j] = comp_val
    # make symmetric
    ccg += ccg.T
    return ccg

def ig_get_tde(comp_triples):
    nodes = list(set(list(chain(*[each.vs['name'] for each in comp_triples]))))
    global_node_to_ij = {n:i for i, n in enumerate(nodes)}
    tde_mat = np.ones((len(nodes), len(nodes)))*np.nan
    for each in comp_triples:
        ind_to_node = {i:n for i,n in enumerate(each.vs['name'])}
        for edge in each.es:
            b,a = edge.tuple
            node_b, node_a = ind_to_node[b], ind_to_node[a]
            tde_mat[global_node_to_ij[node_b], global_node_to_ij[node_a]] = edge['tde']
    return tde_mat, nodes

def ig_chunk_create_tde_data(compatible_solutions, all_cfls, **kwargs):
    raw_tde_by_channelnum = {}
    cfl_ids = {} # for troubleshooting and error tracing
    for i, compat_cfl in enumerate(compatible_solutions):
        #source_graph = ig.union([all_cfls[j] for j in compat_cfl], byname=True)
        source_tde, channels = ig_get_tde([all_cfls[j] for j in compat_cfl])
        #source_tde = np.array(source_graph.get_adjacency(attribute='tde', default=np.nan).data)
        d = source_tde[1:,0]*kwargs['vsound']
        #channels = merged.vs['name']
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

def ig_combine_compatible_triples(triple_list):
    return ig.union(triple_list, byname=True)

if __name__ == "__main__":
    
    nruns = 100
    # sta = ns_time()/1e9
    # for i in tqdm.trange(nruns):
    #     ig_make_edges_for_fundamental_loops(i)
    # print(f'{ns_time()/1e9 - sta} s')

    # sta = ns_time()/1e9
    # for i in tqdm.trange(nruns):
    #     bccg.make_edges_for_fundamental_loops(i)
    # print(f'\n{ns_time()/1e9 - sta} s')
    # print('\n \n')
    # a = ns_time(); a/=1e9
    cfls_from_tdes = bccg.make_consistent_fls(top_K_tdes, **kwargs)
    # print(f'{ns_time()/1e9 - a} s')
    # a = ns_time(); a/=1e9
    ig_cfls_from_tdes = ig_make_consistent_fls(top_K_tdes, **kwargs)
    # print(f'{ns_time()/1e9 - a} s')
    # print(f'len cfls : {len(ig_cfls_from_tdes)}')
    #%%
    sta = ns_time()/1e9
    nx_ccg = bccg.make_ccg_matrix(cfls_from_tdes)
    print(f'NX: {ns_time()/1e9 - sta} s');
    sta = ns_time()/1e9
    ig_ccg = ig_make_ccg_matrix(ig_cfls_from_tdes)
    print(f'IG: {ns_time()/1e9 - sta} s')
    # try:
    #     assert np.all(ig_ccg==nx_ccg)
    # except:
    #     noteq = np.argwhere(ig_ccg!=nx_ccg)
    #     X, Y = [ig_cfls_from_tdes[i] for i in noteq[0,:]]
    #     X, Y = [cfls_from_tdes[i] for i in noteq[0,:]]

    #%% 
    # Comparing nx and ig outputs 
    # for nx_out, ig_out  in zip(cfls_from_tdes, ig_cfls_from_tdes):
    #     nx_tde = nx.adjacency_matrix(nx_out, weight='tde').todense()
    #     ig_tde = np.array(ig_out.get_adjacency(attribute='tde').data)
        
    #     nx_weights = nx.get_edge_attributes(nx_out,'tde')
    #     for edge in ig_out.es:
    #         b,a = edge.tuple
    #         node_b, node_a = ig_out.vs['name'][b], ig_out.vs['name'][a]
    #         try:
    #             assert edge['tde'] == nx_weights[(node_b, node_a)]
    #         except:
    #             assert edge['tde'] == nx_weights[(node_a, node_b)]

    #%%
    sta = ns_time()/1e9
    ii = ig_make_ccg_pll(ig_cfls_from_tdes, **kwargs)
    sto = ns_time()/1e9; print(f'IG PLL {sto-sta} s')   
    # jj = bccg.make_ccg_pll(cfls_from_tdes, **kwargs)
    # print(f'{ns_time()/1e9-sta} s')
    
    #%% 
    ccg_solns = ccl.CCG_solutions(ig_ccg)
    #%%
    a_soln = ccg_solns[-500]
    comp_triples = [ig_cfls_from_tdes[i] for i in a_soln]
    merged = ig.union(comp_triples, byname=True)
    
    
    
    def ig_oldway(comp_triples):
        merged = ig.union(comp_triples, byname=True)
        return np.array(merged.get_adjacency(attribute='tde', default=np.nan).data)
    
    def nx_oldway(comp_triples):
        source_tde = nx.to_numpy_array(ccl.combine_compatible_triples(comp_triples), 
                                   weight='tde', nonedge=np.nan)
        return source_tde
    
    def nx_newway(comp_triples):
        nodes = list(set(list(chain(*[each.nodes for each in comp_triples]))))
        global_node_to_ij = {n:i for i, n in enumerate(nodes)}
        #tde_mat = np.full((len(nodes), len(nodes)), np.nan)
        tde_mat = np.zeros((len(nodes), len(nodes)))
        for each in comp_triples:
            ind_to_node = {i:n for i,n in enumerate(each.nodes)}
            for edge in each.edges:
                b,a = edge
                #node_b, node_a = ind_to_node[b], ind_to_node[a]
                i, j = global_node_to_ij[b], global_node_to_ij[a]
                tde_mat[i,j] = each.get_edge_data(*edge)['tde']
        #tde_mat[np.isnan(tde_mat)] = 0
        tde_mat += tde_mat.T
        tde_mat[tde_mat==0] = np.nan
        return tde_mat, nodes
    
    speedup_to_previg = []
    speedup_to_nx = []
    speedup_newnx_newig = []
    speedup_newnx_oldnx = []
    for each in ccg_solns[:10000]:
        comp_triples = [ig_cfls_from_tdes[i] for i in each]
        nx_comptriples = [cfls_from_tdes[i] for i in each]
        sta = ns_time()/1e9
        tdemat, nodes = ig_get_tde(comp_triples)
        sto_1 = ns_time()/1e9
        qq = ig_oldway(comp_triples)
        sto_2 = ns_time()/1e9
        rr = nx_oldway(nx_comptriples)
        sto_3 = ns_time()/1e9
        z = nx_newway(nx_comptriples)
        sto_4 = ns_time()/1e9
        
        new_ig = sto_1 - sta
        old_ig = sto_2 - sto_1
        old_nx = sto_3 - sto_2 
        new_nx = sto_4 - sto_3
        
        
        speedup_to_previg.append(old_ig/new_ig)
        speedup_to_nx.append(old_nx/new_ig)
        speedup_newnx_newig.append(new_nx/new_ig)
        speedup_newnx_oldnx.append(old_nx/new_nx)
        
        assert np.allclose(np.tril(tdemat) , np.tril(z[0]), equal_nan=True)
        #print(f'myway {sto_1-sta} s, oldway: {sto_2-sto_1} s')                
        assert np.allclose(qq,tdemat,equal_nan=True)
    #%%
    # NX implementation
    num_inds = 100
    sta = ns_time()/1e9
    nx_tde, cfl_ids = ccl.chunk_create_tde_data(ccg_solns[:num_inds], cfls_from_tdes, **kwargs)
    nx_stop = ns_time()/1e9
    ig_tde, ig_cfl_ids = ig_chunk_create_tde_data(ccg_solns[:num_inds], ig_cfls_from_tdes, **kwargs)
    ig_stop = ns_time()/1e9
    print(f'NX: {nx_stop-sta} s, IG: {ig_stop-nx_stop} s ')
    
    for key, values in nx_tde.items():
        assert np.all(ig_tde[key]==values)
    #%%
    %load_ext line_profiler
    %lprun -f ig.union ig_chunk_create_tde_data(ccg_solns[:num_inds], ig_cfls_from_tdes, **kwargs)
    
    #%%
    
    plt.figure()
    a1 = plt.subplot(111)
    vis_style = {}
    vis_style['target'] = a1
    vis_style["edge_label"] = np.around(np.array(merged.es["tde"])*1e3,2)
    vis_style["vertex_label"] = merged.vs['name']
    
    ig.plot(merged,   **vis_style)
    
    
    
    #%%
    g = ig_cfls_from_tdes[-1]
    plt.figure()
    a0 = plt.subplot(111)
    vis_style = {}
    vis_style['target'] = a0
    vis_style["edge_label"] = np.around(np.array(g.es["tde"])*1e3,2)
    vis_style["vertex_label"] = g.vs['name']
    
    ig.plot(g,   **vis_style)
    
    


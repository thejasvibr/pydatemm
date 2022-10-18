#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source generation
=================
Calculates all possible sources that generate consistent graphs.
Also contains the boilerplate code to make the correct data formats
for the Eigen linear algebra. 

Created on Tue Oct 18 15:17:29 2022

@author: thejasvi
"""
import numpy as np 
from itertools import chain
import joblib
from joblib import Parallel, delayed
import pydatemm.localiser as lo
import pydatemm.timediffestim as timediff
import  pydatemm.graph_mainp as gramanip

def get_tde(comp_triples):
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

def chunk_create_tde_data(compatible_solutions, all_cfls, **kwargs):
    raw_tde_by_channelnum = {}
    cfl_ids = {} # for troubleshooting and error tracing
    for i, compat_cfl in enumerate(compatible_solutions):
        #source_graph = ig.union([all_cfls[j] for j in compat_cfl], byname=True)
        source_tde, channels = get_tde([all_cfls[j] for j in compat_cfl])
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

def pll_create_tde_data(compatible_solutions,
                           all_cfls, **kwargs):
    parts = kwargs.get('num_cores', joblib.cpu_count())
    #split data into parts
    split_solns = (compatible_solutions[i::parts] for i in range(parts))
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

# def combine_compatible_triples(triple_list):
#     '''Vestigial function - not performant - but still kept.
#     '''
#     return ig.union(triple_list, byname=True)

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

def localise_sounds_v2(compatible_solutions, all_cfls, **kwargs):
    '''
    '''
    tde_data, cfl_ids = create_tde_data(compatible_solutions, all_cfls, **kwargs)
    all_sources = []
    all_cfls = []
    all_tdedata = []
    for (nchannels, tde_input) in tde_data.items():
       
        if nchannels > 4:
            calc_sources = lo.pll_cppyy_sw2002(tde_input, kwargs['vsound'])
            all_sources.append(calc_sources)
            all_cfls.append(cfl_ids[nchannels])
            all_tdedata.append(tde_input.tolist())
        elif nchannels == 4:
            fourchannel_cflids= []
            for i in range(tde_input.shape[0]):
                calc_sources = lo.row_based_mpr2003(tde_input[i,:])
                if calc_sources.size==6:
                    all_sources.append(calc_sources[0,:])
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    all_tdedata.append(tde_input.tolist())
                    all_sources.append(calc_sources[1,:])
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    all_tdedata.append(tde_input.tolist())
                elif calc_sources.size==3:
                    all_sources.append(calc_sources)
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    all_tdedata.append(tde_input.tolist())
                elif len(calc_sources) == 0:
                    pass
            all_cfls.append(fourchannel_cflids)
        else:
            pass # if <4 channels encountered
    if len(all_sources)>0:
        return np.row_stack(all_sources), list(chain(*all_cfls)), list(chain(*all_tdedata))
    else:
        return np.array([]), [], []


def generate_candidate_sources_v2(sim_audio, **kwargs):
    multich_cc = timediff.generate_multich_crosscorr(sim_audio, **kwargs )
    cc_peaks = timediff.get_multich_tdoas(multich_cc, **kwargs)

    K = kwargs.get('K',5) # number of peaks per channel CC to consider
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
    cfls_from_tdes = gramanip.make_consistent_fls(top_K_tdes, **kwargs)
    print(f'len of cfls: {len(cfls_from_tdes)}')
    # put all consistent loops into fundamental loop 'bins'
    all_fls = gramanip.make_fundamental_loops(kwargs['nchannels'])
    cfls_by_fl = {}
    for fl in all_fls:
        cfls_by_fl[fl] = []
    
    for i,each in enumerate(cfls_from_tdes):
        for fl in all_fls:
            if set(each.vs['name']) == set(fl):
                cfls_by_fl[fl].append(i)
    print('Making CCG matrix')
    if len(cfls_from_tdes) < 200:
        ccg_matrix = gramanip.make_ccg_matrix(cfls_from_tdes)
    else:
        ccg_matrix = gramanip.make_ccg_pll(cfls_from_tdes)
    print('Finding solutions')
    solns_cpp = lo.CCG_solutions(ccg_matrix)
    print('Found solutions')
    print(f'Doing tracking: {len(solns_cpp)}')
    sources, cfl_ids, tdedata = localise_sounds_v2(solns_cpp, cfls_from_tdes, **kwargs)
    print('Done with tracking.')
    return sources, cfl_ids, tdedata
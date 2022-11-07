#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:17:05 2022

@author: thejasvi
"""
import numpy as np 
import joblib
from joblib import Parallel, delayed
from itertools import chain
from pydatemm.localisation_mpr2003 import mellen_pachter_raquet_2003
from pydatemm.tdoa_quality import residual_tdoa_error_nongraph
#import pydatemm.localiser as lo
import pydatemm.timediffestim as timediff
import  pydatemm.graph_manip as gramanip


try:
    import cppyy
    vector_cpp = cppyy.gbl.std.vector
    set_cpp = cppyy.gbl.std.set
except ImportError:
    vector_cpp = cppyy.gbl.std.vector
    set_cpp = cppyy.gbl.std.set
    pass

get_nmics = lambda X: int((X.size+1)/4)


def CCG_solutions(ccg_matrix):
    n_rows = ccg_matrix.shape[0]
    ac_cpp = vector_cpp[vector_cpp[int]]([ccg_matrix[i,:].tolist() for i in range(n_rows)])
    print(type(ac_cpp), ac_cpp.size())
    v_cpp = set_cpp[int](range(n_rows))
    l_cpp = set_cpp[int]([])
    x_cpp = set_cpp[int]([])

    solns_cpp = cppyy.gbl.combine_all(ac_cpp, v_cpp, l_cpp, x_cpp)
    comp_cfls = list([]*len(solns_cpp))
    comp_cfls = [list(each) for each in solns_cpp]
    return comp_cfls



def cppyy_sw2002(micntde):
    as_Vxd = cppyy.gbl.sw_matrix_optim(vector_cpp['double'](micntde),
                                       )
    return np.array(as_Vxd, dtype=np.float64)


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



def make_vect_vect_double(X):
    data = vector_cpp[vector_cpp['double']](X.shape[0])
    for i in range(X.shape[0]):
        data.push_back(X[i,:])
    return data[X.shape[0]:]


def pll_cppyy_sw2002(many_micntde, num_cores, c):
    block_in = make_vect_vect_double(many_micntde)
    block_out = cppyy.gbl.pll_sw_optim(block_in, num_cores, c)
    return np.array([each for each in block_out])


def get_numrows(X):
    try:
        nrows, ncols = X.shape
    except ValueError:
        if len(X)>0:
            nrows = 1 
        else:
            nrows = 0
    return nrows

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
    raw_tde_by_channelnum = {}
    cfl_ids = {} # for troubleshooting and error tracing
    for i, compat_cfl in enumerate(compatible_solutions):
        source_tde, channels = get_tde([all_cfls[j] for j in compat_cfl])
        d = source_tde[1:,0]*kwargs['vsound']
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



def localise_sounds_v2(compatible_solutions, all_in_cfls, **kwargs):
    '''
    '''
    num_cores = kwargs.get('num_cores', joblib.cpu_count())
    tde_data, cfl_ids = create_tde_data(compatible_solutions, all_in_cfls, **kwargs)
    all_sources = []
    all_cfls = []
    all_tdedata = []
    for (nchannels, tde_input) in tde_data.items():
        if nchannels > 4:
            calc_sources = pll_cppyy_sw2002(tde_input, num_cores, kwargs['vsound'])
            all_sources.append(calc_sources)
            all_cfls.append(cfl_ids[nchannels])
            all_tdedata.append(tde_input.tolist())
        elif nchannels == 4:
            miaow = 0
            fourchannel_cflids= []
            fourchannel_tdedata = []
            for i in range(tde_input.shape[0]):
                calc_sources = lo.row_based_mpr2003(tde_input[i,:])
                nrows = get_numrows(calc_sources)
                if nrows == 2:
                    all_sources.append(calc_sources[0,:])
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    fourchannel_tdedata.append(tde_input[i,:].tolist())
                    all_sources.append(calc_sources[1,:])
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    fourchannel_tdedata.append(tde_input[i,:].tolist())
                elif nrows == 1:
                    all_sources.append(calc_sources)
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    fourchannel_tdedata.append(tde_input[i,:].tolist())
                elif nrows == 0:
                    miaow += 1
                    pass                
            all_cfls.append(fourchannel_cflids)
            all_tdedata.append(fourchannel_tdedata)
        else:
            pass # if <4 channels encountered
    if len(all_sources)>0:
        return np.row_stack(all_sources), list(chain(*all_cfls)), list(chain(*all_tdedata))
    else:
        return np.array([]), [], []




def localise_sounds_cpp_v2(compatible_solutions, all_in_cfls, **kwargs):
    '''
    C++ version with compatible_solutions having 
    '''
    num_cores = kwargs.get('num_cores', joblib.cpu_count())
    tde_data, cfl_ids = create_tde_data(compatible_solutions, all_in_cfls, **kwargs)
    all_sources = []
    all_cfls = []
    all_tdedata = []
    for (nchannels, tde_input) in tde_data.items():
        if nchannels > 4:
            calc_sources = lo.pll_cppyy_sw2002(tde_input, num_cores, kwargs['vsound'])
            all_sources.append(calc_sources)
            all_cfls.append(cfl_ids[nchannels])
            all_tdedata.append(tde_input.tolist())
        elif nchannels == 4:
            fourchannel_cflids= []
            fourchannel_tdedata = []
            for i in range(tde_input.shape[0]):
                calc_sources = lo.row_based_mpr2003(tde_input[i,:])
                nrows = get_numrows(calc_sources)
                if nrows == 2:
                    all_sources.append(calc_sources[0,:])
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    fourchannel_tdedata.append(tde_input[i,:].tolist())
                    all_sources.append(calc_sources[1,:])
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    fourchannel_tdedata.append(tde_input[i,:].tolist())
                elif nrows == 1:
                    all_sources.append(calc_sources)
                    fourchannel_cflids.append(cfl_ids[nchannels][i])
                    fourchannel_tdedata.append(tde_input[i,:].tolist())
                elif nrows == 0:
                    pass                
            all_cfls.append(fourchannel_cflids)
            all_tdedata.append(fourchannel_tdedata)
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
    cfls_from_tdes = gramanip.make_consistent_fls(top_K_tdes, **kwargs)

    ccg_matrix = gramanip.make_ccg_matrix(cfls_from_tdes, **kwargs)
    solns_cpp = lo.CCG_solutions(ccg_matrix)
    sources, cfl_ids, tdedata = localise_sounds_v2(solns_cpp, cfls_from_tdes, **kwargs)
    return sources, cfl_ids, tdedata
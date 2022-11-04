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
import os 
import time
import pydatemm.localiser as lo
import pydatemm.timediffestim as timediff
import  pydatemm.graph_manip as gramanip
try:
    import cppyy as cpy
except:
    pass
    

def cpp_make_array_geom(**kwargs):
    ''' Takes the np.array array_geom & converts it 
    into a Eigen::MatrixXd 
    '''
    nmics, ncols = kwargs['array_geom'].shape
    cpp_array_geom = cpy.gbl.Eigen.MatrixXd(nmics, ncols)
    for i in range(nmics):
        for j in range(ncols):
            cpp_array_geom[i,j] = kwargs['array_geom'][i,j]
    return cpp_array_geom



def generate_candidate_sources_hybrid(sim_audio, **kwargs):
    '''
    generate_candidate_sources_v2 but with C++ graphs as Eigen Matrices
    '''
    num_cores = kwargs.get('num_cores', os.cpu_count())
    multich_cc = timediff.generate_multich_crosscorr(sim_audio, **kwargs )
    cc_peaks = timediff.get_multich_tdoas(multich_cc, **kwargs)
    K = kwargs.get('K',7) # number of peaks per channel CC to consider
    top_K_tdes = {}
    for ch_pair, tdes in cc_peaks.items():
        descending_quality = sorted(tdes, key=lambda X: X[-1], reverse=True)
        top_K_tdes[ch_pair] = []
        for i in range(K):
            try:
                top_K_tdes[ch_pair].append(descending_quality[i])
            except:
                pass
    cfls_from_tdes = gramanip.make_consistent_fls_cpp(top_K_tdes, **kwargs)
    ccg_matrix = cpy.gbl.make_ccg_matrix(cfls_from_tdes)
    solns_cpp = lo.CCG_solutions_cpp(ccg_matrix)
    ag = cpp_make_array_geom(**kwargs)
    data_struct = cpy.gbl.localise_sounds_v3(num_cores, ag, solns_cpp, cfls_from_tdes)
    return data_struct





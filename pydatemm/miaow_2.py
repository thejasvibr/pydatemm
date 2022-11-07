#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:43:39 2022

@author: thejasvi
"""
import numpy as np 
import sys
sys.path.append("../examples/")
sys.path.append("tests/igraph_implementation/")
from igraph_based_functions import generate_candidate_sources_v2

import time 
nstime = lambda : time.perf_counter_ns()/1e9
from sim_example import sim_audio, kwargs

from pydatemm.source_generation import generate_candidate_sources

#%%
kwargs['num_cores'] = 4
kwargs['K'] = 2
a = nstime()
dcpp = generate_candidate_sources(sim_audio, **kwargs)
b = nstime()
print(f'C++: {b-a}')
cpp_sources = np.array([each for each in dcpp.sources])
#%%
a = nstime()
py_sources, py_cfls, py_tdein = generate_candidate_sources_v2(sim_audio, **kwargs)
b = nstime()
print(f'Python-based: {b-a}')

#%load_ext line_profiler
#%lprun -f generate_candidate_sources_hybrid generate_candidate_sources_hybrid(sim_audio, **kwargs)

#%% Check to see where the difference in output length appears
cpp_nchannels = [ (each.size()+1)/4 for each in dcpp.tde_in]
cpp_nch, cpp_counts = np.unique(cpp_nchannels, return_counts=True)

py_nchannels = [ (len(each)+1)/4 for each in py_tdein]
py_nch, py_counts = np.unique(py_nchannels, return_counts=True)


#%% Check that the a common input results in the same outputs from 
# for both implementations 
try:
    import cppyy as cpy
except:
    pass
from pydatemm.localisation import spiesberger_wahlberg_solution as sws
from pydatemm.tdoa_quality import residual_tdoa_error_nongraph


difference = []
for each in py_tdein:
    input1 = np.array(each)
    nmics = int((len(input1)+1)/4)
    input1_cpp = cpy.gbl.std.vector["double"](each for each in input1)
    
    cpp_out = cpy.gbl.sw_matrix_optim(input1_cpp)
    cpp_out = np.array(cpp_out)
    try:
        py_out = sws(input1[:nmics*3].reshape(-1,3), input1[-(nmics-1):])
        tdoa_res = residual_tdoa_error_nongraph(input1[-(nmics-1):], py_out,input1[:nmics*3].reshape(-1,3))
        py_out = np.array([*py_out, tdoa_res])
    except ValueError:
        py_out = np.tile(np.nan, 4)

    if not np.sum(np.isnan(py_out)) > 0:
        difference.append(np.sqrt(np.sum((py_out-cpp_out)**2)))
        
difference = np.array(difference)        




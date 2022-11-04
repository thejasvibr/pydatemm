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

from pydatemm.source_generation import generate_candidate_sources_hybrid

#%%
kwargs['num_cores'] = 4
kwargs['K'] = 3
for i in range(1):
    a = nstime()
    dcpp = generate_candidate_sources_hybrid(sim_audio, **kwargs)
    b = nstime()
    print(f'C++: {b-a}')
#%%
a = nstime()
dpy = generate_candidate_sources_v2(sim_audio, **kwargs)
b = nstime()
print(f'Python-based: {b-a}')

#%load_ext line_profiler
#%lprun -f generate_candidate_sources_hybrid generate_candidate_sources_hybrid(sim_audio, **kwargs)

#%% Check to see where the difference in output length appears
cpp_nchannels = [ (each.size()+1)/4 for each in dcpp.tde_in]
cpp_nch, cpp_counts = np.unique(cpp_nchannels, return_counts=True)

py_nchannels = [ (len(each)+1)/4 for each in dpy[2]]
py_nch, py_counts = np.unique(py_nchannels, return_counts=True)



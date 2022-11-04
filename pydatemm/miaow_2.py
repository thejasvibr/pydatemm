#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:43:39 2022

@author: thejasvi
"""
import sys
sys.path.append("../examples/")
import sim_example
import time 
nstime = lambda : time.perf_counter_ns()/1e9
from sim_example import sim_audio, kwargs
#import pydatemm.localiser as lo
#import pydatemm.timediffestim as timediff
#import  pydatemm.graph_manip as gramanip
#try:
#    import cppyy as cpy
#except:
#    pass
from pydatemm.source_generation import generate_candidate_sources_hybrid
from pydatemm.source_generation import generate_candidate_sources_v2

kwargs['num_cores'] = 4
kwargs['K'] = 3
a = nstime()
dd = generate_candidate_sources_hybrid(sim_audio, **kwargs)
b = nstime()
print(f'C++: {b-a}')
#%%
%load_ext line_profiler
%lprun -f generate_candidate_sources_hybrid generate_candidate_sources_hybrid(sim_audio, **kwargs)

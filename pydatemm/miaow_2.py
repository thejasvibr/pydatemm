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
from pydatemm.source_generation import cpp_make_array_geom, generate_candidate_sources_hybrid
from pydatemm.source_generation import generate_candidate_sources_v2
a = nstime()
dd = generate_candidate_sources_hybrid(sim_audio, **kwargs)
b = nstime()
ddv = generate_candidate_sources_v2(sim_audio, **kwargs)
c = nstime()
print(f'C++: {b-a} s, Python: {c-b}')

#
#o = nstime()
#multich_cc = timediff.generate_multich_crosscorr(sim_audio, **kwargs )
#cc_peaks = timediff.get_multich_tdoas(multich_cc, **kwargs)
#
#K = kwargs.get('K',5) # number of peaks per channel CC to consider
#top_K_tdes = {}
#for ch_pair, tdes in cc_peaks.items():
#    descending_quality = sorted(tdes, key=lambda X: X[-1], reverse=True)
#    top_K_tdes[ch_pair] = []
#    for i in range(K):
#        try:
#            top_K_tdes[ch_pair].append(descending_quality[i])
#        except:
#            pass
#print('making the cfls...')
#cfls_from_tdes = gramanip.make_consistent_fls_cpp(top_K_tdes, **kwargs)
#print(f'len of cfls: {len(cfls_from_tdes)}')
#
#print('Making CCG matrix')
#if len(cfls_from_tdes) > 200:
#    ccg_matrix = cpy.gbl.make_ccg_matrix(cfls_from_tdes)
#    
#else:
#    ccg_matrix = gramanip.make_ccg_pll(cfls_from_tdes, **kwargs)
#print('Finding solutions')
#solns_cpp = lo.CCG_solutions_cpp(ccg_matrix)
#print('Found solutions')
#print(f'Doing tracking: {len(solns_cpp)}')
#ag = cpp_make_array_geom(**kwargs)
#a = nstime()
#data_struct = cpy.gbl.localise_sounds_v2(3, ag, solns_cpp, cfls_from_tdes)
#b = nstime(); print(f'{(b-o)/1e9} seconds for localisaation')

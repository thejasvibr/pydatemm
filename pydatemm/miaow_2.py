#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:43:39 2022

@author: thejasvi
"""
import sys
sys.path.append("../examples/")
import sim_example
from sim_example import sim_audio, kwargs
import pydatemm.localiser as lo
import pydatemm.timediffestim as timediff
import  pydatemm.graph_manip as gramanip
try:
    import cppyy as cpy
except:
    pass
from pydatemm.source_generation import cpp_make_array_geom

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
cfls_from_tdes = gramanip.make_consistent_fls_cpp(top_K_tdes, **kwargs)
print(f'len of cfls: {len(cfls_from_tdes)}')

print('Making CCG matrix')
if len(cfls_from_tdes) > 200:
    ccg_matrix = cpy.gbl.make_ccg_matrix(cfls_from_tdes)
    
else:
    ccg_matrix = gramanip.make_ccg_pll(cfls_from_tdes, **kwargs)
print('Finding solutions')
solns_cpp = lo.CCG_solutions_cpp(ccg_matrix)
print('Found solutions')
print(f'Doing tracking: {len(solns_cpp)}')
ag = cpp_make_array_geom(**kwargs)
data_struct = cpy.gbl.localise_sounds_v21(3, ag, solns_cpp, cfls_from_tdes)

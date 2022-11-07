#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Making sure the C++ and Python outputs are the same
===================================================
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:43:39 2022

@author: thejasvi
"""
import numpy as np 
import sys
import unittest
import os 

sys.path.append(os.path.join('../../','examples/'))
#sys.path.append("../examples/")
sys.path.append("./igraph_implementation/")
from igraph_based_functions import generate_candidate_sources_v2
from sim_example import sim_audio, kwargs
from pydatemm.source_generation import generate_candidate_sources
try:
    import cppyy as cpy
except:
    pass
from pydatemm.localisation import spiesberger_wahlberg_solution as sws
from pydatemm.tdoa_quality import residual_tdoa_error_nongraph

class CppPython_Equivalence(unittest.TestCase):
    
    @classmethod 
    def setUpClass(self):
        kwargs['num_cores'] = 3
        kwargs['K'] = 2
        self.dcpp = generate_candidate_sources(sim_audio, **kwargs)
        self.cpp_sources = np.array([each for each in self.dcpp.sources])
        self.py_sources, self.py_cfls, self.py_tdein = generate_candidate_sources_v2(sim_audio, **kwargs)
    
    def test_check_output_length(self):
        cpp_nchannels = [ (each.size()+1)/4 for each in self.dcpp.tde_in]
        cpp_nch, cpp_counts = np.unique(cpp_nchannels, return_counts=True)

        py_nchannels = [ (len(each)+1)/4 for each in self.py_tdein]
        py_nch, py_counts = np.unique(py_nchannels, return_counts=True)
        python_out =  np.concatenate((py_nch, py_counts))
        cpp_out = np.concatenate((cpp_nch, cpp_counts))
        same_channels_and_counts= np.array_equal(python_out, cpp_out)
        self.assertTrue(same_channels_and_counts)
        
    
    def test_check_output_results(self):
        difference = []
        for each in self.py_tdein:
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
        self.assertTrue(np.max(difference<1e-10))


if __name__ == "__main__":
    unittest.main()





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Source Generation
======================
"""
import unittest 
import numpy as np 
np.random.seed(78464)
from pydatemm.source_generation import * 


class TestSourceGeneration(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.empty_audio = np.zeros((2000, 8))
        self.random_audio = np.random.normal(0,1e-3,2000*8).reshape(-1,8)
        self.kwargs = {}
        self.kwargs['fs'] = 192000
        self.kwargs['array_geom'] = np.random.normal(0,1,24).reshape(8,3)
        self.kwargs['vsound'] = 343.0
        self.kwargs['K'] = 5
        self.kwargs['max_loop_residual'] = 1e-5
        self.kwargs['min_peak_diff'] = 1e-3
        self.kwargs['pctile_thresh'] = 95
    
    def test_empty_audio_noerror(self):
        output = generate_candidate_sources(self.empty_audio, **self.kwargs)
        self.assertEqual(output.sources, [])
    
    def test_normal_audio_noerror(self):
        output = generate_candidate_sources(self.random_audio, **self.kwargs)


if __name__ == "__main__":
    unittest.main()
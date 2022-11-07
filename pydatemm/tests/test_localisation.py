# -*- coding: utf-8 -*-
"""
Localisation Tests
==================
Python localisation algorithm tests
"""
from scipy.spatial import distance, distance_matrix 
import unittest
import numpy as np 
import pandas as pd
from pydatemm.localisation import *

def make_single_source_in_array(source1=np.array([1.67,1.66,0.71]), 
                                **kwargs):
    '''
    '''
    seednum = 78464 # 8221, 82319, 78464
    np.random.seed(seednum) # what works np.random.seed(82310)
    if kwargs.get('array_geom') is None:
        array_geom = pd.read_csv('scheuing-yang-2008_micpositions.csv').to_numpy()
    else:
        array_geom = kwargs['array_geom']
    all_points = np.vstack((source1, array_geom))
    distmat = distance_matrix(all_points, all_points)
    mic_dist_to_source = distmat[0,1:]
    ref_channel = 0
    rel_dist =  mic_dist_to_source[1:] - mic_dist_to_source[0]
    return source1, array_geom, rel_dist

class TestSimpleSpiesbergerWahlberg(unittest.TestCase):
    def setUp(self):
        self.original_source, self.array_geom, self.rel_dist = make_single_source_in_array()
    def test_simple(self):
        source = spiesberger_wahlberg_solution(self.array_geom, self.rel_dist)
        print(source)
        same_as_expected = np.allclose(source, self.original_source, atol=1e-5)
        self.assertTrue(same_as_expected)

class TestCorrectSolutionChoice(unittest.TestCase):
    def setUp(self):
        self.source = np.array([5,0.5,-2])
        out = make_single_source_in_array(source1=self.source)
        self.original_source, self.array_geom, self.rel_dist = out
        self.array_geom = self.array_geom[:5,:] 
        self.rel_dist = self.rel_dist[:5]
    
    def test_correct_choice_5channel(self):
        calculated_source = spiesberger_wahlberg_solution(self.array_geom,
                                                          self.rel_dist,)
        print(calculated_source)
        expected_source = np.allclose(self.original_source, calculated_source, atol=1e-5)
        self.assertTrue(expected_source)
    
    def test_correct_choise_nchannel(self):
        for i in range(10):
            random_source = np.random.normal(0,100,3)
            out = make_single_source_in_array(source1=random_source)
            self.original_source, self.array_geom, self.rel_dist = out
            calculated_source = spiesberger_wahlberg_solution(self.array_geom,
                                                              self.rel_dist,)
            expected_source = np.allclose(self.original_source, calculated_source, atol=1e-5)
            self.assertTrue(expected_source)

if __name__ == '__main__':
    unittest.main()
    source = np.array([1.67,1.66,0.71])
    out = make_single_source_in_array(source1=source)
    original_source, array_geom, rel_dist = out

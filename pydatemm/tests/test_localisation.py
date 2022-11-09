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
from pydatemm.localisation_sw2002 import *
from pydatemm.localisation_mpr2003 import *

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

class TestMPRboth_solutions(unittest.TestCase):
    
    def setUp(self):
        '''Choose only 4 channels.
        '''
        self.source = np.array([5,0.5,-2])
        out = make_single_source_in_array(source1=self.source)
        self.original_source, self.array_geom, self.rel_dist = out
        
        self.source2 = np.array([5,10.5,-2])
        out2 = make_single_source_in_array(source1=self.source2)
        self.original_source2, self.array_geom2, self.rel_dist2 = out2
        
        self.array_geom = self.array_geom[:4,:] 
        self.rel_dist = self.rel_dist[:3]
        self.rel_dist2 = self.rel_dist2[:3]
    
    def test_unique_solution(self):
        output = mellen_pachter_raquet_2003(self.array_geom, self.rel_dist)
        residual = np.sqrt(np.sum((output - self.source)**2))
            
        one_solution_correct = residual<1e-13
        self.assertTrue(one_solution_correct)
    
    def test_two_solutions(self):
        
        output = mellen_pachter_raquet_2003(self.array_geom, self.rel_dist2)
        # check that self.source2 is one of the outputs
        residual = np.sum((output - self.source2)**2, 1)
        one_solution_correct = sum(residual<1e-13)==1
        self.assertTrue(one_solution_correct)

if __name__ == '__main__':
    unittest.main()
#    source = np.array([5, 10.5, -2])
#    out = make_single_source_in_array(source1=source)
#    original_source, array_geom, rel_dist = out
#    array_geom = array_geom[:4,:]
#    rel_dist = rel_dist[:3]
#    out = mellen_pachter_raquet_2003(array_geom, rel_dist)
#    resid = np.sum((out-source)**2,1)
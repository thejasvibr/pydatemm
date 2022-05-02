# -*- coding: utf-8 -*-
"""
Common Functions Test Suite
===========================
Created on Mon May  2 17:01:39 2022

@author: thejasvi beleyur
"""
import unittest 
from pydatemm.common_funcs import *

class TestMergeGraphs(unittest.TestCase):
    
    def  setUp(self):
        self.a = np.empty((4,4))
        self.a[:,:] = np.nan
        self.b = self.a.copy()
    
    def prep_good_merge(self):
        '''
        Makes two symmetric, incomplete and mergeable 4 node graphs
        '''
        self.a[1,0] = -0.5; self.a[0,1] = 0.5 
        self.a[2,0] = 1; self.a[0,2] = -1
        self.a[3,1] = 1; self.a[1,3] = -1
        
        self.b[2,0] = 1; self.b[0,2] = -1
        self.b[3,0] = 2; self.b[0,3] = -2
        self.b[3,1] = 1; self.b[1,3] = -1
    
    def test_simple_merge(self):
        self.prep_good_merge()
        output_graph = merge_graphs([self.a, self.b])
        num_valid_entries = int(np.sum(~np.isnan(output_graph))*0.5)
        expected_valid_entries = 4
        self.assertEqual(num_valid_entries, expected_valid_entries)
    
    def test_bad_merge(self):
        self.prep_good_merge()
        self.a[1,0] = 2
        print(self.a, '\n', self.b)
        output_graph = merge_graphs([self.a, self.b])
        #with self.assertRaises(ValueError, msg='Unmergeable graphs. Some entries are incompatible'):
            
        
if __name__ == '__main__':
    unittest.main()
        
        
        
    


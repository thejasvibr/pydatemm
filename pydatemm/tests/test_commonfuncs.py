# -*- coding: utf-8 -*-
"""
Common Functions Test Suite
===========================
Created on Mon May  2 17:01:39 2022

@author: thejasvi beleyur
"""
import unittest 
from pydatemm.common_funcs import *
from test_graphsynthesis import make_3_mergeable_triplets
from networkx.utils.misc import edges_equal

class TestFindUniqueGraphs(unittest.TestCase):
    
    def setUp(self):
        self.trips = make_3_mergeable_triplets()
    def test_simple(self):
        '''All entries unique. Input==Output'''
        out = find_unique_graphs([*self.trips])
        self.assertTrue(len(out)==3)
        common_graphs = set(out).intersection(set([*self.trips]))
        self.assertTrue(len(common_graphs)==3)
    def test_twotimesreps(self):
        double_repeat = [*self.trips, *self.trips]
        out = find_unique_graphs(double_repeat)
        common_graphs = set(out).intersection(set(list(self.trips)))
        self.assertTrue(len(common_graphs)==3)

class TestRemoveGraphsInPool(unittest.TestCase):
    def setUp(self):
        self.three_trips = make_3_mergeable_triplets()
    def test_simple(self):
        pruned = remove_graphs_in_pool([self.three_trips[0]], self.three_trips)
        self.assertEqual(set(self.three_trips[1:]), set(pruned))
    def test_empty(self):
        pruned = remove_graphs_in_pool([], self.three_trips)
        self.assertEqual(set(pruned), set(self.three_trips))
    def test_empty_pool(self):
        pruned = remove_graphs_in_pool(self.three_trips, self.three_trips)
        self.assertEqual(pruned, [])
        
        


if __name__ == '__main__':
    unittest.main()
        
        
        
    


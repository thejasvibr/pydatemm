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
from networkx.utils.misc import graphs_equal

class TestFindUniqueGraphs(unittest.TestCase):
    
    def setUp(self):
        self.trips = make_3_mergeable_triplets()
    def test_simple(self):
        '''All entries unique. Input==Output'''
        out = find_unique_graphs([*self.trips])
        matches = []
        for (each,every) in zip(out, [*self.trips]):
            matches.append(graphs_equal(each,every))
        self.assertTrue(np.all(matches))
    def test_twotimesreps(self):
        double_repeat = [*self.trips, *self.trips]
        out = find_unique_graphs(double_repeat)
        matches = []
        for (each,every) in zip(out, [*self.trips]):
            matches.append(graphs_equal(each,every))
        self.assertTrue(np.all(matches))


if __name__ == '__main__':
    unittest.main()
        
        
        
    


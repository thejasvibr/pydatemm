# -*- coding: utf-8 -*-
"""
Tests for Triple-Generation
===========================



Created on Thu May 19 10:43:13 2022
@author: Thejasvi Beleyur
"""
import unittest
from random import random
import networkx as nx
from pydatemm.triple_generation import * 

class TestGenConsTriples(unittest.TestCase):
    '''
    Tests if two consistent triples are recovered from a larger group
    of tdoa peaks which do not correspond to any of the triples. 
    '''

    def setUp(self):
        tdoa_peaks = {}
        # AB
        tdoa_peaks[(0,1)] = [(1e3, 2, 99, 99),
                             (1e3, -3, 99, 99),
                             (1e3, 12, 99, 99)]
        # BC
        tdoa_peaks[(1,2)] = [(2e3, 1, 99, 99),
                             (2e3, -55, 99, 99),
                             (2e3, 4, 99, 99)]
        # CA 
        tdoa_peaks[(2,0)] = [(3e3, -3, 99, 99),
                             (3e3, -122, 99, 99),
                             (3e3, -1, 99, 99)]
        self.tdoas = mirror_Pprime_kl(tdoa_peaks)
        self.kwargs = {'nchannels':3, 'twtm':0.5}

    def test_maiwo(self):
        consistent_triples = generate_consistent_triples(self.tdoas, **self.kwargs)
        
        trip1 = nx.DiGraph()
        trip1.add_edge(0,1, **{'tde':2, 'peak_score':5})
        trip1.add_edge(1,0, **{'tde':-2, 'peak_score':5})
        trip1.add_edge(1,2, **{'tde':1, 'peak_score':5})
        trip1.add_edge(2,1, **{'tde':-1, 'peak_score':5})
        trip1.add_edge(0,2, **{'tde':3, 'peak_score':5})
        trip1.add_edge(2,0, **{'tde':-3, 'peak_score':5})
        
        trip2 = nx.DiGraph()
        trip2.add_edge(0,1, **{'tde':-3, 'peak_score':5})
        trip2.add_edge(1,0, **{'tde':3, 'peak_score':5})
        trip2.add_edge(1,2, **{'tde':-4, 'peak_score':5})
        trip2.add_edge(2,1, **{'tde':4, 'peak_score':5})
        trip2.add_edge(0,2, **{'tde':1, 'peak_score':5})
        trip2.add_edge(2,0, **{'tde':-1, 'peak_score':5})
        
        graph_match = []
        for obtained, expected in zip(consistent_triples, [trip1, trip2]):
            graph_match.append(nx.is_isomorphic(obtained, expected))
        self.assertTrue(np.all(graph_match))
        
        
if __name__ == '__main__':
    unittest.main()
        


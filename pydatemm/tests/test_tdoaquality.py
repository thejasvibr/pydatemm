'''
Tests for TDOA Quality
======================
'''

import unittest
import networkx as nx
from pydatemm.tdoa_quality import * 

class TestTripleQuality(unittest.TestCase):
    
    def setUp(self):
        self.trip1 = nx.DiGraph()
        self.trip1.add_edge(0,1, **{'tde':2, 'peak_score':5})
        self.trip1.add_edge(1,0, **{'tde':-2, 'peak_score':5})
        self.trip1.add_edge(1,2, **{'tde':1, 'peak_score':5})
        self.trip1.add_edge(2,1, **{'tde':-1, 'peak_score':5})
        self.trip1.add_edge(0,2, **{'tde':3, 'peak_score':5})
        self.trip1.add_edge(2,0, **{'tde':-3, 'peak_score':5})
        self.twtm = 1e-3
        
    def test_triple(self):
        score = triplet_quality(self.trip1, twtm=self.twtm)
        self.assertEqual(score, 15)

if __name__ == '__main__':
    unittest.main()
        
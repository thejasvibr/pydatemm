# -*- coding: utf-8 -*-
"""
Tests for Main Graph Synthesis
==============================


Created on Wed May 18 08:34:11 2022
@author: thejasvi beleyur
"""

import unittest
import random
import networkx as nx
from networkx.utils.misc import edges_equal
from pydatemm.tdoa_objects import triple
from pydatemm.main_graph_synthesis import *


class TestFillUpTripleHols(unittest.TestCase):
    
    def test_00(self):
        self.assertTrue(False)

class TestGetUsableTDOAs(unittest.TestCase):
    def setUp(self):
        self.G = nx.generators.classic.complete_graph(5, create_using=nx.DiGraph)
    def remove_01(self):
        self.G.remove_edge(0,1)
        self.G.remove_edge(1,0)
    def gen_expected_graphs(self):
        G1 = nx.generators.classic.complete_graph(4, create_using=nx.DiGraph)
        G2 = nx.generators.classic.complete_graph(4, create_using=nx.DiGraph)
        G1 = nx.relabel.relabel_nodes(G1, {0:4},copy=False)
        G2 = nx.relabel_nodes(G2, {1:4}, copy=False)
        return [G2, G1]
    def test_simple(self):
        # remove one edge pair
        self.remove_01()
        obtained_tdoas = get_usable_TDOAs_from_graph(self.G)
        expected_tdoas = self.gen_expected_graphs()
        graph_match = []
        for (exp,obs) in zip(expected_tdoas, obtained_tdoas):
            graph_match.append(edges_equal(exp.edges, obs.edges))
        self.assertTrue(np.all(graph_match))
        
if __name__ == '__main__':
    unittest.main()

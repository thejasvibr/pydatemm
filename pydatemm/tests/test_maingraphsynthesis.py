# -*- coding: utf-8 -*-
"""
Tests for Main Graph Synthesis
==============================


Created on Wed May 18 08:34:11 2022
@author: thejasvi beleyur
"""

import unittest
import random
from pydatemm.tdoa_objects import triple
from pydatemm.main_graph_synthesis import *

class TestPruneObjects(unittest.TestCase):
    
    def setUp(self):
        self.triple_list = []
        for each in range(10):
            threenodes = tuple((random.randint(1,10) for each in range(3)))
            three_tdes = tuple(((random.random(), random.random()) for i in  range(3)))
            self.triple_list.append(triple(threenodes, *three_tdes))

    def test_remove1(self):
        '''Removes the first triple in the list'''
        pruned = prune_triple_pool(self.triple_list[0], self.triple_list, [])
        self.assertEqual(pruned, self.triple_list[1:])
    
    def test_remove_nonexistent(self):
        '''when you try to remove a triple that isn't in the pool'''
        threenodes = tuple((random.randint(11,20) for each in range(3)))
        three_tdes = tuple(((random.random(), random.random()) for i in  range(3)))
        madeup_triple = triple(threenodes, *three_tdes)
        pruned = prune_triple_pool(madeup_triple, self.triple_list, [])
        print(len(pruned))

if __name__ == '__main__':
    unittest.main()

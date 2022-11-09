# -*- coding: utf-8 -*-
"""
Tests for Graph Synthesis
=========================


Created on Wed May 18 08:34:11 2022
@author: thejasvi beleyur
    """
from itertools import combinations
import numpy as np 
import unittest
from pydatemm.compilation_utils import load_and_compile_with_own_flags
load_and_compile_with_own_flags()
import cppyy as cpy
Eigen = cpy.gbl.Eigen
cpp_vect = cpy.gbl.std.vector
cpp_set = cpy.gbl.std.set

class NodeIdentification(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.mm = Eigen.MatrixXd(3,3)
        self.mm[1,0] = 1
        self.mm[0,1] = 1
        self.expected_nodes = [0,1]
        
        self.nn = Eigen.MatrixXd(5,5)
        self.nn[1,0] = 1
        self.nn[0,1] = 1
        # add in an asymmetric component - just check that it still works
        self.nn[3,0] = 1
        self.expected_nn = [0,1,3]
        
        
    def test_simple(self):
        output = list(cpy.gbl.get_nodes(self.mm))
        self.assertTrue(output == self.expected_nodes)
    
    def test_next_simple(self):
        output = list(cpy.gbl.get_nodes(self.nn))
        self.assertTrue(output==self.expected_nn)

class GraphMerging(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.graph_a = Eigen.MatrixXd(4,4)
        self.graph_a[0,1] = 0.5; self.graph_a[1,0] = 0.5;
        self.graph_a[0,2] = -0.5; self.graph_a[2,0] = -0.5;
        
        self.graph_b = Eigen.MatrixXd(4,4)
        self.graph_b[0,1] = 0.5; self.graph_b[1,0] = 0.5;
        self.graph_b[0,3] = 0.25; self.graph_b[3,0] = 0.25;

        self.graph_c = Eigen.MatrixXd(4,4)
        self.graph_c[0,3] = 0.25; self.graph_c[3,0] = 0.25;
        self.graph_c[0,2] = -0.5; self.graph_c[2,0] = -0.5;
        
        self.all_cfls = []
        self.ccg_solution = set()
        for i, each in enumerate([self.graph_a, self.graph_b, self.graph_c]):
            self.all_cfls.append(each)
            self.ccg_solution.add(i)
        
    def test_merging(self):
        merged = cpy.gbl.combine_graphs(self.ccg_solution, self.all_cfls)
        expected = Eigen.MatrixXd(4,4)
        expected[0,1] = 0.5; expected[1,0] = 0.5; 
        expected[0,2] = -0.5; expected[2,0] = -0.5;
        expected[0,3] = 0.25; expected[3,0] = 0.25;
        values_equal = []
        for (i,j) in combinations(range(4),2):
            values_nan = np.logical_or(np.isnan(merged[i,j]), np.isnan(expected[i,j]))
            if not values_nan:
                values_equal.append(merged[i,j]==expected[i,j])

        self.assertTrue(np.all(values_equal))
        
    


if __name__ == '__main__':
    unittest.main()

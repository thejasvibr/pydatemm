'''
Tests for graph synthesis
'''
import unittest
from pydatemm.graph_synthesis import *

#%% This example triplet-> quadruplet merger is from Fig 11 of Scheuing & Yang 2008

klm = [(0,1,2), (5,00), (-1, 00), (-4,00)]
klo = [(0,1,3), (5,00), (-7,00), (2, 00)]
kmo = [(0,2,3), (4,00), (-6,00), (2,00)]

# m0 = make_triplet_graph(klm, 4)
# m1 = make_triplet_graph(klo, 4)
# m2 = make_triplet_graph(kmo, 4)
# sum_trips = add_three_triplets(m0,m1,m2)

#%% Test cases for build full toads 
# 4, 5, 20 channels


class TestProgressSeedIndex(unittest.TestCase):
    
    def setUp(self):
        self.current_ind = 8
        self.invalid_inds = [0,2,5,8,9]
        self.all_objects = list(range(29))
    
    def test_basic(self):
        progress_possible, next_ind = progress_seed_index(self.current_ind,
                                                          self.invalid_inds,
                                                          self.all_objects)
        self.assertTrue(progress_possible)
        self.assertEqual(next_ind, 10)
        #print(progress_possible, next_ind, self.all_objects[next_ind])        
    
    def test_no_more_valid(self):
        self.all_objects = [0,1,2,3,4,5,6,7,8]
        progress_possible, next_ind = progress_seed_index(self.current_ind,
                                                          self.invalid_inds,
                                                          self.all_objects)
        print(progress_possible, next_ind, )     
        

class TestMissingTriplets(unittest.TestCase):
    
    def setUp(self):
        self.graph = np.random.normal(0,1,25).reshape(5,5)
        self.graph[np.diag_indices(5)] = np.nan
        self.graph[2,1] = np.nan; self.graph[1,2] = np.nan

    def test_2_missing_weight(self):
        output = missing_triplets(self.graph)
        expected = [(0,1,2), (1,2,3), (1,2,4)]
        self.assertEqual(output, expected)

    def test_three_missing_weights(self):
        self.graph[1,3] = np.nan; self.graph[3,1] = np.nan
        output = missing_triplets(self.graph)
        expected = [(1,2,3)]
        self.assertEqual(output, expected)
    
if __name__ == '__main__':
    unittest.main()
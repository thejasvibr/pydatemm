'''
Tests for graph synthesis
'''
import unittest
import networkx as nx
from networkx.utils.misc import graphs_equal, edges_equal
from pydatemm.graph_synthesis import *

class TestTripleSorting(unittest.TestCase):
    
    def setUp(self):
        self.trip1 = nx.DiGraph()
        self.trip1.add_edge(0,1, **{'tde':2, 'peak_score':5})
        self.trip1.add_edge(1,0, **{'tde':-2, 'peak_score':5})
        self.trip1.add_edge(1,2, **{'tde':1, 'peak_score':5})
        self.trip1.add_edge(2,1, **{'tde':-1, 'peak_score':5})
        self.trip1.add_edge(0,2, **{'tde':3, 'peak_score':5})
        self.trip1.add_edge(2,0, **{'tde':-3, 'peak_score':5})
        
        self.trip2 = nx.DiGraph()
        self.trip2.add_edge(0,1, **{'tde':-3, 'peak_score':2})
        self.trip2.add_edge(1,0, **{'tde':3, 'peak_score':2})
        self.trip2.add_edge(1,2, **{'tde':-4, 'peak_score':1})
        self.trip2.add_edge(2,1, **{'tde':4, 'peak_score':1})
        self.trip2.add_edge(0,2, **{'tde':1, 'peak_score':1})
        self.trip2.add_edge(2,0, **{'tde':-1, 'peak_score':1})
        
        self.trip3 = nx.DiGraph()
        self.trip3.add_edge(0,1, **{'tde':-3.5, 'peak_score':2})
        self.trip3.add_edge(1,0, **{'tde':3.5, 'peak_score':2})
        self.trip3.add_edge(1,2, **{'tde':-4, 'peak_score':1})
        self.trip3.add_edge(2,1, **{'tde':4, 'peak_score':1})
        self.trip3.add_edge(0,2, **{'tde':1, 'peak_score':1})
        self.trip3.add_edge(2,0, **{'tde':-1, 'peak_score':1})
        
        self.triple_pool = [self.trip3, self.trip2, self.trip1]
        
        self.twtm = 0.6

    def test_check2triple_sorting(self):
        sorted_triples = sort_triples_by_quality(self.triple_pool[1:],
                                                 twtm=self.twtm)        
        self.assertEqual(sorted_triples, [self.trip1, self.trip2])

    def test_check3triple_sorting(self):
        sorted_triples = sorted_triples = sort_triples_by_quality(self.triple_pool,
                                                                  twtm=self.twtm)        
        self.assertEqual(sorted_triples, [self.trip1, self.trip2, self.trip3])

def make_3_mergeable_triplets():
    '''
    Example triplets taken from Fig. 11  Scheuing & Yang 2008
    '''
    trip1, trip2, trip3 = [nx.DiGraph() for each in range(3)]
    trip1.add_edge('k','m', **{'tde':4, 'peak_score':5})
    trip1.add_edge('m','k', **{'tde':-4, 'peak_score':5})
    trip1.add_edge('o','m', **{'tde':6, 'peak_score':5})
    trip1.add_edge('m','o', **{'tde':-6, 'peak_score':5})
    trip1.add_edge('o','k', **{'tde':2, 'peak_score':5})
    trip1.add_edge('k','o', **{'tde':-2, 'peak_score':5})
    
    trip2.add_edge('k','m', **{'tde':4, 'peak_score':5})
    trip2.add_edge('m','k', **{'tde':-4, 'peak_score':5})
    trip2.add_edge('m','l', **{'tde':1, 'peak_score':5})
    trip2.add_edge('l','m', **{'tde':-1, 'peak_score':5})
    trip2.add_edge('k','l', **{'tde':5, 'peak_score':5})
    trip2.add_edge('l','k', **{'tde':-5, 'peak_score':5})
    
    trip3.add_edge('k','l', **{'tde':5, 'peak_score':5})
    trip3.add_edge('l','k', **{'tde':-5, 'peak_score':5})
    trip3.add_edge('o','l', **{'tde':7, 'peak_score':5})
    trip3.add_edge('l','o', **{'tde':-7, 'peak_score':5})
    trip3.add_edge('o','k', **{'tde':2, 'peak_score':5})
    trip3.add_edge('k','o', **{'tde':-5, 'peak_score':5})
    return trip1, trip2, trip3

def created_expected_quad_fig11():
    out = nx.DiGraph()
    out.add_edge('k','l', **{'tde':5, 'peak_score':5})
    out.add_edge('l','k', **{'tde':-5, 'peak_score':5})
    out.add_edge('m','l', **{'tde':1, 'peak_score':5})
    out.add_edge('l','m', **{'tde':-1, 'peak_score':5})
    out.add_edge('o','m', **{'tde':6, 'peak_score':5})
    out.add_edge('m','o', **{'tde':-6, 'peak_score':5})
    out.add_edge('k','o', **{'tde':-2, 'peak_score':5})
    out.add_edge('o','k', **{'tde':2, 'peak_score':5})
    out.add_edge('k','m', **{'tde':4, 'peak_score':5})
    out.add_edge('m','k', **{'tde':-4, 'peak_score':5})
    out.add_edge('o','l', **{'tde':7, 'peak_score':5})
    out.add_edge('l','o', **{'tde':-7, 'peak_score':5})
    return out
    
    

class TestTripleTripletMerge(unittest.TestCase):
    def setUp(self):
        self.trips = make_3_mergeable_triplets()
    def test_positivecase(self):
        outcome = validate_triple_triplet_merge(self.trips[0], self.trips[1],
                                                self.trips[2])
        self.assertTrue(outcome)
    def test_negativecase(self):
        outcome = validate_triple_triplet_merge(self.trips[0], self.trips[2],
                                                self.trips[2])
        self.assertFalse(outcome)
        

class TestQuadGeneration(unittest.TestCase):
    '''Tests for triplet_to_quadruplet'''

    def setUp(self):
        self.trips = make_3_mergeable_triplets()
    def test_quad_creation(self):
        outcome_quad = make_quadruplet_graph(*self.trips)
        expected = created_expected_quad_fig11()
        expected_match = edges_equal(outcome_quad.edges, expected.edges)
        self.assertTrue(expected_match)

class TestGenQuadsFromSeedTriple(unittest.TestCase):
    '''Generating valid quadruples from a list of triples'''

    def setUp(self):
        self.trips = make_3_mergeable_triplets()
        trip_4 = self.trips[0].copy()
        trip_4 = nx.relabel_nodes(trip_4, {'o':'p'})
        trip_4.edges[('m','p')]['tde'] = 99
        trip_4.edges[('p','m')]['tde'] = -99
        
        trip_4.edges[('k','p')]['tde'] = 24
        trip_4.edges[('p','k')]['tde'] = -24
        self.sorted_triples = [*self.trips[1:], trip_4]

    def test_generate_one_quad(self):
        out_quads = generate_quads_from_seed_triple(self.trips[0],
                                                    self.sorted_triples)
        expected_quad = created_expected_quad_fig11()
        self.assertEqual(len(out_quads), 1)
        self.assertTrue(edges_equal(expected_quad.edges, out_quads[0].edges))

    def test_when_multiple_quads(self):
        self.sorted_triples = [*self.trips[1:], self.trips[-1]]
        out_quads = generate_quads_from_seed_triple(self.trips[0],
                                                    self.sorted_triples)
        expected_quad = created_expected_quad_fig11()
        #self.assertEqual(len(out_quads), 1)
        self.assertTrue(edges_equal(expected_quad.edges, out_quads[0].edges))



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

class TestMissingTriplets(unittest.TestCase):
    
    def setUp(self):
        '''Generate a fully connected graph, and then progressive remove
        some edges to test'''
        self.G = nx.generators.classic.complete_graph(4, create_using=nx.DiGraph)

    def remove_01(self):
        self.G.remove_edge(0,1)
        self.G.remove_edge(1,0)

    def remove_13(self):
        self.G.remove_edge(1,3)
        self.G.remove_edge(3,1)

    def test_1edge_missing(self):
        self.remove_01()
        output = missing_triplets(self.G)
        expected = set([(0,1,2), (0,1,3)])
        self.assertEqual(output, expected)

    def test_2edge_missing(self):
        self.remove_01()
        self.remove_13()
        output = missing_triplets(self.G)
        expected = expected = set([(0,1,2), (0,1,3), (0,1,3), (1,2,3)])
        self.assertEqual(output, expected)


if __name__ == '__main__':
    unittest.main()

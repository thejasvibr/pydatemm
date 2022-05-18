'''
TDOA objects
------------
Dataclasses for triplets, quadruples and stars. 

Triplets, quadruples and stars are consistent TDOA graphs with 3, 4, and >=5
nodes respectively. This module holds the dataclasses which represent the 
TDOAs. 

'''
from dataclasses import dataclass
import numpy as np 
from pydatemm.common_funcs import nancompare

@dataclass
class triple():
    '''
    Parameters
    ----------
    nodes : tuple
        Tuple with a,b,c nodes (channel #'s) e.g. (0,1,2)
    tde_ab, tde_bc, tde_ca: tuple
        Tuple which represents the time-difference estimate
        of each channel pair in that order, along with the 
        quality of that time-difference. e.g. for tde_ab
        the tuple may be (-5.5e-3, 29.3)
    
    Attributes
    ----------
    triple_id : str
        A concatenation of the nodes and time differences
    quality : float
        The triple quality score. The closer the sum of differences
        is to zero, the higher the score. See eqn. 23 Scheuing & Yang 2008

    See Also
    --------
    generate_consistent_triples
    '''
    nodes : tuple
    tde_ab: tuple
    tde_bc: tuple
    tde_ca: tuple
    
    def __post_init__(self):
        '''generate the triple ID'''
        self.triple_id = str(self.nodes)+'_'+str(self.tde_ab)+'_'+str(self.tde_bc)+'_'+str(self.tde_ca)
        self.quality = 0
        self.values = [self.nodes, self.tde_ab, self.tde_bc, self.tde_ca, self.quality]

    def __eq__(self, other):
        if not type(other) is type(self):
            raise NotImplementedError(f'{other.__class__} and a triple {self.__class__} cannot be compared')
        
        same_entries = other.values == self.values
        return same_entries

    def __repr__(self):
        textout = (f'nodes:{self.nodes}:tdes-ab,bc,ca:{self.tde_ab[0],self.tde_bc[0],self.tde_ca[0]}')        
        return (textout)

@dataclass
class quadruple():
    '''
    Parameters
    ----------
    nodes : tuple
        Tuple with a,b,c,d nodes (channel #'s) e.g. (0,1,2)
    graph : (Nmics,Nmics) np.array
        An incomplete graph with the TDEs for 4 mics filled out.

    Attributes
    ----------
    quadruple_id : str
        A concatenation of the nodes and time differences
    component_triples : list
        List of triplets that were used to make this quadruplet.
        Each triplet is a :code:`triplet` dataclass

    See Also
    --------
    pydatemm.triple_generation.triplet
    '''
    nodes : tuple
    graph : list # don't know how to hint at a np.array
    
    def __post_init__(self):
        '''generate the quadruple ID'''
        self.tdes = []
        self.quadruple_id = str(self.nodes)
        self.component_triples = []
    
    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            raise NotImplementedError(f'{other.__class__} and a {self.__class__} cannot be compared')
        same_nodes = self.nodes==other.nodes
        same_graphs = nancompare(self.graph, other.graph)
        return np.all([same_nodes, same_graphs])
    
    def is_complete_graph(self):
        ''''Checks that all possible TDOAs are filled out'''
        diagonals_nan = np.all(np.isnan(self.graph.diagonal()))
        num_nodes = len(self.nodes)
        total_exp_entries = int(num_nodes*(num_nodes-1))
        num_tdoas_match = np.sum(~np.isnan(self.graph))==total_exp_entries
        if np.logical_and(diagonals_nan, num_tdoas_match):
            return True
        else:
            return False

    def __repr__(self):
        #textout = (f'nodes:{self.nodes}:tdes-ab,bc,ca:{self.tde_ab[0],self.tde_bc[0],self.tde_ca[0]}')        
        return (f'{self.quadruple_id}, {str(self.graph)}')        


@dataclass
class star():
    '''
    A TDOA object with at least 5 nodes.

    Parameters
    ----------
    nodes : tuple
        Tuple with a,b,c,d,e... nodes (channel #'s) e.g. (0,1,2...6)
    graph : (Nmics,Nmics) np.array
        An incomplete graph with the TDEs for 4 mics filled out.

    Attributes
    ----------
    star_id : str
        A concatenation of the nodes and time differences
    component_quads, component_triples : list
        List of quads and triplets that were used to make this quadruplet.
        Each triplet is a :code:`triplet` dataclass
    graph :(nchannels,nchannels)  np.array
        

    See Also
    --------
    pydatemm.graph_synthesis.quads_to_star
    '''
    nodes : tuple
    graph : list # don't know how to hint at a np.array
    
    def __post_init__(self):
        '''generate the quadruple ID'''
        self.tdes = []
        self.star_id = str(self.nodes)
        self.component_quads = []
        self.component_triples = []

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            raise NotImplementedError(f'{other.__class__} and a star cannot be compared')
        same_nodes = self.nodes==other.nodes
        same_graphs = nancompare(self.graph, other.graph)
        return np.all([same_nodes, same_graphs])

    def __repr__(self):
        #textout = (f'nodes:{self.nodes}:tdes-ab,bc,ca:{self.tde_ab[0],self.tde_bc[0],self.tde_ca[0]}')        
        return (f'{self.star_id}, {str(self.graph)}')
    
    def is_complete_graph(self):
        ''''Checks that all possible TDOAs are filled out'''
        diagonals_nan = np.all(np.isnan(self.graph.diagonal()))
        num_offdiag = self.graph.size - self.graph.diagonal().size
        others_notnan = np.sum(~np.isnan(self.graph)) == num_offdiag
        if np.logical_and(diagonals_nan, others_notnan):
            return True
        else:
            return False
    
    def forms_complete_subgraph(self):
        '''Checks that there are consistent subgraphs present in the current
        graph. 
        '''
        
    
    def missing_nodes(self):
        return set(range(self.graph.shape[0]))-set(self.nodes)
        


# -*- coding: utf-8 -*-
"""
Build CCG
=========
Module containing functions which help in creating the compatibility-conflict
graph proposed in Kreissig & Yang 2013. 

"""
from itertools import combinations, product
import joblib
from joblib import Parallel, delayed
import numpy as np 
import networkx as nx

def make_fundamental_loops(nchannels):
    G = nx.complete_graph(nchannels)
    minspan_G = nx.minimum_spanning_tree(G)
    main_node = [0]
    co_tree = nx.complement(minspan_G)
    fundamental_loops = []
    for edge in co_tree.edges():
        fl_nodes = tuple(set(main_node + list(edge)))
        fundamental_loops.append(fl_nodes)
    return fundamental_loops

def make_edges_for_fundamental_loops(nchannels):
    '''
    Generates the minimum set of edges required to make all fundamental 
    loops for an nchannels graph. 
    
    Here the 'minimum' set means that only one edge is used all throughout
    i.e. (2,0) instead of both (2,0) and (0,2). 
    
    Parameters
    ----------
    nchannels : int

    Returns
    -------
    triple_definition : dict
        Key is fundamental loop nodes (e.g. (0,1,2)). Entry is a list 
        with tuples. Each tuple has 2 elements. The first is the 
        'polarity' of the weight, and the second is the edge name. 
        The final weight is polarity*TDE[edge name] - where TDE is 
        the matrix of all pairwise time difference estimates

    Examples
    --------
    >>> make_edges_for_fundamental_loops(4)
    >>> 
    {(0, 1, 2): [(1, (0, 1)), (1, (1, 2)), (1, (2, 0))],
     (0, 1, 3): [(1, (0, 1)), (1, (1, 3)), (1, (3, 0))],
     (0, 2, 3): [(-1, (2, 0)), (1, (2, 3)), (1, (3, 0))]}
    
    '''
    funda_loops = make_fundamental_loops(nchannels)
    triple_definition = {}
    for fun_loop in funda_loops:
        edges = make_triple_pairs(fun_loop)
        triple_definition[fun_loop] = []
        # if the edge (ab) is present but the 'other' way round (ba) - then 
        # reverse polarity. 
        for edge in edges:
            triple_definition[fun_loop].append(edge)
    return triple_definition

def make_all_fundaloops_from_tdemat(tdematrix):
    '''
    '''
    all_edges_fls = make_edges_for_fundamental_loops(tdematrix.shape[0])
    all_cfls = []
    for fundaloop, edges in all_edges_fls.items():
        print(fundaloop)
        this_cfl = nx.ordered.Graph()
        for e in sorted(edges):
            print(e)
            this_cfl.add_edge(e[0], e[1], tde=tdematrix[e[0],e[1]])
        all_cfls.append(this_cfl)
    return all_cfls

def check_for_one_common_edge(X,Y):
    '''Checks that X and Y have one common edge 
    with the same weight.
    X and Y are assumed to be undirected graphs!!
    '''
    X_edge_weights = [ X.edges()[i]['tde'] for i in X.edges]
    Y_edge_weights = [ Y.edges()[i]['tde'] for i in Y.edges]
    common_edge = set(Y_edge_weights).intersection(set(X_edge_weights))
    if len(common_edge)==1:
        return 1
    else:
        return -1

def ccg_definer(X,Y):
    '''
    Assesses the compatibility, conflict or lack of connection between 
    two triples.
    
    `When two consistent subgraphs have the same TDOAs on common
    edges, they are called compatible and we can combine them together
    to a larger graph containing both subgraphs. If the TDOAs on com-
    mon edges are different, the two subgraphs are in conflict and we
    are not allowed to combine them. The same applies when the two
    subgraphs have no common edges at all. ` (Kreissig-Yang ICASSP 2013)

    Parameters
    ----------
    X,Y : nx.DiGraph
        consistent Fundamental Loops (cFL) that need to be compared
        for their compatibility
    Returns
    -------
    relation : int
        1 means compatible, 0 means no common edges, -1 means conflict.

    '''
    xx = nx.intersection(X, Y)
    n_common_nodes = len(xx.nodes)
    if n_common_nodes >= 2:
        if n_common_nodes < 3:
            relation = check_for_one_common_edge(X, Y)
        else:
            # all nodes the same
            relation = -1
    else:
        relation = -1
    return relation

def make_ccg_matrix(cfls):
    '''
    Sped up version. Previous version had explicit assignment of i,j and j,i
    compatibilities.
    '''
        
    num_cfls = len(cfls)
    ccg = np.zeros((num_cfls, num_cfls), dtype='int32')
    cfl_ij = combinations(range(num_cfls), 2)
    for (i,j) in cfl_ij:
        trip1, trip2  = cfls[i], cfls[j]
        cc_out = ccg_definer(trip1, trip2)
        ccg[i,j] = cc_out
    ccg += ccg.T
    return ccg

def combine_compatible_triples(compatible_triples):
    combined = nx.compose_all(compatible_triples)
    nodesorted = nx.Graph()
    nodesorted.add_nodes_from(sorted(combined.nodes(data=True)))
    nodesorted.add_edges_from(combined.edges(data=True))
    return nodesorted
    

def make_triple_pairs(triple):
    pairs = combinations(sorted(triple),2)
    # reverse order to make them into j,i pairs from i,j pairs
    ji_pairs = list(map(lambda X: X[::-1], pairs))
    return ji_pairs


def get_compatibility(cfls, ij_combis):
    output = []
    for (i,j) in ij_combis:
        trip1, trip2  = cfls[i], cfls[j]
        cc_out = ccg_definer(trip1, trip2)
        output.append(cc_out)
    return output

def make_ccg_pll(cfls, **kwargs):
    '''Parallel version of make_ccg_matrix'''
    num_cores = kwargs.get('num_cores', joblib.cpu_count())
    num_cfls = len(cfls)
    cfl_ij_parts = [list(combinations(range(num_cfls), 2))[i::num_cores] for i in range(num_cores)]
    compatibility = Parallel(n_jobs=num_cores)(delayed(get_compatibility)(cfls, ij_parts)for ij_parts in cfl_ij_parts)
    ccg = np.zeros((num_cfls, num_cfls), dtype='int32')
    for (ij_parts, compat_ijparts) in zip(cfl_ij_parts, compatibility):
        for (i,j), (comp_val) in zip(ij_parts, compat_ijparts):
            ccg[i,j] = comp_val
    # make symmetric
    ccg += ccg.T
    return ccg

def make_consistent_fls(multich_tdes, **kwargs):
    '''

    Parameters
    ----------
    multich_tdes : TYPE
        DESCRIPTION.
    nchannels : int>0
    max_loop_residual : float>0, optional 
        Defaults to 1e-6

    Returns
    -------
    cFLs : list
        List with nx.DiGraphs of all consistent FLs
    '''
    max_loop_residual = kwargs.get('max_loop_residual', 1e-6)
    all_edges_fls = make_edges_for_fundamental_loops(kwargs['nchannels'])
    all_cfls = []
   
    for fundaloop, edges in all_edges_fls.items():
        #print(fundaloop)
        a,b,c = fundaloop
        ba_tdes = multich_tdes[(b,a)]
        ca_tdes = multich_tdes[(c,a)]
        cb_tdes = multich_tdes[(c,b)]
        abc_combinations = product(ba_tdes, ca_tdes, cb_tdes)
        for i, (tde1, tde2, tde3) in enumerate(abc_combinations):
            if abs(tde1[1]-tde2[1]+tde3[1]) < max_loop_residual:
                this_cfl = nx.ordered.Graph()
                for e, tde in zip(edges, [tde1, tde2, tde3]):
                    #print(e, tde)
                    this_cfl.add_edge(e[0], e[1], tde=tde[1])
                    all_cfls.append(this_cfl)
    return all_cfls
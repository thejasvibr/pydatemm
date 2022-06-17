# -*- coding: utf-8 -*-
"""
Generate synthetic data to troubleshoot Kreissig Yang algorithm
===============================================================

Created on Thu Jun 16 08:40:49 2022

@author: theja
"""
from pydatemm.localisation import spiesberger_wahlberg_solution
from pydatemm.triple_generation import make_channel_pairs_from_triple
from itertools import combinations, product, chain
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import networkx as nx
from scipy.spatial import distance_matrix
from combineall import combine_all, format_combineall
seednum = 78464 # 8221, 82319, 78464
np.random.seed(seednum) # what works np.random.seed(82310)
vsound = 340
#%%
def plot_graph_w_labels(graph, curr_ax):
    pos = nx.circular_layout(graph)
    nx.draw_circular(graph, with_labels=True)
    weight_labels = {}
    for e in graph.edges():
        try:
            weight_labels[e] = np.around(graph.edges[e]['tde']*1e3, 3)
        except KeyError:
            weight_labels[e] = np.around(graph.edges[e]['weight'], 3)
    nx.draw_networkx_edge_labels(graph, pos,edge_labels=weight_labels)

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
    existing_edges = []
    triple_definition = {}
    for fun_loop in funda_loops:
        edges = make_triple_pairs(fun_loop)
        triple_definition[fun_loop] = []
        # if the edge (ab) is present but the 'other' way round (ba) - then 
        # reverse polarity. 
        for edge in edges:
            # if edge_present_with_reversed_polarity(edge, existing_edges):
            #     polarity = -1
            #     edge = edge[::-1]
            # else:
            #     polarity = 1
            #     existing_edges.append(edge)
            triple_definition[fun_loop].append(edge)
    return triple_definition

def edge_present_with_reversed_polarity(edge, existing_edges):
    '''
    False if edge nodes are not at all present.
    False if present but in same order as existing_edges.
    True if it's present in reversed order (edge=ab, but ba in existing_edges)
    '''
    edges_wo_considering_order = list(map(lambda X: set(X), existing_edges))
    if set(edge) in edges_wo_considering_order:
        
        edge_location = edges_wo_considering_order.index(set(edge))
        existing_edge = existing_edges[edge_location]
        # same order as previously seen edge
        if tuple(existing_edge) == edge:
            return False
        # opposite order as previously seen edge
        elif tuple(existing_edge) == edge[::-1]:
            return True
        else:
            raise ValueError('Soemthing wrong has happened here: {edge}, {existing_edges}')
    else:
        # if the edge isn't at all in the list of existing edges
        return False

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

def weight_sum(triple):
    return np.sum([triple.edges()[e]['tde'] for e in triple.edges()])


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
    xx = nx.intersection(X,Y)
    n_common_nodes = len(xx.nodes)
    if n_common_nodes >= 2:
        if n_common_nodes==3:
            relation = -1
        else:
            relation = check_for_one_common_edge(X,Y)
    else:
        relation = 0
    return relation

def make_ccg_matrix(cfls):
        
    num_cfls = len(cfls)
    ccg = np.zeros((num_cfls, num_cfls))
    cfl_ij = combinations(range(num_cfls), 2)
    for (i,j) in cfl_ij:
        trip1, trip2  = cfls[i], cfls[j]
        cc_out = ccg_definer(trip1, trip2)
        ccg[i,j] = cc_out; ccg[j,i] = cc_out
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

def get_graph_weights(graph):
    edge_weights = {}
    for e in graph.edges():
        edge_weights[e] = graph.edges()[e]['tde']
    return edge_weights



if __name__ == '__main__':
    array_geom = pd.read_csv('../pydatemm/tests/scheuing-yang-2008_micpositions.csv').to_numpy()
    array_geom = array_geom[:5,:]
    
    nchannels = array_geom.shape[0]
    source1 = np.array([1,2,3])
    source2 = np.array([5,0.5,-2])
    
    def mic2source(sourcexyz, arraygeom):
        mic_source= distance_matrix(np.vstack((sourcexyz, arraygeom)),
                                     np.vstack((sourcexyz, arraygeom)))[1:,0]
        return mic_source
    
    mic2sources = [mic2source(each, array_geom) for each in [source1, source2]]
    mic2source1, mic2source2 = mic2sources
    
    deltaR1 = np.zeros((nchannels, nchannels))
    deltaR2 = deltaR1.copy()
    for i,j in product(range(nchannels), range(nchannels)):
        deltaR1[i,j] = (mic2source1[j]-mic2source1[i])/vsound
        deltaR2[i,j] = (mic2source2[j]-mic2source2[i])/vsound
    #%%
    # Make the cfls now:
    cfls_s12 = [make_all_fundaloops_from_tdemat(deltaR) for deltaR in [deltaR1, deltaR2]]
    ccg_s12 = [make_ccg_matrix(cfls_s) for cfls_s in cfls_s12]
    qq1 = combine_all(ccg_s12[0], set(range(nchannels)), set([]), set([]))
    qq2 = combine_all(ccg_s12[1], set(range(nchannels)), set([]), set([]))
    # And now let's combine the two cfls sets together and see how well it all
    # works
    cfls_combined = list(chain(*cfls_s12))
    ccg_combined = make_ccg_matrix(cfls_combined)
    qq_combined = combine_all(ccg_combined, set(range(len(ccg_combined))), set([]), set([]))    
    comp_cfls = format_combineall(qq_combined)
    #%%
    all_channel_pairs = combinations(range(nchannels),2)
    part_R1 = np.zeros(deltaR1.shape)
    for (i,j) in all_channel_pairs:
        part_R1[j,i] = deltaR1[j,i]
    g_parts1 = nx.from_numpy_array(part_R1)
    plt.figure()
    plot_graph_w_labels(g_parts1, plt.gca())
    #%%
    g_s1 = nx.from_numpy_array(deltaR1*1e3)
    g_s2 = nx.from_numpy_array(deltaR2*1e3)

    plt.figure()
    plot_graph_w_labels(g_s1, plt.gca())
    plt.title('Original S1')
    #%%
    # Compose. 
    cfl1 = [cfls_combined[each] for each in comp_cfls[0]]
    s1_composed = combine_compatible_triples(cfl1)
    s1c_tde = nx.to_numpy_array(s1_composed, weight='tde')
    print(spiesberger_wahlberg_solution(array_geom, s1c_tde[1:,0]*340))
    #%%
    # Now let's try to combine the compatible triples. 
    #qq_s1, qq_s2 = comp_cfls
    #trips_s1 = [cfls_combined[each] for each in qq_s1]
    plt.figure()
    for i,subp in enumerate(range(511, 516)):
        plt.subplot(subp)
        plot_graph_w_labels(cfls_s12[0][i], plt.gca())
    
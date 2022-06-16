# -*- coding: utf-8 -*-
"""
Implementing Kreissig Yang 2012 (ICASSP)
========================================
KY2012 suggest an exhaustive and sequential search to combine all
compatible consistent triples. Given there are X fundamental
loops (l) for a given system, KY2012 suggests an iterative search
through all loops. 

Basically, given K instances of all l_{x} (x=1...X), then search
through all K^2 l1xl2 pairs. Of the successful l12 merges, then search
for all l12xl3 merges, and then l123 x l4 ..etc. The idea is that there
aren't going to be very many merges that are valid and so many of the 
merges will not be successful. 

Created on Wed Jun 15 09:56:33 2022

@author: theja
"""
import networkx as nx
import tqdm 
def iterative_merge_loops(cfl_container):
    ''' This isn't perfect. 
    Right now it only does A-B, AC, AD, and in case there is no match 
    somehwere it moves right onto B. 
    Also doesn't take care of the case when merged graphs returns an empty list.
    '''
    fls = list(cfl_container.keys()) # all fundamental loops
    
    merged_graphs = cfl_container[fls[0]]
    for funda_loop in tqdm.tqdm(fls[1:]):
        mergeable_pairs = determine_compatibility(merged_graphs,
                                                  cfl_container[funda_loop])
        if len(mergeable_pairs)>0:
            # get indices of mergeable pairs
            merged_graphs = merge_all_compatible_pairs(mergeable_pairs,
                                                        merged_graphs, 
                                                        cfl_container[funda_loop])
    if merged_graphs != cfl_container[fls[0]]:
        return merged_graphs
    else:
        return []

def check_for_one_common_edge(X,Y):
    '''Checks that X and Y have one common edge 
    with the same weight.
    '''
    X_edge_weights = [ (i, X.edges()[i]['tde']) for i in X.edges]
    Y_edge_weights = [ (i, Y.edges()[i]['tde']) for i in Y.edges]
    common_edge = set(Y_edge_weights).intersection(set(X_edge_weights))
    return len(common_edge)==1

def compatible(X,Y):
    xx = nx.intersection(X,Y)    
    two_common_nodes = len(xx.nodes) == 2
    if two_common_nodes:
        return check_for_one_common_edge(X,Y)

def determine_compatibility(merged_graphs, candidate_graphs):
    compatible_pairs = []
    for i, each in enumerate(merged_graphs):
        for j, every in enumerate(candidate_graphs):
            if compatible(each, every):
                compatible_pairs.append((i,j))
    return compatible_pairs

def merge_all_compatible_pairs(mergeable_pairs, graphs_a, graphs_b):
    merged_graphs = []
    for (i,j) in mergeable_pairs:
        merged_graphs.append(nx.compose(graphs_a[i], graphs_b[j]))
    return merged_graphs
                
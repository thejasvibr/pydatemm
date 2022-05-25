'''
Common functions
================
A bunch of general purpose purpose functions
'''

import numpy as np
import networkx as nx
from networkx.utils.misc import edges_equal
nona = lambda X: X[~np.isnan(X)]
# removes all np.nans and checks that all entries are the same
nancompare = lambda X,Y: np.all(nona(X)==nona(Y))

def find_unique_graphs(graphs):
    '''
    Finds repeated graph objects in a list and eliminates them.

    Parameters
    ----------
    graphs : list
        List with networkx DiGraph's.

    Returns
    -------
    unique_graphs : list
        If all TOAD graphs in the input :code:`tuplets` are unique, then 
        the input object is returned as is. If not, then a list with a 
        subset of TOAD objects, :code:`unique_graphs` is returned.
    '''
    unique_graphs = list(set(graphs))
    return unique_graphs

def calc_range(X):    
    return np.nanmax(X)-np.nanmin(X)

def merge_graphs(graphs):
    '''
    Combine graphs with consistent edges into a consensus graph. 
    Each graph is a symmetrical NodexNode array. All numerical entries are 
    checked if they are identical - or unique along the 3rd dimension. 
    
    Parameters
    ----------
    graphs : list
        List with graphs. 
    
    TODO
    ----
    * IMPLEMENT CONSISTENCY CHECKING. ALONG DEPTH ALL VALUES MUST
    BE THE SAME OR ONE VALUE + OTHERS BEING NANS!!
    
    Notes
    -----
    If the range of values (max-min) along the 'depth' is 0, then all values
    are the same. However, even if there's only one value and the others are
    NaNs too the range will be zero!
    '''
    # check that input graphs are symmetric!
    for i,each in enumerate(graphs):
        symm_graph = check_opp_symmetric(each)
        if not symm_graph:
            raise ValueError(f'The {i}th graph in input list is not symmetric. Cannot proceed.')
    multi_graphs = np.dstack(graphs)
    
    # check that all graphs in the stack are compatible
    diff_graph = np.apply_along_axis(calc_range, 2, multi_graphs)
    all_zero = np.all(0 ==  diff_graph[~np.isnan(diff_graph)])
    if all_zero:
        merged_graph = np.nanmax(multi_graphs,2)
        # check that the final merged_graph entries are the
        # same as 
        return merged_graph
    else:
        raise ValueError('Unmergeable graphs. Some entries are incompatible')

def check_opp_symmetric(input_mat):
    '''
    Checks that an input array is symmetric with opposite signs
    across the diagonal, i.e. :math:`M[i,j]==-M[j,i]`
    '''
    symm_check = []
    for i in range(input_mat.shape[0]):
        for j in range(input_mat.shape[1]):
            same_indices = i==j
            if not same_indices:
                if not np.all(np.isnan([input_mat[i,j],input_mat[j,i]])):
                    equality_check = input_mat[i,j]==-input_mat[j,i]
                    symm_check.append(equality_check)
    return np.all(symm_check)

def remove_graphs_in_pool(objs_to_remove, object_pool):
    '''
    Parameters
    ----------
    objs_to_remove : list
        List with tdoa objects to be removed
    object_pool : list
        List with the whole pool of tdoa objects.

    Returns
    -------
    filtered_object_pools : list
        List with a subset of tdoa objects - without objs_to_remove
    
    Notes
    -----
    If the input has a 'foreign' object (that's not in the object pool),
    then an unaltered copy of the object_pool is returned.
    '''
    pool_indices = []
    for each in objs_to_remove:
        for i, every in enumerate(object_pool):
            if edges_equal(each.edges, every.edges):
                pool_indices.append(i)
    indices_to_keep = set(range(len(object_pool))) - set(pool_indices)
    filtered_object_pools = [object_pool[j] for j in indices_to_keep]
    return filtered_object_pools

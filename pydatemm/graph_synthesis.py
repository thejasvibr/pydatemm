'''
Graph synthesis
===============
Builds up larger graphs using approximately consistent triplets.

Implements S2-8 of Table 1. using an array type representation. Each
graph is represented as a Nchannels x Nchannels array. Triples may have
missing entries, as well as 

Reference
---------
* Scheuing & Yang 2008, ICASSP

TODO
----
* Make the data structure more intuitive. e.g. setup a triplet dataclass
with attributes like `tdoas`, `graph`, or other stuff. This will allow
following through the workflow and identifies the origin of components etc.
'''
import numpy as np
from pydatemm.common_funcs import nona



def quads_to_star():
    '''

    Returns
    -------
    None.

    '''



def triplet_to_quadruplet(triplet1, triplet2, triplet3, **kwargs):
    '''
    Parameters
    ----------
    triplet1,triplet2,triplet3 : list
        List with tuples representing a triplet. 
    nchannels: int
        Number of channels in original input audio.
    Returns
    -------
    quadruplet : (4,4) np.array
        Array filled with np.nan's if no merge possible, otherwise
        a valid quadruplet graph.
        
    '''
    graph1, graph2, graph3 = [make_triplet_graph(each, **kwargs) for each in [triplet1, triplet2, triplet3]]
    merge_to_quad = validate_triple_triplet_merge(graph1, graph2, graph3)
    if merge_to_quad:
        quadruplet = make_quadruplet_graph(graph1, graph2, graph3)
    else:
        quadruplet = np.empty([kwargs['nchannels']]*2)
        quadruplet[:,:] = np.nan
    return quadruplet

def make_quadruplet_graph(graph1, graph2, graph3):
    '''
    '''
    stacked = np.dstack((graph1, graph2, graph3))
    quadruplet_graph = np.nanmax(stacked, 2)
    return quadruplet_graph

def validate_triple_triplet_merge(graph1, graph2, graph3):
    '''
    Validates the 'mergeability' of the three triplets into a quadruplet.
    Works on the principle that when all triplets are mergeable, then the
    the :code:`sum_stacked` graph will reflect that two sides of each node
    are 'added twice'.

    Parameters
    ----------
    graph1, graph2, graph3 : (nchannel,nchannel) np.array
        Where nchannel is the # of audio channels.

    Returns
    -------
    mergeable : bool
        True if the three triplets make a valid quadruple.
    '''
    sum_stack = add_three_triplets(graph1, graph2, graph3)
    lower_tri = np.tril_indices(sum_stack.shape[0]) # lower triangle
    expected_repeats = [2,2,1]
    repeat_match = []
    for graph in [graph1, graph2, graph3]:
        repeat_matrix = sum_stack/graph
        value_repeats = sorted(nona(repeat_matrix[lower_tri]), reverse=True)
        match_observation = np.all(value_repeats==expected_repeats)
        repeat_match.append(match_observation)
    if np.all(repeat_match):
        mergeable = True
    else:
        mergeable = False
    return mergeable

def make_triplet_graph(triplet, **kwargs): 
    '''
    Converts triplet from a list of tuples into an nxnchannel array with 
    TDOAs as node to node weights. 

    Parameters
    ----------
    triplet
    nchannels
    
    Returns
    -------
    graph : (nchannels,nchannels) np.array
        A diagonal-symmetric array with tdes across triplet pairs. 
        Missing entries have np.nan assigned.
    
    Notes
    -----
    The output :code:`graph` array should be read left to right. Let's say we
    have a triplet `klm` in a 4 channel system. The graph is an 'incomplete'
    4x4 array.
    
    .. code-block:: python
        
       |  | k  | l | m| o|
       |k | nan| 5 | 4| o|
       |l | -5 |nan|-1| o|
       |m | -4 |nan| m| o|
       |o | nan|nan| m| o|

    The 'direction' to read is in the row->column. e.g. if you want
    the TDOA for `k->l`, then it is 5, `l->m` is -1, etc.
    '''
    trip_name,_,_,_ = triplet
    ch1, ch2, ch3 = trip_name
    pair1, pair2, pair3 = (ch1,ch2), (ch2, ch3), (ch3, ch1)
    tde1, tde2, tde3 = [each[0] for each in triplet[1:]]
    graph = np.empty((kwargs['nchannels'], kwargs['nchannels']))
    graph[:,:] = np.nan
    for tde, pair in zip([tde1, tde2, tde3], [pair1, pair2, pair3]):
        graph[pair] = tde
        graph[tuple(reversed(pair))] = -tde
    return graph

def add_three_triplets(graph1, graph2, graph3):
    '''
    '''
    stacked = np.dstack((graph1, graph2, graph3))
    sum_stacked = np.nansum(stacked, 2)    
    return sum_stacked

def sort_triples_by_quality(triples, **kwargs):
    '''
    Implements Step 2 in Table 1.
    
    Parameters
    ---------
    triples : list
        List with sublists. Each sublist has 4 entries
        [(triplename), td_ab, td_bc, td_ca]
    twtm : float
        Tolerance width of triple match in seconds. 
    
    Returns
    -------
    sorted_triples : list
        The input :code:`triples` list sorted in descending order of triplet
        quality.

    See Also
    --------
    triplet_quality
    '''
    triples_quality = []
    for each in triples:
        quality = triplet_quality(each, **kwargs)
        triples_quality.append(quality)
    # thanks to https://www.adamsmith.haus/python/answers/how-to-sort-a-list-based-on-another-list-in-python
    zipped_sorted_lists = sorted(zip(triples_quality, triples), reverse=True)
    sorted_triples = [element for _, element in zipped_sorted_lists]
    return sorted_triples

def triplet_quality(triplet, **kwargs):
    '''
    Calculates triplet quality score- which is the product of the 
    TFTM output and the sum of individual TDOA qualities.
    This metric is defined in eqn. 23
    '''
    triplet_name, t12, t23, t31 = triplet
    tdoa_quality_sum = t12[1] + t23[1] + t31[1]
    tdoa_tftm_score = gamma_tftm(t12[0],t23[0],t31[0], **kwargs)
    quality = tdoa_tftm_score*tdoa_quality_sum
    return quality 

def gamma_tftm(tdoa_ab, tdoa_bc, tdoa_ca,**kwargs):
    '''
    Calculates the tolerance width of triple match.
    
    Parameters
    ----------
    tdoa_ab,tdoa_bc,tdoa_ca: float
    twtm : float
        Tolerance width of triple match
    Returns
    -------
    twtm_out : float
        Final score
    '''
    residual = tdoa_ab + tdoa_bc + tdoa_ca
    twtm = kwargs['twtm']
    if abs(residual) < 0.5*twtm:
        twtm_out = 1 - (abs(residual))/(0.5*twtm)
    elif abs(residual)>= 0.5*twtm:
        twtm_out = 0
    return twtm_out

if __name__ == '__main__':
    from simdata import simulate_1source_and1reflector_general
    from pydatemm.timediffestim import *
    from pydatemm.raster_matching import multichannel_raster_matcher
    from pydatemm.triple_generation import mirror_Pprime_kl, generate_consistent_triples
    from itertools import permutations
    np.random.seed(82310)
    nchannels = 5
    audio, distmat, arraygeom, _ = simulate_1source_and1reflector_general(nmics=nchannels)
    fs = 192000
    
    kwargs = {'twrm': 50/fs,
              'array_geom':arraygeom,
              'twtm': 192/fs,
              'nchannels':nchannels,
              'fs':192000}
    multich_cc = generate_multich_crosscorr(audio, use_gcc=True)
    multich_ac = generate_multich_autocorr(audio)
    cc_peaks = get_multich_tdoas(multich_cc, min_height=2, fs=192000)
    multiaa = get_multich_aa_tdes(multich_ac, fs=192000,
                                  min_height=2) 
   
    tdoas_rm = multichannel_raster_matcher(cc_peaks, multiaa,
                                           **kwargs)
    tdoas_mirrored = mirror_Pprime_kl(tdoas_rm)
    true_tdoas = {}
    for chpair, _ in cc_peaks.items():
        ch1, ch2 = chpair
        # get true tDOA
        tdoa = (distmat[0,ch1]-distmat[0,ch2])/340
        true_tdoas[chpair] = tdoa
    #%%
    # Now get all approx consistent triples
    consistent_triples = generate_consistent_triples(tdoas_mirrored, **kwargs)
    sorted_triples = sort_triples_by_quality(consistent_triples, **kwargs)  

    #%% choose triplet with highest quality score and then begin to build out
    best_triple = sorted_triples[0]
    best_trip_name = best_triple[0]
    # Remove all triplets that have the current triplet name!
    all_triple_pool = list(filter(lambda X: X[0]!=best_trip_name, sorted_triples))
    # Keep only those tripley which with 2 common nodes
    two_nodes_common = lambda X,Y : len(set(X).intersection(set(Y)))==2
    valid_triple_pool = []
    for each in all_triple_pool:
        trip_name = each[0]
        if two_nodes_common(trip_name, best_trip_name):
            valid_triple_pool.append(each)
    #%%
    # Generate all possible pairs from the valid_triple_pool (even thought some of
    # these don't make sense!)
    possible_pairs = list(combinations(range(len(valid_triple_pool)), 2))
    valid_quads = []
    for (triple2, triple3) in possible_pairs:
        out = triplet_to_quadruplet(best_triple, valid_triple_pool[triple2],
                                    valid_triple_pool[triple3], **kwargs)
        # if all values are np.nan
        if np.all(np.isnan(out)):
            pass
        else:
            valid_quads.append(out)
    # Are there >4 channels? If yes, then try to fuse these quads into a start
    def nancompare(X,Y):
        return np.all(X[~np.isnan(X)]==Y[~np.isnan(Y)])
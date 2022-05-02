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
'''
import numpy as np
from pydatemm.common_funcs import nona, find_unique_Ntuplets, merge_graphs
from pydatemm.common_funcs import remove_objects_in_pool
from pydatemm.tdoa_objects import quadruple, star
from itertools import combinations, product
from copy  import deepcopy
#%%

def fill_up_triple_hole_in_star(star, triple_pool):
    '''
    Tries to fill a missing triple in a star's graph.

    Parameters
    ----------
    star : dataclass
        tdoa_objects.star instance
    triple_pool : list
        List with consistent triples sorted by quality

    Returns
    -------
    filled : bool
        Whether the star could be filled up or not.
    filled_star : dataclass
        A copy of the input :code:`star` object. If fillled, then 
        with a different graph structure, else an identical copy.
    '''
    filled_star = deepcopy(star)
    filled = False
    if not star.is_complete_graph():
        num_missing_entries = np.sum(np.isnan(star.graph))-star.graph.shape[0]
        num_missing_entries *= 0.5 # consider only one of the lower/upper triangles
        if num_missing_entries==3:
            target_triple_nodes = missing_triplet_id(star.graph)
            potential_triples = list(filter(lambda X: X.nodes==target_triple_nodes,
                                                                triple_pool))
            if len(potential_triples)>0:
                filler_triple = potential_triples[0]
                triple_graph = make_triplet_graph(filler_triple, **kwargs)
                filled_star.graph = merge_graphs([filled_star.graph, triple_graph])
                filled_star.component_triples.append(filler_triple)
                filled = True
        else:
            filled = False

    return filled, filled_star

def get_component_triples(tdoa_object):
    '''
    Generates all unique triples that went into construction of 
    this object
    
    Parameters
    ----------
    tdoa_object : quadruple or star object
    
    Returns
    -------
    all_triples_in_obj : list
        List with triples
    '''
    triple_in_quads = []
    try:
        component_quads = tdoa_object.component_quads
        for each in component_quads:
            for every in each.component_triples:
                triple_in_quads.append(every)
        triple_in_quads = find_unique_Ntuplets(triple_in_quads)
    except:
        pass
    
    all_triplets_w_repeats = triple_in_quads+tdoa_object.component_triples
    all_triples_in_obj = find_unique_Ntuplets(all_triplets_w_repeats)
    return all_triples_in_obj       


def group_into_nodeset_combis(quads_w_common_trip):
    '''
    Creates unique combinations of nodesets with 
    overlapping triples. Sometimes there can be two 
    quads with the same nodeset, and here we make sure to 
    create unique combinations. 
    
    Parameters
    ----------
    quads_w_common_trip : list
        List with unique quads having a common triple
    Returns
    -------
    nodeset_combis : list
        List with tuples for each nodeset combination.
        Each nodeset combination is a tuple with X quad objects. 
    '''
    node_sets = {}
    for each in quads_w_common_trip:
        if each.nodes not in node_sets.keys():
            node_sets[each.nodes] = []
        node_sets[each.nodes].append(each)
    nodeset_combis = list(product(*node_sets.values()))
    return nodeset_combis
    
def merge_quads_to_star(quads):
    '''
    quads : list of quad objs
    '''
    # merge graphs
    merged_graph = merge_graphs([each.graph for each in quads])

    star_nodesets = [each.nodes for each in quads]
    star_nodes = tuple(np.unique(np.array(star_nodesets).flatten()))
    startuplet = star(star_nodes, merged_graph)
    startuplet.component_quads = quads
    return startuplet

def missing_triplet_id(graph):
    '''finds missing weights, and the appropriate triplets'''
    # search all possible triples in sorted consistent triples list
    missing_entries = np.argwhere(np.isnan(graph))
    # remove diagonal elements
    offdiagonal_inds = [each for each in missing_entries if not each[0]==each[1]]
    target_triple_nodes = tuple(set(np.array(offdiagonal_inds).flatten()))
    if len(target_triple_nodes)==3:
        return target_triple_nodes
    else:
        # make all possible triplet combinations
        raise NotImplementedError('>1 triplet missing not implemented')

def validate_multi_quad_merge(quads):
    '''

    Parameters
    ----------
    quads: list
        A list of quadruple objects.

    Returns
    -------
    mergeable : bool
    
    TODO
    ----
    * IMplement more checking beyond the simple check that there are 3 nodes common
    e.g. check that the TDEs are common too. Or is this all just overkill?
    '''
    
    # check that they have three nodes in common
    common_nodes = tuple(set.intersection(*[set(each.nodes) for each in quads]))
    if len(common_nodes)!=3:
        mergeable = False
        return mergeable
    # # now check that the shared triples also have the same tdes
    # quad1_common_triple = list(filter(lambda X: X.nodes==common_nodes, quad1.component_triples))[0]
    # quad2_common_triple = list(filter(lambda X: X.nodes==common_nodes, quad2.component_triples))[0]
    
    # if quad1_common_triple==quad2_common_triple:
    #     mergeable = True
    # else:
    #     mergeable = False
    # return mergeable
    

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
    
    nodes = tuple(set(sorted(triplet1.nodes+triplet2.nodes+triplet3.nodes)))
    quad_candidate = quadruple(nodes,[])
    if merge_to_quad:
        quadruplet_graph = make_quadruplet_graph(graph1, graph2, graph3)
    else:    
        quadruplet_graph = np.empty([kwargs['nchannels']]*2)
        quadruplet_graph[:,:] = np.nan
    quad_candidate.graph = quadruplet_graph
    quad_candidate.component_triples = [triplet1, triplet2, triplet3]
    return quad_candidate

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
    ch1, ch2, ch3 = triplet.nodes
    pair1, pair2, pair3 = (ch1,ch2), (ch2, ch3), (ch3, ch1)
    tde1, tde2, tde3 = [each[0] for each in [triplet.tde_ab, triplet.tde_bc, triplet.tde_ca]]
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
    triples : triple dataclass
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
        each.quality = quality
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
    t12, t23, t31 = triplet.tde_ab, triplet.tde_bc, triplet.tde_ca
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

def generate_quads_from_seed_triple(seed_triple, sorted_triples):
    '''
    '''

    # Keep only those tripley which with 2 common nodes
    two_nodes_common = lambda X,Y : len(set(X).intersection(set(Y)))==2
    valid_triple_pool = []
    for each in sorted_triples:
        if two_nodes_common(each.nodes, seed_triple.nodes):
            valid_triple_pool.append(each)
    # Generate all possible pairs from the valid_triple_pool (even thought some of
    # these don't make sense!)
    possible_pairs = list(combinations(range(len(valid_triple_pool)), 2))
    valid_quads = []
    for (triple2, triple3) in possible_pairs:
        out = triplet_to_quadruplet(best_triple,
                                    valid_triple_pool[triple2],
                                    valid_triple_pool[triple3], **kwargs)
        # if all values are np.nan
        if np.all(np.isnan(out.graph)):
            pass
        else:
            valid_quads.append(out)
    # get all unique quadruples
    unique_quads = find_unique_Ntuplets(valid_quads)
    return unique_quads

if __name__ == '__main__':
    from simdata import simulate_1source_and1reflector_general
    from pydatemm.timediffestim import *
    from pydatemm.raster_matching import multichannel_raster_matcher
    from pydatemm.triple_generation import mirror_Pprime_kl, generate_consistent_triples
    from itertools import permutations
    np.random.seed(82310)
    nchannels = 7
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
    consistent_triples = generate_consistent_triples(tdoas_mirrored, **kwargs)
    sorted_triples_full = sort_triples_by_quality(consistent_triples, **kwargs)  
    used_triple_pool = deepcopy(sorted_triples_full)
    #%% choose triplet with highest quality score and then begin to build out
    potential_source_tdoas = []
    seed_triples_present = True
    rounds = 0
    while seed_triples_present:
        best_triple = used_triple_pool[0]
        potential_quads = generate_quads_from_seed_triple(best_triple, used_triple_pool)
        nodeset_combis = group_into_nodeset_combis(potential_quads)
        for each_combi in nodeset_combis:
            if len(each_combi)>0:
                star1 = merge_quads_to_star(each_combi)
                success, ff_star  = fill_up_triple_hole_in_star(star1, used_triple_pool)     
                if success:
                    # remove all component triplets that went into making the filled star
                    comp_triples = get_component_triples(ff_star)   
                    used_triple_pool = remove_objects_in_pool(comp_triples, used_triple_pool)
                    # append the filled star to a list of potential sources
                    potential_source_tdoas.append(ff_star)
                else:
                    # move onto the next nodeset combination
                    pass
        rounds += 1 
        if rounds>= len(used_triple_pool):
            break
        

    #%%   
    # #%% Actuall gra
    # source1_graph = np.zeros([kwargs['nchannels']]*2)
    # for i in range(kwargs['nchannels']):
    #     for j in range(kwargs['nchannels']):
    #         diff_dist = j-i
    #         source1_graph[i,j] = diff_dist/340
    
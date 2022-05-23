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
* Make the code robust to starter seed :| ...
'''
import networkx as nx
import numpy as np
from pydatemm.common_funcs import nona, find_unique_graphs, merge_graphs
from pydatemm.common_funcs import remove_objects_in_pool
from pydatemm.tdoa_objects import quadruple, star
from pydatemm.tdoa_quality import triplet_quality, gamma_tftm
from tqdm import tqdm
from itertools import combinations, product
from copy  import deepcopy
#%%

def progress_seed_index(current_index, invalid_inds, object_pool):
    '''
    Generates the next seed-triple index to begin graph building attempts.
    Signals the end of graph-building if there are only 2 valid triples after
    removal of invalid indices. 

    Parameters
    ----------
    current_index : int
    invalid_inds : list
        List with indices of invalid objects
    object_pool : list
        List with all objects in it

    Returns
    -------
    seed_triples_present : bool
        If True, then graph building continues. Otherwise it stops. 
    seed_triple_index : int/np.nan
        Next seed-triple index to begin graph building attempts.
        If there are no valid indices then the output is np.nan
    '''
    valid_indices = set(range(len(object_pool))) - set(invalid_inds) - set([current_index])
    valid_indices = np.array(list(valid_indices), dtype=np.int64).flatten()

    if len(valid_indices)>2:
        greater_than_current = np.argwhere(valid_indices>current_index)
        if sum(greater_than_current)>1:
            next_index = int(valid_indices[greater_than_current][0])
            seed_triples_present = True
        else:
            next_index = np.nan
            seed_triples_present = False
    else:
        next_index = np.nan
        seed_triples_present = False
    return seed_triples_present, next_index

def find_triple_indices(target_triples, triple_list):
    '''
    Parameters
    ----------
    target_triples : list 
        List of triples whose index locations need to be determined
    triples_list : list
        'Reference' list of triples.
    Returns
    -------
    List with integers indices
    '''
    triple_indices = []
    for each in target_triples:
        for i, obj in enumerate(triple_list):
            if not obj==each:
                pass
            else:
                triple_indices.append(i)
    return list(set(triple_indices))

def fill_up_triple_hole_in_star(star_in, triple_pool, **kwargs):
    '''
    Tries to fill a missing triple in a star's graph.

    Parameters
    ----------
    star_in : dataclass
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
    filled_star = deepcopy(star_in)
    if not star_in.is_complete_graph():
        num_missing_entries = np.sum(np.isnan(star_in.graph))-star_in.graph.shape[0]
        num_missing_entries *= 0.5 # consider only one of the lower/upper triangles
    else: 
        raise FilledGraphError('Complete graph input - cannot fill up triple holes')  
    target_triple_nodes = missing_triplets(star_in.graph)

    for each in target_triple_nodes:
        potential_triples = list(filter(lambda X: X.nodes==each,
                                                            triple_pool))
        for filler_triple in potential_triples:
            triple_graph = make_triplet_graph(filler_triple, **kwargs)
            try:
                one_triple_filled = merge_graphs([filled_star.graph,
                                                  triple_graph])
                # if matching triple found - then move onto searching for the next
                # unique triple nodeset
                filled_star.graph = one_triple_filled
                filled_star.component_triples.append(filler_triple)
                break
            except:
                # what if the merge can't happen with this candidate triple??
                # then move onto next potential triple
                pass
    filled =  filled_star.is_complete_graph()
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
        triple_in_quads = find_unique_graphs(triple_in_quads)
    except:
        pass
    
    all_triplets_w_repeats = triple_in_quads+tdoa_object.component_triples
    all_triples_in_obj = find_unique_graphs(all_triplets_w_repeats)
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
    Notes
    -----
    This function systematically generates all possible combinations that 
    could be formed from multiple quadruplets. 
    Let's say we start with k1-l1-m1-p1, k2-l2-m2-p2, l0-m0-p0-q0
    We have two 'node sets' - klmp and lmpq, with two unique quadruplets
    in the klmp nodeset.
    
    A nodeset-combi is a list with the combinations of quadruplets. For this
    example case we'll get [k1-l1-m1-p1,  l0-m0-p0-q0] and [k2-l2-m2-p2, l0-m0-p0-q0]
    '''
    node_sets = {}
    for each in quads_w_common_trip:
        nodes = tuple(each.nodes)
        try:
            if nodes not in node_sets.keys():
                node_sets[nodes] = []
            node_sets[nodes].append(each)
        except TypeError:
            node_sets[nodes] = []
    # generate all possible combinations across node_sets A,B,C
    nodeset_combis = list(product(*node_sets.values()))
    return nodeset_combis

def merge_quads_to_star(quads):
    '''
    Checks if all quads in a list have a common triple, and if so, then 
    puts them into a star (>4 node) graph.

    Parameters
    ----------
    quads : list
        List of nx.DiGraph objects with 4 nodes.

    Returns
    -------
    startuplet : nx.DiGraph
        Star graph with >4 nodes.

    '''
    # merge graphs
    merged_graph = merge_graphs([each.graph for each in quads])

    star_nodesets = [each.nodes for each in quads]
    star_nodes = tuple(np.unique(np.array(star_nodesets).flatten()))
    startuplet = star(star_nodes, merged_graph)
    startuplet.component_quads = quads
    return startuplet

def missing_triplets(graph):
    '''
    Finds missing weights, and the appropriate triplets

    Parameters
    ----------
    graph : np.array
    
    Returns
    -------
    target_triples : list 
        List with tuples with missing tuple nodes.
    '''
    # search all possible triples in sorted consistent triples list
    missing_entries = np.argwhere(np.isnan(graph))
     # remove diagonal elements
    offdiagonal_inds = [each for each in missing_entries if not each[0]==each[1]]
    target_triple_nodes = tuple(set(np.array(offdiagonal_inds).flatten()))
    if len(target_triple_nodes)>=3:
        target_triples = list(combinations(target_triple_nodes, 3))
    elif len(target_triple_nodes)==2:
        # generate all possible combinations of 
        other_nodes = set(range(graph.shape[0]))-set(target_triple_nodes)
        target_triples = [ sorted([*target_triple_nodes, each]) for each in other_nodes]
        target_triples = [tuple(each) for each in target_triples]
    else:
        raise ValueError('Unable to find target triple nodes!')
    return target_triples

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
    

def triplets_to_quadruplet(triplet1, triplet2, triplet3):
    '''
    Parameters
    ----------
    triplet1,triplet2,triplet3 : nx.DiGraph
        3 node graphs

    Returns
    -------
    quadruplet : nx.DiGraph
        If merger is possible. Output is a 4 node graph made from the merger of
        the 3 input triplet graphs. If not possible, then the output is 
        an empty nx.DiGraph
    '''
    merge_to_quad = validate_triple_triplet_merge(triplet1, triplet2, 
                                                  triplet3)

    if merge_to_quad:
        quadruplet = make_quadruplet_graph(triplet1, triplet2, 
                                                      triplet3)    
        return quadruplet
    else:
        return nx.DiGraph()

def make_quadruplet_graph(graph1, graph2, graph3):
    '''
    '''
    abc = nx.compose(graph1, graph2)
    abcd = nx.compose(abc, graph3)
    return abcd

def validate_triple_triplet_merge(graph1, graph2, graph3):
    '''
    Validates the 'mergeability' of the three triplets into a quadruplet.
    Works on the principle that when all triplets are mergeable, then the
    each pair (a,b), (b,c), (c,a) will have exactly 2 common edges and 
    2 common nodes.

    Parameters
    ----------
    graph1, graph2, graph3 : nx.DiGraph

    Returns
    -------
    mergeable : bool
        True if the three triplets make a valid quadruple.
    '''
    pairs = combinations([graph1, graph2, graph3], 2)
    matched = []
    for (x,y) in pairs:
        matched.append(two_common_nodes_and_edges(x, y))
    mergeable = np.all(matched)
    return mergeable

def two_common_nodes_and_edges(X, Y):
    xy = nx.intersection(X, Y)
    return np.all([len(xy.nodes)==2, len(xy.edges)==2])

def two_nodes_common(X,Y):
    common_nodes = set(X.nodes).intersection(Y.nodes)
    if len(common_nodes)==2:
        return True
    else:
        return False


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
    triples : list 
        List with consistent triple graph objects.
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
    argsorted = np.argsort(triples_quality)[::-1]# descending order
    sorted_triples = [triples[index] for index in argsorted]
    return sorted_triples

def generate_quads_from_seed_triple(seed_triple, sorted_triples):
    '''
    Parameters
    ----------
    seed_triple : nx.DiGraph
        A graph with 3 nodes
    sorted_triples : list
        List with 3 node nx.DiGraphs 

    Returns
    -------
    unique_quads : list
        List with unique quadruple objects.
    '''
    # Keep only those triples which with 2 common nodes
    
    valid_triple_pool = []
    for each in sorted_triples:
        if two_nodes_common(each, seed_triple):
            valid_triple_pool.append(each)
    # Generate all possible pairs from the valid_triple_pool (even thought some of
    # these don't make sense!)
    possible_pairs = list(combinations(valid_triple_pool, 2))
    if not len(possible_pairs) >= 1:
        return []

    valid_quads = []
    for (triple2, triple3) in possible_pairs:
        out = triplets_to_quadruplet(seed_triple,
                                    triple2,
                                    triple3)
        # if all values are np.nan
        if len(out.nodes)==4:
            valid_quads.append(out)
        else:
            pass
    # get all unique quadruples
    unique_quads = find_unique_graphs(valid_quads)
    return unique_quads

class FilledGraphError(ValueError):
    def __init__(self, errormsg):
        print(errormsg)

if __name__ == '__main__':
    #%%
    from simdata import simulate_1source_and1reflector_general
    from pydatemm.timediffestim import *
    from pydatemm.raster_matching import multichannel_raster_matcher
    from pydatemm.triple_generation import mirror_Pprime_kl, generate_consistent_triples
    from pydatemm.tdoa_quality import residual_tdoa_error as ncap
    from pydatemm.simdata import make_chirp
    import pyroomacoustics as pra
    import soundfile as sf
    #%load_ext line_profiler
    from itertools import permutations
    seednum = 8221 # 8221, 82319, 78464
    np.random.seed(seednum) # what works np.random.seed(82310)
    
    #audio, distmat, arraygeom, source_reflect = simulate_1source_and1reflector_general(nmics=nchannels)
    #%%
    nchannels = 7
    fs = 192000
    kwargs = {'twrm': 50/fs,
              'twtm': 192/fs,
              'nchannels':nchannels,
              'fs':fs}
    room_dim = [5,5,5]
    
# We invert Sabine's formula to obtain the parameters for the ISM simulator
    rt60 = 0.5
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    
    room = pra.ShoeBox(room_dim, fs=kwargs['fs'],
                       materials=pra.Material(0.5), max_order=0)
    #mic_locs = np.random.normal(0,2,3*kwargs['nchannels']).reshape(3,nchannels)
    array_geom = np.abs(np.random.normal(0,2,3*nchannels).reshape(3,nchannels))
    kwargs['array_geom'] = array_geom.T
    room.add_microphone_array(array_geom)
    
    # add one source
    pbk_signals = make_chirp()
    source_positions = [[3,2,1], [2,4,1]]
    for i,each in enumerate(source_positions):
        room.add_source(position=each, signal=pbk_signals, delay=i*0.005)
    room.compute_rir()
    room.simulate()

    # plt.figure()
    # plt.specgram(room.mic_array.signals[1,:], Fs=fs)
    audio = room.mic_array.signals.T
    sf.write('pyroom_audio.wav', audio, fs)
    
    #%%
    
    
    multich_cc = generate_multich_crosscorr(audio, use_gcc=True)
    multich_ac = generate_multich_autocorr(audio)
    cc_peaks = get_multich_tdoas(multich_cc, min_height=1, fs=192000)
    multiaa = get_multich_aa_tdes(multich_ac, fs=192000,
                                  min_height=0.2) 
    
    tdoas_rm = multichannel_raster_matcher(cc_peaks, multiaa,
                                           **kwargs)
    tdoas_mirrored = mirror_Pprime_kl(tdoas_rm)
    
    consistent_triples = generate_consistent_triples(tdoas_mirrored, **kwargs)
    sorted_triples_full = sort_triples_by_quality(consistent_triples, **kwargs)  
    #used_triple_pool = deepcopy(sorted_triples_full)
    print(f'seed: {seednum}, len-sorted-trips{len(sorted_triples_full)}')
    #%%
    sorted_triples_part = deepcopy(sorted_triples_full)
    trippool, pot_tdoas = build_full_tdoas(sorted_triples_part)
    #%lprun -f build_full_tdoas build_full_tdoas(sorted_triples_part)
    #%% Let's try to localise the sources from each of the sound sources
    from pydatemm.localisation import spiesberger_wahlberg_solution
    all_sources = []
    all_ncap = []
    for i,each in enumerate(pot_tdoas):
        d_0 = each.graph[1:,0]*340
        try:
            sources = spiesberger_wahlberg_solution(kwargs['array_geom'],  d=d_0)
            print(sources)
            all_sources.append(sources)
            all_ncap.append(ncap(each, sources, kwargs['array_geom']))
        except:
            all_sources.append(np.nan)
            all_ncap.append(np.nan)
    #%%
    yy = [1,2,3,4,5]

    print('z')
    for i in yy:
        print(i)
        if i == 3:
            break
        else:
            print('jjj')
        print('yy')
    
# -*- coding: utf-8 -*-
"""
Graph synthesis execution
=========================
Module which performs the actual graph building given a set of triples ordered
by their quality. 

Created on Tue May 17 11:24:01 2022
@author: Thejasvi Beleyur
"""
from copy import deepcopy
from pydatemm.graph_synthesis import *
from pydatemm.common_funcs import remove_graphs_in_pool_pll, remove_graphs_in_pool_setstyle
from networkx.algorithms.clique import find_cliques
#%%
def assemble_tdoa_graphs(sorted_triples, **kwargs):
    '''

    Parameters
    ----------
    sorted_triples : list
        List with nx.DiGraphs with 3 nodes
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    tdoa_candidates : list
        List with nx.DiGraphs of >= 4 nodes
    '''
    pruned_triple_pool = deepcopy(sorted_triples)    
    tdoa_candidates = []
    enough_seed_triples = True
    roundnum = 0
    while enough_seed_triples:
        seed_triple = pruned_triple_pool[0]
        # steps S4-S7 in the paper
        print('...making stars')
        tdoa_sources = make_stars(seed_triple, pruned_triple_pool, **kwargs)
        print('...star making done...')
        if len(tdoa_sources)>=1:
            print('sources present')
            for tdoas in tdoa_sources:
                tdoa_candidates.append(tdoas)
        # remove all triples used to build tdoa graphs
        print('pruning...')
        pruned_triple_pool = prune_triple_pool(seed_triple,
                                               pruned_triple_pool,
                                               tdoa_sources)
        print('pruned...')
        enough_seed_triples = check_for_enough_seed_triples(pruned_triple_pool)
        print('checking enough seed triples')
        roundnum +=1 
        if len(pruned_triple_pool) % 5 < 3:
            print('AAAAAAAAAAAAAA',len(pruned_triple_pool))
        #print(len(pruned_triple_pool), roundnum)
    return tdoa_candidates

def check_for_enough_seed_triples(triple_pool):
    if len(triple_pool)<4:
        return False
    else:
        return True

def prune_triple_pool(seed_triple, triple_pool, tdoas):
    '''
    Parameters
    ----------
    seed_triple : nx.DiGraph
        3 node graph
    triple_pool : list
        List with triples (3 node nx.DiGraphs)
    tdoas : list
        List with >=5 node nx.DiGraphs

    Returns
    -------
    pruned_pool : list
        Subset of input triple pool without the seed_triple and 
        component triples of all the tdoas
    '''
    # get all component triples in each of the sources
    source_triples = []
    if len(tdoas)>0:
        for each in tdoas:
            triples_in_source = get_component_triples(each)
            for every in triples_in_source:
                source_triples.append(every)
        # remove component triples from triple_pool
        source_triples.append(seed_triple)
        unique_source_triples = find_unique_graphs(source_triples)
        pruned_pool = remove_graphs_in_pool_setstyle(unique_source_triples, triple_pool)
    else:
        pruned_pool = remove_graphs_in_pool_setstyle([seed_triple], triple_pool)
    return pruned_pool 

def make_stars(seed_triple, triple_pool, **kwargs):
    '''
    Generates filled stars or quadruples starting from a seed triple. 

    Parameters
    ----------
    sorted_triples : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    filled_stars : list
        List with quadruples or stars with >=5 nodes. If no 
        star can be made, then an empty list is returned. 

    TODO
    ----
    * What if there are only 4 channels?
    '''
    if kwargs['nchannels']<=4:
        raise NotImplementedError(f"{kwargs['nchannels']} channel case not handled")
    # assemble groups of 3 triples with klm to make multiple quadruples
    quadruplets = generate_quads_from_seed_triple(seed_triple, triple_pool)
    if not len(quadruplets)>=1:
        return []
    quadruplet_sets = group_into_nodeset_combis(quadruplets)
    # join quadruples to form stars
    stars = [merge_quads_to_star(each) for each in quadruplet_sets]
    # try to fill the 'holes' in the stars
    filled_stars = []
    for this_star in stars:
        try:
            filled, filled_star = fill_up_triple_hole_in_star(this_star,
                                                              triple_pool,
                                                              **kwargs)
            if filled:
                filled_stars.append(filled_star)
            else:
                # if not filled then find the maximum clique with >= 4  nodes
                # and add it into filled_stars
                usable_tdoa_graphs = get_usable_TDOAs_from_graph(filled_star)
                for each in usable_tdoa_graphs:
                    filled_stars.append(each)
        except FilledGraphError:
            filled_stars.append(this_star)
    return filled_stars 

def get_usable_TDOAs_from_graph(partial_star):
    largest_cliques = find_cliques(partial_star.to_undirected())
    usable_TDOAS = []
    for each in largest_cliques:
        if len(each)>=4:
            usable_TDOAS.append(partial_star.subgraph(each))
    return usable_TDOAS

if __name__ == '__main__':
    #%%
    from simdata import simulate_1source_and1reflector_general
    from pydatemm.timediffestim import *
    from pydatemm.raster_matching import multichannel_raster_matcher
    from pydatemm.triple_generation import mirror_Pprime_kl, generate_consistent_triples
    from pydatemm.graph_synthesis import sort_triples_by_quality
    from pydatemm.tdoa_quality import residual_tdoa_error as ncap
    from pydatemm.simdata import make_chirp
    import pyroomacoustics as pra
    import pandas as pd
    import soundfile as sf
    import time
    
    from itertools import permutations
    print('starting sim audio...')
    seednum = 78464 # 8221, 82319, 78464
    np.random.seed(seednum) # what works np.random.seed(82310)
    array_geom = pd.read_csv('tests/scheuing-yang-2008_micpositions.csv').to_numpy()
    # reduce the # of channels 
    array_geom = array_geom[:5,:]
    
    nchannels = array_geom.shape[0]
    fs = 96000
    paper_twrm = 7/fs
    paper_twtm = 9/fs
    kwargs = {'twrm': paper_twrm,
              'twtm': paper_twtm,
              'nchannels':nchannels,
              'fs':fs}
    room_dim = [4,2,2]

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    
    rt60 = 0.1
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=kwargs['fs'],
                       materials=pra.Material(0.5), max_order=0)
    #mic_locs = np.random.normal(0,2,3*kwargs['nchannels']).reshape(3,nchannels)
    # array_geom = np.abs(np.random.normal(0,1,3*nchannels).reshape(3,nchannels))
    
    kwargs['array_geom'] = array_geom
    room.add_microphone_array(array_geom.T)
    
    # add one source
    pbk_signals = [make_chirp(chirp_durn=0.05, start_freq=80000, end_freq=50000)*0.5,
                   make_chirp(chirp_durn=0.05, start_freq=40000, end_freq=10000)*0.5]
    source1 = [1.67,1.66,0.71]
    source2 = [2.72,0.65,1.25]
    sources = np.vstack((source1, source2))
    source_positions = [source1, source2]
    for i,each in enumerate(source_positions):
        room.add_source(position=each, signal=pbk_signals[i], delay=i*0.02)
    room.compute_rir()
    room.simulate()

    plt.figure()
    plt.specgram(room.mic_array.signals[1,:], Fs=fs)
    audio = room.mic_array.signals.T
    sf.write('pyroom_audio.wav', audio, fs)
    print('done w sim audio...')
    #%%
    print('starting cc and acc...')
    multich_cc = generate_multich_crosscorr(audio, use_gcc=True)
    multich_ac = generate_multich_autocorr(audio)
    cc_peaks = get_multich_tdoas(multich_cc, min_height=10, fs=192000)
    multiaa = get_multich_aa_tdes(multich_ac, fs=192000,
                                  min_height=3) 
    print('raster matching...')
    tdoas_rm = multichannel_raster_matcher(cc_peaks, multiaa,
                                           **kwargs)
    tdoas_mirrored = mirror_Pprime_kl(tdoas_rm)    
    print('triple generation')
    start = time.time()
    consistent_triples = generate_consistent_triples(tdoas_mirrored, **kwargs)
    print(f'time taken generating: {time.time()-start}')
    sorted_triples_full = sort_triples_by_quality(consistent_triples, **kwargs)
    triple_quality = [triplet_quality(each, **kwargs) for each in sorted_triples_full]
    quality_thresh = np.nanpercentile(triple_quality,[0])
    triples_goodq = []
    for (quality, triple) in zip(triple_quality, sorted_triples_full):
        if quality>quality_thresh:
            triples_goodq.append(triple)

    print(f'time taken sorting: {time.time()-start}')
    print(f'seed: {seednum}, len-sorted-trips{len(sorted_triples_full)}, nonzero quality: {len(triples_goodq)}')
    #%%
    #used_triple_pool = deepcopy(sorted_triples_full)
    #one_star = make_stars(sorted_triples_full[0], sorted_triples_full[:30], **kwargs)
    print('miaow')
    # #%%
    # #%load_ext line_profiler
    # #%lprun -f prune_triple_pool assemble_tdoa_graphs(triples_goodq[:20],  **kwargs)
    tdoas = assemble_tdoa_graphs(triples_goodq[:20],  **kwargs)
    # actual_tdoas = [each for each in tdoas if len(each.nodes)>0]
    # #%%
    # from pydatemm.localisation import spiesberger_wahlberg_solution
    # from pydatemm.tdoa_quality import residual_tdoa_error
    # def track_from_tdoa(tdoa_nx, **kwargs):
    #     vsound = kwargs.get('vsound', 340)
    #     d_0 = nx.to_numpy_array(tdoa_nx, weight='tde')[1:,0]*vsound
    #     #d_0 = nx.to_numpy_array(tdoa_nx, weight='tde')[0,1:]*vsound
    #     array_geom_part = kwargs['array_geom'][list(tdoa_nx.nodes),:]
    #     try:
    #         source = spiesberger_wahlberg_solution(array_geom_part,  d_0)
    #     except:
    #         source = np.nan
    #     return source, array_geom_part
    # #%%
    # all_sources = []
    # ncap = []
    # for i, each in enumerate(actual_tdoas):
    #     source, part_array = track_from_tdoa(each, **kwargs)
    #     if not np.sum(np.isnan(source))>0:
    #         # if euclidean(source, source2)<0.3:
    #         #     print(i,'good source')
    #         all_sources.append(source)
    #         ncap_source = residual_tdoa_error(each, source, part_array)
    #         ncap.append(ncap_source)
    # all_sources = np.array(all_sources)
    # #%%
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(all_sources[:,0], all_sources[:,1], '*')
    # plt.plot(source2[0], source2[1], 'r^')
    # plt.plot(source1[0], source1[1], 'g^')
    # plt.xlabel('x', labelpad=-5.5);plt.ylabel('y')
    # plt.subplot(212)
    # plt.plot(all_sources[:,1], all_sources[:,2], '*')
    # plt.plot(source2[1], source2[2], 'r^')
    # plt.plot(source1[1], source1[2], 'g^')
    # plt.xlabel('z');plt.ylabel('y')
    # #%%
    # plt.figure()
    # plt.hist(ncap)
    #%%
    # #sorted_triples_part = deepcopy(sorted_triples_full)
    # tdoa_sources = assemble_tdoa_graphs(sorted_triples_full, **kwargs)
    # #trippool, pot_tdoas = build_full_tdoas(sorted_triples_part)
    # 
    # all_sources = []
    # all_ncap = []
    # for i,each in enumerate(one_star):
    #     d_0 = nx.to_numpy_array(each,weight='tde')[1:,0]*340
    #     array_geom = kwargs['array_geom'][list(each.nodes),:]
    #     try:
    #         sources = spiesberger_wahlberg_solution(array_geom,  d_0)
    #         all_sources.append(sources)
    #         all_ncap.append(ncap(nx.to_numpy_array(each,weight='tde'), sources, array_geom))
    #     except:
    #         all_sources.append(np.nan)
    #         all_ncap.append(np.nan)    
    # #%%
    # complete = [i.is_complete_graph() for i in tdoa_sources]
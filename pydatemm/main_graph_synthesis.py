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
%load_ext line_profiler
#%%
def assemble_tdoa_graphs(sorted_triples, **kwargs):
    '''

    Parameters
    ----------
    sorted_triples : list
        List with triples
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    tdoa_candidates : list
        List with quadruples, stars, etc. 
    '''
    pruned_triple_pool = deepcopy(sorted_triples)    
    tdoa_candidates = []
    enough_seed_triples = True
    roundnum = 0
    while enough_seed_triples:
        seed_triple = pruned_triple_pool[0]
        # steps S4-S7 in the paper
        tdoa_sources = make_stars(seed_triple, pruned_triple_pool, **kwargs)
        if len(tdoa_sources)>1:
            #print('sources present')
            for tdoas in tdoa_sources:
                tdoa_candidates.append(tdoas)
            # remove all triples used to build tdoa graphs
        pruned_triple_pool = prune_triple_pool(seed_triple,
                                               pruned_triple_pool,
                                               tdoa_sources)
        enough_seed_triples = check_for_enough_seed_triples(pruned_triple_pool)
        roundnum +=1 
        if len(pruned_triple_pool) % 20 < 3:
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
    seed_triple : tdoa_objects.triple class
    triple_pool : list
        List with triples
    tdoas : list
        List with >=5 nodes

    Returns
    -------
    pruned_pool : list
        Subset of input triple pool without the seed_triple and 
        component triples of all the tdoas
    '''
    # remove the seed_triple from triple_pool
    pruned_pool = remove_objects_in_pool([seed_triple], triple_pool)
    # get all component triples in each of the sources
    source_triples = []
    if len(tdoas)>0:
        for each in tdoas:
            triples_in_source = get_component_triples(each)
            for every in triples_in_source:
                source_triples.append(every)
        # remove component triples from triple_pool
        pruned_pool = remove_objects_in_pool(source_triples, pruned_pool)
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
    quadruplets = generate_quads_from_seed_triple(seed_triple, triple_pool,
                                                              **kwargs)
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
                                                              triple_pool, **kwargs)
            if filled:
                filled_stars.append(filled_star)
        except FilledGraphError:
            filled_stars.append(filled_star)
    return filled_stars 

if __name__ == '__main__':
    #%%
    from simdata import simulate_1source_and1reflector_general
    from pydatemm.timediffestim import *
    from pydatemm.raster_matching import multichannel_raster_matcher
    from pydatemm.triple_generation import mirror_Pprime_kl, generate_consistent_triples
    from pydatemm.tdoa_quality import residual_tdoa_error as ncap
    from pydatemm.simdata import make_chirp
    import pyroomacoustics as pra
    import pandas as pd
    import soundfile as sf
    #%load_ext line_profiler
    from itertools import permutations
    seednum = 8221 # 8221, 82319, 78464
    np.random.seed(seednum) # what works np.random.seed(82310)
    array_geom = pd.read_csv('tests/scheuing-yang-2008_micpositions.csv').to_numpy()

    nchannels = array_geom.shape[0]
    fs = 192000
    kwargs = {'twrm': 50/fs,
              'twtm': 192/fs,
              'nchannels':nchannels,
              'fs':fs}
    room_dim = [4,2,2]
    
    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    
    rt60 = 0.3
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=kwargs['fs'],
                       materials=pra.Material(0.5), max_order=1)
    #mic_locs = np.random.normal(0,2,3*kwargs['nchannels']).reshape(3,nchannels)
    # array_geom = np.abs(np.random.normal(0,1,3*nchannels).reshape(3,nchannels))
    
    kwargs['array_geom'] = array_geom
    room.add_microphone_array(array_geom.T)
    
    # add one source
    pbk_signals = [make_chirp(chirp_durn=0.025, start_freq=50000)*0.5,
                   make_chirp(chirp_durn=0.05)*0.5]
    source_positions = [[1.67,1.66,0.71], [2.72,0.65,1.25]]
    for i,each in enumerate(source_positions):
        room.add_source(position=each, signal=pbk_signals[i])
    room.compute_rir()
    room.simulate()

    # plt.figure()
    # plt.specgram(room.mic_array.signals[1,:], Fs=fs)
    audio = room.mic_array.signals.T
    sf.write('pyroom_audio.wav', audio, fs)
    
    #%%
    multich_cc = generate_multich_crosscorr(audio, use_gcc=True)
    multich_ac = generate_multich_autocorr(audio)
    cc_peaks = get_multich_tdoas(multich_cc, min_height=17, fs=192000)
    multiaa = get_multich_aa_tdes(multich_ac, fs=192000,
                                  min_height=8) 
    
    tdoas_rm = multichannel_raster_matcher(cc_peaks, multiaa,
                                           **kwargs)
    tdoas_mirrored = mirror_Pprime_kl(tdoas_rm)    
    consistent_triples = generate_consistent_triples(tdoas_mirrored, **kwargs)
    sorted_triples_full = sort_triples_by_quality(consistent_triples, **kwargs)  
    #used_triple_pool = deepcopy(sorted_triples_full)
    print(f'seed: {seednum}, len-sorted-trips{len(sorted_triples_full)}')
    #%%
    %load_ext line_profiler
    %lprun -f fill_up_triple_hole_in_star make_stars(sorted_triples_full[0], sorted_triples_full[:100],q**kwargs)
    
    #%%
    #sorted_triples_part = deepcopy(sorted_triples_full)
    tdoa_sources = assemble_tdoa_graphs(sorted_triples_full, **kwargs)
    #trippool, pot_tdoas = build_full_tdoas(sorted_triples_part)
    from pydatemm.localisation import spiesberger_wahlberg_solution
    all_sources = []
    all_ncap = []
    for i,each in enumerate(tdoa_sources):
        d_0 = each.graph[1:,0]*340
        try:
            sources = spiesberger_wahlberg_solution(kwargs['array_geom'],  d_0)
            all_sources.append(sources)
            all_ncap.append(ncap(each, sources, kwargs['array_geom']))
        except:
            all_sources.append(np.nan)
            all_ncap.append(np.nan)    
    #%%
    complete = [i.is_complete_graph() for i in tdoa_sources]
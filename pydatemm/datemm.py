# -*- coding: utf-8 -*-
"""
DATEMM
======
Module that implements the final workflow including all sub-steps.

"""
import numpy as np 
import graph_building as grapher
import localise
import timediffestim as tde
import raster_matching as rasmat
import triple_generation as tripgen

def datemm_track(audio,**kwargs):
    '''
    '''
    num_chunks = int(audio.shape[0]/kwargs.get('block_size', 1024))
    # break the audio into chunks
    # TODO: break the audio into overlapping chunks, with a user-specified
    # overlap parameter
    chunks = np.array_split(audio, num_chunks)
    # for each 
    all_localisations = []
    all_suppdata = []
    for chunk in chunks:
        sources, supp_data = datemm_per_chunk(chunk, **kwargs)
        all_localisations.append(sources)
        all_suppdata.append(supp_data)
        
    return all_localisations, all_suppdata

def datemm_per_chunk(audio, **kwargs):
    '''
    Performs DATEMM tracking on all audio blocks
    
    Parameters
    ----------
    array_geom
    v_sound
    '''
    # pair-wise cross-corelations (cc) and TDOA detection
    multich_cc = tde.generate_multich_crosscorr(audio, **kwargs)
    multich_tdoas = tde.get_multich_tdoas(multich_cc, **kwargs)
    # remove impossible pairwise TDOAs based on array geometry
    valid_tdoas = tde.geometrically_valid(multich_tdoas, **kwargs)
    # channel wise auto-correlations (aa) and peak detection
    multich_aa = tde.generate_multich_autocorr(audio)
    # CONTINUE FROM HERE!
    aa_tdes = tde.get_multich_aa_tdes(multich_aa)
    # Raster matching - which of the pair-wise TDEs come from indirect paths?
    tdoas_rastermatched = rasmat.raster_matcher(valid_tdoas, aa_tdes, **kwargs)
    # Generate consistent triples from all the pairwise TDOAs
    triple_candidates = tripgen.triple_generate(tdoas_rastermatched, **kwargs)  
    # Expand triples and attemp building larger graphs
    graph_candidates, graph_quality = grapher.triple_to_graphs(triple_candidates, **kwargs)
    # Localise from all generated graphs
    all_sources = []
    for graph, quality in zip(graph_candidates, graph_quality): 
        sources = localise.localise(graph, **kwargs)
        all_sources.append(sources)
    graph_quality_score = []
    sources = []
    return all_sources, graph_quality_score
    
if __name__ == '__main__':
    audio = np.random.normal(0,1,9000*4).reshape(-1,4)
    a, b = datemm_track(audio)
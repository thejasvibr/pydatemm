#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source generation
=================
Calculates all possible sources that generate consistent graphs.
"""
import os 
import pydatemm.localiser as lo
import pydatemm.timediffestim as timediff
import  pydatemm.graph_manip as gramanip
try:
    import cppyy as cpy
except:
    pass

def cpp_make_array_geom(**kwargs):
    ''' Takes the np.array array_geom & converts it 
    into a Eigen::MatrixXd 
    
    Keyword Arguments
    -----------------
    array_geom : (Nmics, 3) np.array
    
    Returns
    -------
    cpp_array_geom : (Nmics, 3) Eigen::MatrixXd
    '''
    nmics, ncols = kwargs['array_geom'].shape
    cpp_array_geom = cpy.gbl.Eigen.MatrixXd(nmics, ncols)
    for i in range(nmics):
        for j in range(ncols):
            cpp_array_geom[i,j] = kwargs['array_geom'][i,j]
    return cpp_array_geom

def generate_candidate_sources(sim_audio, **kwargs):
    '''
    Parameters
    ----------
    sim_audio : (Msamples, Nchannels) np.array
    
    Keyword Arguments
    -----------------
    num_cores : int, optional
        Number of CPU cores to use. Defaults to all. 
    K : int, optional 
        Number of peaks per channel-pair cross-correlation to use. 
        Defaults to 5. 
    vsound : float, optional 
        Speed of sound in m/s. Defaults to 343 m/s. 
    max_loop_residual : float, optional 
        Maximum loop residual in seconds. An ideal consistent loop should have
        0 residual. Defaults to 1e-6 s.
    fs : int >0
        Frequency of sampling in Hz.
    array_geom : (Nchannels,3) np.array
        XYZ of microphone positions
    min_peak_diff : float
        Minimum peak difference in detection time. The minimum 'distance' detected
        peaks need to be at. The shorter this time is, the more peaks are likely
        to be detected. 
    
    Returns 
    -------
    sources_and_data : C++ struct
        A C++ struct object with the attributes 'sources', 'cfl_ids', 'tde_in'.
        'sources' has all the candidate sources as a vector<vector<double>>.
        'cfl_ids' has all of the indices of the consistent Fundamental Loops output
        by `make_consistent_fls_cpp`. 
        'tde_in' is the 1D data fed into the localisation routines to generate
        candidate sources. 
    
    Example
    -------
    >>> from pydatemm.source_generation import generate_candidate_sources
    Load the audio here, and assign it to ```sim_audio```
    Also load the microphone positions into ```mic_xyz```
    >>> parameters = {'fs' : 192000, 
                      'max_loop_residual' : 1e-5,
                      'nchannels' : 8,
                      'K' : 6,
                      'array_geom': mic_xyz,
                      'min_peak_diff' : 0.35e-4,
                      }
    >>> candidate_sources = generate_candidate_sources(sim_audio, **parameters)
    '''
    num_cores = kwargs.get('num_cores', os.cpu_count())
    multich_cc = timediff.generate_multich_crosscorr(sim_audio, **kwargs )
    cc_peaks = timediff.get_multich_tdoas(multich_cc, **kwargs)
    K = kwargs.get('K',5) # number of peaks per channel CC to consider
    top_K_tdes = {}
    for ch_pair, tdes in cc_peaks.items():
        descending_quality = sorted(tdes, key=lambda X: X[-1], reverse=True)
        top_K_tdes[ch_pair] = []
        for i in range(K):
            try:
                top_K_tdes[ch_pair].append(descending_quality[i])
            except:
                pass
    cfls_from_tdes = gramanip.make_consistent_fls_cpp(top_K_tdes, **kwargs)
    ccg_matrix = cpy.gbl.make_ccg_matrix(cfls_from_tdes)
    solns_cpp = lo.CCG_solutions_cpp(ccg_matrix)
    ag = cpp_make_array_geom(**kwargs)
    sources_and_data = cpy.gbl.localise_sounds_v3(num_cores, ag, solns_cpp, cfls_from_tdes)
    return sources_and_data





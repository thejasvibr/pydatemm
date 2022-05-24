# -*- coding: utf-8 -*-
"""
TDOA quality metrics
--------------------
Collection of functions to calculate the following metrics:
    * connectivity
    * residual position error
    * residual TDOA error

Created on Wed May 11 07:51:27 2022
@author: Thejasvi Beleyur
"""
from itertools import combinations
import numpy as np
from scipy.spatial import distance
euclidean = distance.euclidean

def nan_euclidean(u,v):
    nan_indices = [int(np.argwhere(np.isnan(each))) for each in [u,v]]
    print(nan_indices)
    if not len(set(nan_indices))==1:
        raise ValueError('Cannot compare. The Nans are at diff positions')
    else:
        u_nonan = u[~np.isnan(u)]
        v_nonan = v[~np.isnan(v)]
    dist = euclidean(u_nonan, v_nonan)
    return dist

def triplet_quality(triplet, **kwargs):
    '''
    Calculates triplet quality score- which is the product of the 
    TFTM output and the sum of individual TDOA qualities.
    This metric is defined in eqn. 23

    Parameters
    ----------
    triplet: triple graph
        A nx.DiGraph with 3 nodes. Each edge is expected to have the 
        'tde' and 'peak_score' weight attributes.
    twtm : float
        Tolerance width of triple match in seconds. 

    Returns
    -------
    quality : float
        The triple quality based on how close it is to zero and the 
        windowing function of :code:`gamma_tftm`

    See Also
    --------
    pydatemm.tdoa_quality.gamma_tftm
    '''
    a, b, c = tuple(triplet.nodes)
    t12, t23, t31 = [triplet.edges[edge]['tde'] for edge in [(a,b), (b,c), (c,a)]]
    peak_q12, peak_q23, peak_q31 = [triplet.edges[edge]['peak_score'] for edge in [(a,b), (b,c), (c,a)]]
    tdoa_quality_sum = peak_q12+peak_q23+peak_q31
    tdoa_tftm_score = gamma_tftm(t12, t23, t31, **kwargs)
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

def graph_connectivity_w(tdoa_object, **kwargs):
    '''
    Calculates connectivity 'w' of a synthesised graph. 
    Defined in eqn. 24 of Scheuing & Yang 2018. 
    
    '''
    if not tdoa_object.is_complete_graph():
        raise ValueError('Input TDOA object does not have a complete graph.\
                         Cannot proceed')
    # get all component triples
    object_graph = tdoa_object.graph
    num_nodes = object_graph.shape[0]
    # calculate gamma_TFTM for each of the triples
    triple_combis = list(combinations(range(num_nodes), 3))
    # sum it up.
    gamma_tftm_scores = []
    for each in triple_combis:
        a,b,c = each
        tab, tbc, tca =  object_graph[a,b], object_graph[b,c], object_graph[c,a]
        tftm_score = gamma_tftm(tab, tbc, tca, **kwargs)
        gamma_tftm_scores.append(tftm_score)
    w = np.sum(gamma_tftm_scores)
    return w

def residual_position_error(true_pos, obtained_pos):
    '''
    Eqn. 31
    '''
    return euclidean(true_pos, obtained_pos)

def residual_tdoa_error(tdoa_array, source, array_geom, **kwargs):
    '''
    Implements Eqn. 32 in Scheuing & Yang 2008. The residual 
    tdoa error (n cap)  'compares the implicit microphone positions
    in the TDOA graph with the true ones'. 

    Parameters
    ----------
    tdoa_array : np.array
        TDOA array
    source : (3)/(3,1) np.array
    array_geom : (Nmics,3) np.array


    References
    ----------
    * Scheuing & Yang 2008, ICASSP
    
    See Also
    --------
    pydatemm.tdoa_objects
    '''
    ref_channel = kwargs.get('ref_channel', 0)
    n_channels = kwargs['nchannels']
    distmat = np.apply_along_axis(euclidean, 1, array_geom, source)
    # the TDOA vector measured from data
    measured_n_cap = tdoa_array[:,ref_channel]
    obtained_n_tilde = np.zeros(n_channels)
    for i in range(n_channels):
        diff = distmat[i] - distmat[ref_channel]
        if diff == 0:
            obtained_n_tilde[i] = np.nan
        else:
            obtained_n_tilde[i] = diff
    obtained_n_tilde /= kwargs.get('c', 340)
    # tdoa residual 
    print(measured_n_cap, obtained_n_tilde)
    tdoa_resid = nan_euclidean(measured_n_cap, obtained_n_tilde)
    tdoa_resid /= np.sqrt(n_channels)
    return tdoa_resid
    
    
    
    
    
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

def graph_connectivity_w(tdoa_object, **kwargs):
    '''
    Calculates connectivity 'w' of a synthesised graph. 
    Defined in eqn. 24 of Scheuing & Yang 2018. 
    
    '''
    if not tdoa_object.is_complete_graph():
        raise ValueError(f'Input TDOA object does not have a complete graph.\
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

def residual_tdoa_error(tdoa_object, source, array_geom, **kwargs):
    '''
    Eqn. 32
    '''
    ref_channel = kwargs.get('ref_channel', 0)
    n_channels = tdoa_object.graph.shape[0]
    distmat = np.apply_along_axis(euclidean, 1, array_geom, source)
    # the TDOA vector measured from data
    measured_n_cap = tdoa_object.graph[:,ref_channel]
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
    
    
    
    
    
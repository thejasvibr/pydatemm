#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementing a vector based implementation of SW2002
====================================================
Created on Mon Sep 26 13:40:03 2022

@author: thejasvi
"""

import numpy as np
import scipy.spatial as spl
euclid = spl.distance.euclidean
#np.random.seed(82319) 
matmul = np.matmul

def get_nmics(tde_data):
    
    if (tde_data.size+1) % 4 == 0:
        return int((tde_data.size+1)/4)
    else:
        raise ValueError(f'{tde_data.size} columns input - unable to calculate Nmics')

def sw_matrix_optim(mic_ntde_orig, c=343.0):
    '''
    mic_ntde : 3*nmics + nmics-1
    The last nmics-1 entries are range differences in meters!!
    '''
    nmics = get_nmics(mic_ntde_orig)
    mic_ntde = mic_ntde_orig.copy()
    #print(mic_ntde.shape)
    position_inds = nmics*3
    mic0 = mic_ntde[:3]
    starts, stops = np.arange(3,position_inds,3), np.arange(6,position_inds+3,3)
    for start, stop in zip(starts, stops):
        mic_ntde[start:stop] -= mic0
    
    tau = mic_ntde[-(nmics-1):]/c
    R = mic_ntde[3:position_inds].reshape(-1,3)
    
    # R_inv = np.linalg.pinv(R)
    R_inv,*_  = np.linalg.lstsq(R, np.eye(R.shape[0]), rcond=None)
    #print(R_inv)
    Nrec_minus1 = R.shape[0]
    b = np.zeros(Nrec_minus1)
    f = np.zeros(Nrec_minus1)
    g = np.zeros(Nrec_minus1)
    #print(R, tau)
    for i in range(Nrec_minus1):
        b[i] = np.linalg.norm(R[i,:])**2 - (c*tau[i])**2
        f[i] = (c**2)*tau[i]
        g[i] = 0.5*(c**2-c**2)

    a1 = matmul(matmul(R_inv, b).T, matmul(R_inv,b))
    a2 = matmul(matmul(R_inv, b).T, matmul(R_inv,f))
    #print(f'Rinv-f {matmul(R_inv,f)}')
    a3 = matmul(matmul(R_inv, f).T, matmul(R_inv,f))
    
    
    a_quad = a3 - c**2
    b_quad = -a2
    c_quad = a1/4.0
    #print(f"a_quad {a_quad}, b_quad {b_quad}, c_quad {c_quad}")
    #print(f"a1,2,3: {a1,a2,a3}")
    #print(f"yy_pt1: {b_quad**2 } yy_pt2 { 4*a_quad*c_quad }")
    #print(f'Potential: {(b_quad**2) - 4*a_quad*c_quad}, {np.sqrt((b_quad**2) - 4*a_quad*c_quad)}')
    t_soln1 = (-b_quad + np.sqrt(b_quad**2 - 4*a_quad*c_quad))/(2*a_quad)
    t_soln2 = (-b_quad - np.sqrt(b_quad**2 - 4*a_quad*c_quad))/(2*a_quad)
    
    #print(a_quad, b_quad, c_quad, t_soln1, t_soln2, )
    s12 = np.zeros(6)
    s12[:3] = matmul(R_inv,b*0.5) - matmul(R_inv,f)*t_soln1
    s12[3:] = matmul(R_inv,b*0.5) - matmul(R_inv,f)*t_soln2

    s12[:3] += mic0
    s12[3:] += mic0
    final_solution = choose_correct_solution(s12, mic_ntde_orig[:nmics*3].reshape(-1,3),
                                             tau*c)
    return final_solution


def choose_correct_solution(all_sources, array_geom, rangediffs, **kwargs):
    '''
    The Spiesberger-Wahlberg 2002 method always provides 2 potential solutions.
    The authors themselves suggest comparing the observed channel 5 and 1
    time difference ():math:`\tau_{51}` ) and the predicted :math:`\tau_{51}`
    from each source to see which one is a better fit. 

    Parameters
    ----------
    sources : list
        List with 2 sources. Each source is a (3,)/(3,1) np.array
    array_geom : (Nmics, M) np.array
    rangediffs : (N-1,) np.array
        Range differences to reference microphone. 

    Returns
    -------
    valid_solution : (3)/(3,1) np.array
        The correct solution of the two potential solutions.
    '''
    sources = [all_sources[:3], all_sources[3:]]
    tau_ch1_sources = [rangediff_pair(each, 4, array_geom) for each in sources]
    residuals = [rangediffs[3]-tauch1 for tauch1 in tau_ch1_sources]

    # choose the source with lower rangediff residuals
    lower_error_source = np.argmin(np.abs(residuals))
    valid_solution = sources[lower_error_source]
    return valid_solution

def rangediff_pair(source, chX, array_geom):
    ch0_dist = np.linalg.norm((source-array_geom[0,:]))
    chX_dist = np.linalg.norm(source- array_geom[chX,:])
    return chX_dist - ch0_dist



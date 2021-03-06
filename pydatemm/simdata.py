# -*- coding: utf-8 -*-
"""
Simulated data 
==============
Generates simulated data for use during development. 

Created on Fri Apr 22 11:46:37 2022

@author: thejasvi
"""
import numpy as np 
import scipy.spatial as spatial 
import scipy.signal as signal 



def simulate_1source_and_1reflector(**kwargs):
    '''
    Parameters
    ----------
    sound_pos : (1,3) np.array
    refelction_source : (1,3) np.array
    
    Returns
    -------
    audio : (Nsamples, 4) np.array
    dist_mat : (Nsamples,Nsamples) np.array
    tristar : (4, 3) np.array
    (sound_pos, reflection_source) : (np.array, np.array)
    '''
    # microphone array geometry
    tristar = make_tristar(**kwargs)
    
    sound_pos = kwargs.get('sound_pos',np.array([3,2,1]))
    reflection_source = kwargs.get('reflection_source',np.array([1,4,1]))
    direct_indirect_sources = np.row_stack((sound_pos, reflection_source))
    # direct path propagation:
    
    dist_mat = spatial.distance_matrix(direct_indirect_sources, tristar)
    #add the distance of propagation from source to reflection point
    source_to_reflectionpoint = spatial.distance.euclidean(sound_pos, reflection_source)
    dist_mat[1,:] += source_to_reflectionpoint
    
    # make the direct
    
    chirp_durn = 0.003
    fs = 192000
    t = np.linspace(0,chirp_durn,int(fs*chirp_durn))
    chirp = signal.chirp(t,80000,t[-1],25000)
    chirp *= signal.hann(chirp.size)*0.5
    
    
    vsound = 340.0
    audio = np.zeros((int(fs*0.03),4))
    toa_sounds = dist_mat/vsound
    toa_samples = np.int64(toa_sounds*fs)
    for channel in range(4):
        random_atten = np.random.choice(np.linspace(0.2,0.9,20),2)
        start_direct, start_indirect = toa_samples[0,channel], toa_samples[1,channel]
        audio[start_direct:start_direct+chirp.size,channel] += chirp*random_atten[0]
        audio[start_indirect:start_indirect+chirp.size,channel] += chirp*random_atten[1]
    audio += np.random.normal(0,1e-5,audio.size).reshape(audio.shape)
    return audio , dist_mat, tristar, (sound_pos, reflection_source)

def make_tristar(**kwargs):
    '''
    Defaults to making a 'giant' 1.2 m tristar array
    Parameters
    ----------
    R : radius of tristar in m
    Returns 
    -------
    tristar: (4,3) np..array
    '''
    R = kwargs.get('R', 1.2)
    theta = np.pi/3
    tristar = np.row_stack(([0,0,0],
                            [-R*np.sin(theta), 0, -R*np.cos(theta)],
                            [R*np.sin(theta), 0, -R*np.cos(theta)],
                            [0,0, R]))
    tristar[:,1] += [1e-3, 0.5e-3, 0.25e-3, 0.15e-3] # many algorithms don't
    # like it if your position is 0,0,0
    return tristar
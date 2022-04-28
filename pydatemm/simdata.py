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

def  simulate_1source_and1reflector_general(**kwargs):
    '''
    if nmics>4 , then nmics with an x-y-z 
    distribution of N(0,2) are created.

    Parameters
    ----------
    sound_pos: (1,3) np.array, optional
    reflection_source: (1,3) np.array, optional
    nmics : int, optional

    Returns
    -------
    None.

    '''
    nmics = kwargs.get('nmics', 4)
    if nmics == 4:
        array_geom = make_3dtristar(**kwargs)
    else:
        array_geom = np.random.normal(0,2,nmics*3).reshape(-1,3)
    
    sound_pos = kwargs.get('sound_pos',np.array([3,2,1]))
    reflection_source = kwargs.get('reflection_source',np.array([1,4,1]))
    direct_indirect_sources = np.row_stack((sound_pos, reflection_source))
    # direct path propagation:
    
    dist_mat = spatial.distance_matrix(direct_indirect_sources, array_geom)
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
    toa_sounds = dist_mat/vsound
    audio = np.zeros((int(fs*np.max(toa_sounds+0.01)),nmics))
    toa_samples = np.int64(toa_sounds*fs)
    for channel in range(nmics):
        random_atten = np.random.choice(np.linspace(0.2,0.9,20),2)
        start_direct, start_indirect = toa_samples[0,channel], toa_samples[1,channel]
        audio[start_direct:start_direct+chirp.size,channel] += chirp*random_atten[0]
        audio[start_indirect:start_indirect+chirp.size,channel] += chirp*random_atten[1]
    audio += np.random.normal(0,1e-5,audio.size).reshape(audio.shape)
    return audio , dist_mat, array_geom, (sound_pos, reflection_source)


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
    toa_sounds = dist_mat/vsound
    audio = np.zeros((int(fs*np.max(toa_sounds+0.01)),4))
    toa_samples = np.int64(toa_sounds*fs)
    for channel in range(4):
        random_atten = np.random.choice(np.linspace(0.2,0.9,20),2)
        start_direct, start_indirect = toa_samples[0,channel], toa_samples[1,channel]
        audio[start_direct:start_direct+chirp.size,channel] += chirp*random_atten[0]
        audio[start_indirect:start_indirect+chirp.size,channel] += chirp*random_atten[1]
    audio += np.random.normal(0,1e-5,audio.size).reshape(audio.shape)
    return audio , dist_mat, tristar, (sound_pos, reflection_source)

def simulate_1source_and_1reflector_3dtristar(**kwargs):
    '''
    1 source and reflector with 3d tristar

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
    tristar = make_3dtristar(**kwargs)
    
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
    toa_sounds = dist_mat/vsound
    audio = np.zeros((int(fs*np.max(toa_sounds+0.01)),4))
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

def make_3dtristar(**kwargs):
    array_geom = make_tristar(**kwargs)
    array_geom += np.random.normal(0,0.1,array_geom.size).reshape(array_geom.shape)
    return array_geom

def simulate_1source_and_2reflector(**kwargs):
    '''
    Runs simulate_1source_and_1reflector twice.
    Second time round runs with a diff source and reflector
    Only first source and first reflector can be specified. 
    Second source and reflector are hard coded.

    Parameters
    ----------
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    audio , dist_mat, tristar, (sound_pos, reflection_source) = simulate_1source_and_1reflector(**kwargs)
    audio2, _, _, (sound_pos, reflection_source2) = simulate_1source_and_1reflector(**{'sound_pos':sound_pos,
                                                     'reflection_source':np.array([4,4,-2])})
    both_audio = [audio, audio2]
    samples_order = np.argsort([audio.shape[0] for each in both_audio])
    ordered_audio = [both_audio[each] for each in samples_order] 
    
    final_audio = np.zeros(ordered_audio[1].shape)
    final_audio[:ordered_audio[0].shape[0],:] += ordered_audio[0]
    final_audio += ordered_audio[1]
    
    final_audio /= np.max(np.abs(final_audio))
    final_audio *= 0.8
    return final_audio, dist_mat, tristar, (sound_pos, [reflection_source, reflection_source2])


def simulate_1source_and_3reflector(**kwargs):
    '''
    Runs simulate_1source_and_1reflector twice.
    Second time round runs with a diff source and reflector
    Only first source and first reflector can be specified. 
    Second source and reflector are hard coded.

    Parameters
    ----------
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    audio , dist_mat, tristar, (sound_pos, reflection_source) = simulate_1source_and_2reflector(**kwargs)
    audio2, _, _, (sound_pos, reflection_source2) = simulate_1source_and_1reflector(**{'sound_pos':sound_pos,
                                                     'reflection_source':np.array([4,5,-2])})
    both_audio = [audio, audio2]
    samples_order = np.argsort([audio.shape[0] for each in both_audio])
    ordered_audio = [both_audio[each] for each in samples_order] 
    
    final_audio = np.zeros(ordered_audio[1].shape)
    final_audio[:ordered_audio[0].shape[0],:] += ordered_audio[0]
    final_audio += ordered_audio[1]
    
    final_audio /= np.max(np.abs(final_audio))
    final_audio *= 0.8
    return final_audio, dist_mat, tristar, (sound_pos, [reflection_source, reflection_source2])

if __name__=='__main__':
    a,_,_,_ = simulate_1source_and_3reflector()


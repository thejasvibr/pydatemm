# -*- coding: utf-8 -*-
"""
GCC map creation and manipulation
=================================

Created on Sat May 11 07:58:31 2024

@author: theja
"""
import numpy as np 
from gccestimating import GCC, corrlags

def range_difference(micpair, micxyz, traj):
    '''
    Parameters
    ----------
    micpair : (ch1, ch2) tuple
    micxyz : (Mmics,3) np.array
    traj : (Npositions,3) np.array

    Returns
    -------
    rangediff : (Npositions,) np.array
        Range difference between channel 1 and channel 2, with channel 2 as reference
    '''
    ch1, ch2 = micpair
    rangediff = np.linalg.norm(traj - micxyz[ch1,:], axis=1) - np.linalg.norm(traj - micxyz[ch2,:], axis=1)
    return rangediff


def gcc_phat_siggigue(s1, s2):
    '''
    Thin wrapper around SiggiGue's original implementation 
    
    TODO
    ----
    * The peaks may be +/- 1 sample off when s1,s2 are of particular lengths...
    '''
    gcce_phat = GCC(s1, s2).phat().sig
    # format size so the GCC output signal is the same size as s1/s2
    midpoint = int(gcce_phat.size*0.5)
    audio_samples = s1.size
    leftflank = int(audio_samples*0.5) 
    rightflank = s1.size - (leftflank - midpoint)
    return gcce_phat[leftflank:rightflank]

def compute_gcc_scores_overlapping(sounds, sample_rate, chunk_length, overlap,
                                   speed_of_sound = 343):
    """ Computes the GCC-PHAT correlation maps with overlapping chunks
    
    This function is a modified version of Viktor Larsson's compute_gcc_scores
    from the 'audiometric' repository
    
    Parameters
    ----------
    sounds : (Msamples, Nchannels) np.array
    sample_rate : int
    chunk_length : int
        How long each individual audio chunk should be
    overlap : 0<float<1
        The extent of overlap between chunks. Rounded down to the nearest
        integer value of the chunk_length
    speed_of_sound : float
        Defaults to 343 m/s
    
    Returns
    -------
    gcc_scores : (Mchannels, Mchannels, Chunklength, Chunktimes) np.array
        A 4D GCC-PHAT map.
    chunk_times : np.array
        The starting time for each chunk. 
    """

    #constants
    onechunk_every = chunk_length - int(np.floor(chunk_length*overlap))
    maxn_chunks, remaining_samples = np.divmod(sounds.shape[0], onechunk_every)
    n_mics = sounds.shape[1]

    # divide sound into chunks - discard the last remainder chunk if it's not full-length
    chunks = []
    for i in range(maxn_chunks):
        start_ind, stop_ind = i*onechunk_every, i*onechunk_every+chunk_length
        if stop_ind <= sounds.shape[0]:
            chunks.append(sounds[start_ind:stop_ind,:])

    if chunks[-1].shape[0] != chunk_length:
        chunks.pop(-1)
    n_chunks = len(chunks)
    
    chunk_times = np.arange(0, sounds.shape[0], onechunk_every)[:len(chunks)]/sample_rate

    # compute gcc-phat
    gcc_scores = np.zeros((n_mics,n_mics,chunk_length,n_chunks))    

    for chunk_i, chunk in enumerate(chunks):
        for mic_1 in range(n_mics):
            for mic_2 in range(mic_1+1, n_mics):
                gcc_scores[mic_1,mic_2,:,chunk_i] = gcc_phat_siggigue(chunk[:,mic_1],chunk[:,mic_2])
                gcc_scores[mic_2,mic_1,:,chunk_i] = np.flip(gcc_scores[mic_1,mic_2,:,chunk_i])

    return gcc_scores, chunk_times

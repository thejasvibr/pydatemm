# -*- coding: utf-8 -*-
"""
Investigating robust TDE estimation in multi-source reverberant audio
=====================================================================
Been having a problem with 





Created on Fri Aug 19 15:27:28 2022

@author: theja
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import pyroomacoustics as pra
import scipy.signal as signal
import scipy.spatial as spl
from scipy.spatial import distance 
import soundfile as sf
from pydatemm.timediffestim import estimate_gcc
from gccestimating import GCC, corrlags

def expected_tdes(sources, micB, micA, vsound=343):
    tdes = []
    for s in sources:
        # [RmicBsource, RmicAsource]
        micBA_dist = [distance.euclidean(s, mic) for mic in [micB, micA]]
        # RmicBsource - RmicAsource]
        tde = micBA_dist[0] - micBA_dist[1]
        tde /= vsound
        tdes.append(tde)
    return tdes

def max_interch_delay(array_geom, chBA, vsound=343):
    chB, chA = chBA
    interch_dist = distance.euclidean(array_geom[chB,:], array_geom[chA,:])/vsound
    return interch_dist
    
    
def crosscor(B,A):
    return signal.correlate(B,A)

# from the pra docs
def audiosim_room(ref_order, array_geom, **kwargs):
    """
    ref_order : int
        Order or reflections to implement. Zero means only direct paths are
        simulated. 
    array_geom : (Nchannels, 3)
        xyz of microphones.
    """
    room_dim = [9, 7.5, 3.5]  # meters
    fs = kwargs.get('fs', 192000)
    call_durn = kwargs.get('call_durn', 5e-3)
    ray_trace = kwargs.get('ray_trace', True)
    rt60_tgt = 0.25  # seconds
    e_absorption, _ = pra.inverse_sabine(rt60_tgt, room_dim)
    
    reflection_max_order = ref_order
    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), 
            max_order=reflection_max_order, air_absorption=True, 
            ray_tracing=ray_trace
    )

    t_call = np.linspace(0,call_durn, int(fs*call_durn))
    batcall = signal.chirp(t_call, 9000, t_call[-1], 85000,'linear')
    batcall *= signal.hanning(batcall.size)
    batcall *= 0.6
    default_sources = [[2.5, 5, 3],
               [8, 3, 2],
               [1, 4, 0.5],
               [2,4,2]]
    sources = kwargs.get('sources', default_sources)

    delay = np.linspace(0,0.010,len(sources))
    for each, emission_delay in zip(sources, delay):
        room.add_source(position=each, signal=batcall, delay=emission_delay)
    
    room.add_microphone_array(array_geom.T)
    room.compute_rir()
    room.simulate()
    sim_audio = room.mic_array.signals.T
    filename = f'simaudio_maxorder-{reflection_max_order}_raytracing-{str(ray_trace)}.wav'
    sf.write(filename, sim_audio, samplerate=fs)
    return sim_audio, sources 

if __name__ == "__main__":
    #%% Generate simulated audio
    array_xyz = pd.read_csv('../pydatemm/tests/scheuing-yang-2008_micpositions.csv').to_numpy()
    fs = 192000
    kwargs = {'fs':fs, 'ray_trace':False}   
    simaudio_direct, sources = audiosim_room(0, array_xyz, **kwargs)
    kwargs['ray_trace'] = True
    simaudio_1ref, sources = audiosim_room(1, array_xyz, **kwargs)
    #%% Compare the crosscor between channels in direct audio vs with reflections
    chnoB, chnoA = 6,2
    chB, chB1 = simaudio_direct[:,chnoB], simaudio_1ref[:,chnoB]
    chA, chA1 = simaudio_direct[:,chnoA], simaudio_1ref[:,chnoA]
    
    max_delay = max_interch_delay(array_xyz, (chnoB, chnoA))
    max_delay_int = int(fs*max_delay)
    exp_tdes = expected_tdes(sources, array_xyz[chnoB,:], array_xyz[chnoA,:])
    exp_tdes_int = np.int64(np.array(exp_tdes)*fs)
    
    def make_t_cc(cc,fs):
        t = np.linspace(-cc.size/2, cc.size/2, cc.size)/fs
        plt.plot(t,cc)
    
    cc_direct = estimate_gcc(chB, chA)
    cc_1ref = estimate_gcc(chB1, chA1)
    
    #%% Using gccestimating
    gcc = GCC(simaudio_direct[:,chnoB], simaudio_direct[:,chnoA])
    gcc1 = GCC(simaudio_1ref[:,chnoB], simaudio_1ref[:,chnoA])
    
    
    # Hanan Thomson - Maximum Likelihood estimator seems to work well.
    gcc_dir = gcc.scot()
    exp_tdes_samples = gcc_dir.sig.size*0.5
    gcc_wref = gcc1.scot()
    
    # plt.figure()
    # a0 = plt.subplot(211)
    # plt.plot(gcc_dir)
    # for tde in (simaudio_direct.shape[0]+exp_tdes_int):
    #     plt.vlines(tde, np.max(gcc_dir)*0.6, np.max(gcc_dir), 'r')
    # #plt.xlim(np.min(exp_tdes_int)-1000, np.max(exp_tdes_int)+1000)
    
    # plt.subplot(212)
    # plt.plot(gcc_wref); plt.title('w reflections')
    # for tde in (simaudio_1ref.shape[0]+exp_tdes_int):
    #     plt.vlines(tde, np.max(gcc_wref)*0.4, np.max(gcc_wref), 'r')
    #plt.xlim(np.min(exp_tdes_int)-1000, np.max(exp_tdes_int)+1000)
    
    
    hilbert_gcc = [np.abs(signal.hilbert(each)) for each in [gcc_dir, gcc_wref]]
    hilbert_norm = [each/np.min(each) for each in hilbert_gcc]
    
    #%% Peak finding.
    thresholds = [99, 99.75]
    peaks_hilb = []
    for (hilb, th) in zip(hilbert_norm, thresholds):
        threshold = np.percentile(hilb, th)
        pk_output = signal.find_peaks(hilb,
                                      height=threshold,distance=int(fs*0.5e-4))
        peaks_hilb.append(pk_output[0])
    
    #%%
    plt.figure()
    a1 = plt.subplot(211)
    plt.plot(hilbert_norm[0])
    for tde in (simaudio_direct.shape[0]+exp_tdes_int):
        plt.plot(tde, hilbert_norm[0][tde]*1.2, 'k^')
    plt.xlim(simaudio_direct.shape[0]-max_delay_int, simaudio_direct.shape[0]+max_delay_int)
    for each in peaks_hilb[0]:
        plt.plot(each, hilbert_norm[0][each], '*')
    
    plt.subplot(212, )
    plt.plot(hilbert_norm[1])
    for tde in (simaudio_1ref.shape[0]+exp_tdes_int):
        plt.plot(tde, hilbert_norm[1][tde]*1.2, 'k^')
    
    for each in peaks_hilb[1]:
        plt.plot(each, hilbert_norm[1][each], '*')
    plt.xlim(simaudio_1ref.shape[0]-max_delay_int, simaudio_1ref.shape[0]+max_delay_int)
    

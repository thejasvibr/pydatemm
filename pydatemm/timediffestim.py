"""
Time difference estimation
==========================
Performs time-difference-of-arrival (tdoa) + autocorrelation calculations +
peak detection. 


TODO
----
Make unique keyword arguments to identify peak detection thresholds separately
for cross-correlation and auto-correlation signals.
"""


import matplotlib.pyplot as plt
import numpy as np 
import scipy.signal as signal 
import scipy.spatial as spatial
from itertools import combinations

def estimate_gcc(signal1, ref_ch):
    '''
    Parameters
    ----------
    signal1, ref_ch: np.array
        Both arrays of same size
    Returns
    -------
    ifft_gcc : np.array
        Generalised cross-correlation of the two signals
    '''
    
    fft1 = np.fft.rfft(signal1)
    fft2 = np.fft.rfft(ref_ch)
    cross_spectrum = fft1*np.conjugate(fft2)
    gcc = cross_spectrum/(np.abs(cross_spectrum))
    
    ifft_gcc = np.fft.irfft(gcc)
    ifft_gcc = np.roll(ifft_gcc, int(ifft_gcc.size*0.5))
    return ifft_gcc

def generate_multich_crosscorr(input_audio, **kwargs):
    '''
    Generates all unique pair cross-correlations: (NxN-1)/2 pairs. Each pair is
    designated by a tuple where the second number is the reference channel, eg. (1,0)
    where channel 1 is cross-correlated with reference to channel 0. 
    Parameters
    ----------
    input_audio: np.array
        M samples x N channels
    gcc : boolean
        Whether to use a gcc instead of the standard cross-correlation.    
        Defaults to False.
        
    Returns
    -------
    multichannel_cc : dictionary 
        Keys indicate the channel pair, and entries are the cross-correlation. 
        Each cross-correlation has M samples (same size as one audio channel).
    '''
    use_gcc = kwargs.get('gcc',False)
    num_channels = input_audio.shape[1]
    unique_pairs = list(combinations(range(num_channels), 2))
    multichannel_cc = {}
    for cha, chb in unique_pairs:
        # make sure the lower channel number is the reference signal 
        signal_ch, ref_signal = sorted([cha, chb], reverse=True)
        if use_gcc:
            multichannel_cc[(signal_ch, ref_signal)] = estimate_gcc(input_audio[:,signal_ch],
                                                                     input_audio[:,ref_signal])
        else:
            multichannel_cc[(signal_ch, ref_signal)] = signal.correlate(input_audio[:,signal_ch],
                                                                 input_audio[:,ref_signal],'full')
    return multichannel_cc

def generate_multich_autocorr(input_audio):
    '''
    
    Parameters
    ----------
    input_audio : np.array
        M samples x Nchannels

    Returns 
    -------
    multichannel_autocor : np.array
        M samples x Nchannels
    '''
    return np.apply_along_axis(lambda X: signal.correlate(X,X,'same'),0, input_audio)


def get_multich_aa_tdes(multich_aa, **kwargs):
    '''
    

    Parameters
    ----------
    multich_aa : (samples, Mmics) np.array
    fs : int

    Returns
    -------
    multich_aapeaks : dict
        Keys with channel number (1,1) and entries with peaks in seconds. 
    
    See Also
    --------
    timediffestim.get_peaks
    '''
    multich_aapeaks = {}
    nchannels = multich_aa.shape[1]
    for channel in range(nchannels):
        peaks = get_peaks(multich_aa[:,channel], **kwargs)
        centred_peaks = np.array(peaks-int(multich_aa[:,channel].size/2.0), dtype=np.float64)
        centred_peaks_sec = centred_peaks/kwargs['fs']
        multich_aapeaks[(channel,channel)] = centred_peaks_sec
    return multich_aapeaks

def get_multich_tdoas(multich_cc, **kwargs):
    '''
    

    Parameters
    ----------
    multich_cc : dict
        Dictionary with channel pair keys (tuples) and pairwise cross-correlation (np.array)
        entries
    fs : int
        Frequency of sampling in Hz
    
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    multich_tdoas = {}
    for ch_pair, crosscor in multich_cc.items():
        multich_tdoas[ch_pair] = np.array( np.array(get_peaks(crosscor, **kwargs)) - int(crosscor.size/2.0),
                                                              dtype=np.float64)
        multich_tdoas[ch_pair] /= np.float64(kwargs['fs'])
        multich_tdoas[ch_pair] = multich_tdoas[ch_pair].tolist()
    return multich_tdoas

def get_peaks(X,  **kwargs):
    '''
    Uses scipy.find_peaks

    Parameters
    ----------
    X : np.array
        1D signal 
    fs : int, optional
        DESCRIPTION. The default is 192000.
    
    Keyword Arguments
    -----------------
    min_peak_diff
    min_height : float>0 or list with 2 entris
        Either just the minimum or (minimum, maximum) height

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    min_peak_diff = kwargs.get('min_peak_diff', 1e-4)
    min_height = kwargs.get('min_height', 0.11)
    return signal.find_peaks(X, min_height,distance=int(kwargs['fs']*min_peak_diff))[0] 

def geometrically_valid(multich_tdoas:dict, **kwargs):
    '''
    Checks that all time differences are <= max delay defined by inter-mic
    distance.

    Parameters
    ----------
    multich_tdoas : dict
        Dictionary with channel pair keys (tuples) and pairwise TDOAs in seconds
        (np.array) entries.
    array_geom: (Mmics,3) np.array
        XYZ coordinates of M microphones
    v_sound: float, optional 
        speed of sound in m/s. Default is 340 m/s.

    Returns
    -------
    geom_valid_tdoas : dict
        Keys are channel pair number e.g. (0,2) and entries are time difference
        of arrivals in seconds as a list. 
    '''
    v_sound = kwargs.get('v_sound', 340)
    distmat = spatial.distance_matrix(kwargs['array_geom'], kwargs['array_geom'])
    delaymat = distmat/v_sound
    geom_valid_tdoas = {}
    
    for ch_pair, pair_tdoas in multich_tdoas.items():
        m1, m2 = ch_pair
        mic2mic_delay = delaymat[m1,m2]
        geom_valid_tdoas[ch_pair] = []

        if len(pair_tdoas)==0:
            continue 
        # if not empty
        for tdoa in pair_tdoas:
            if abs(tdoa)<=mic2mic_delay:
                geom_valid_tdoas[ch_pair].append(tdoa)
    return geom_valid_tdoas
                

if __name__ == '__main__':
    from simdata import simulate_1source_and_1reflector
    audio, _, _, _ = simulate_1source_and_1reflector()
    multich_cc = generate_multich_crosscorr(audio, use_gcc=True)
    multich_ac = generate_multich_autocorr(audio)

    #%%
    rnaudio = np.random.normal(0,1,1000)
    mch_audio = np.column_stack((rnaudio,
                                 np.roll(rnaudio,10),
                                 np.roll(rnaudio,-10),
                                 np.roll(rnaudio,-30)))
    
    
    mcs_cc = generate_multich_crosscorr(mch_audio)
    mch_peaks = get_multich_tdoas(mcs_cc, min_height=200, fs=192000.0)
    #%%
    mch_audio[:,0] += np.roll(mch_audio[:,0], 30)*0.2
    multiaa = get_multich_aa_tdes(mch_audio, fs=192000)
    
    
    

    
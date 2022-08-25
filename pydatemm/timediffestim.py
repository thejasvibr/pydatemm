"""
Time difference estimation
==========================
Performs time-difference-of-arrival (tdoa) + autocorrelation calculations +
peak detection. 


TODO
----
* Make unique keyword arguments to identify peak detection thresholds separately
for cross-correlation and auto-correlation signals.

References
----------
* Scheuing, J., & Yang, B. (2008). Disambiguation of TDOA estimation for multiple
sources in reverberant environments. IEEE transactions on audio, speech, and
language processing, 16(8), 1479-1489.
"""

import matplotlib.pyplot as plt
import numpy as np 
import scipy.signal as signal 
import scipy.spatial as spatial
from itertools import combinations
from gccestimating import GCC


def max_interch_delay(chBA, array_geom, vsound=343):
    chB, chA = chBA
    interch_dist = spatial.distance.euclidean(array_geom[chB,:],
                                              array_geom[chA,:])/vsound
    return interch_dist

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

def perform_gcc_variant(Y,X,variant):
    ch_gcc = GCC(Y,X)
    if variant == 'phat':
        out = ch_gcc.phat().sig
    elif variant == 'scot':
        out = ch_gcc.scot().sig
    elif variant == 'roth':
        out =  ch_gcc.roth().sig
    else:
        raise ValueError(f'Variant {variant} not found. Valid are: phat, scot or roth')
    return out

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
    gcc_variant : str
        One of ['phat', 'roth', 'scot'] - default is 'phat'. 
        

    Returns
    -------
    multichannel_cc : dictionary 
        Keys indicate the channel pair, and entries are the cross-correlation. 
        Each cross-correlation has M samples (same size as one audio channel).
    '''
    use_gcc = kwargs.get('gcc',False)
    if use_gcc:
        gcc_variant = kwargs.get('gcc_variant', 'phat')

    num_channels = input_audio.shape[1]
    unique_pairs = list(combinations(range(num_channels), 2))
    multichannel_cc = {}
    for cha, chb in unique_pairs:
        # make sure the lower channel number is the reference signal 
        signal_ch, ref_signal = sorted([cha, chb], reverse=True)
        if use_gcc:
            # multichannel_cc[(signal_ch, ref_signal)] = estimate_gcc(input_audio[:,signal_ch],
            #                                                          input_audio[:,ref_signal])
            multichannel_cc[(signal_ch, ref_signal)] = perform_gcc_variant(input_audio[:,signal_ch]
                                                                           , input_audio[:,ref_signal],
                                                                           gcc_variant)
        else:
            multichannel_cc[(signal_ch, ref_signal)] = signal.correlate(input_audio[:,signal_ch],
                                                                 input_audio[:,ref_signal],'full')
    return multichannel_cc

def generate_multich_autocorr(input_audio, **kwargs):
    '''
    
    Parameters
    ----------
    input_audio : np.array
        M samples x Nchannels
    gcc : boolean
        Whether to use a gcc instead of the standard cross-correlation.    
        Defaults to False.
    gcc_variant : str
        Required if gcc=True.
        One of ['phat', 'roth', 'scot'] - default is 'phat'. 

    Returns 
    -------
    multichannel_autocor : dict
        Keys are (channel,channel) and entry is an np.array of the 
        autocorrelation
    '''
    channels = input_audio.shape[1]
    multichannel_autocor = {}
    if kwargs.get('use_gcc', False):
        gcc_variant = kwargs.get('gcc_variant', 'phat')
        for channel in range(channels):
            multichannel_autocor[(channel, channel)] = perform_gcc_variant(input_audio[:,channel],
                                                                            input_audio[:,channel],
                                                                            gcc_variant)
    else:
        multich_aa_array = np.apply_along_axis(lambda X: signal.correlate(X,X,'same'),0, input_audio)
        for i in range(channels):
            multichannel_autocor[(i,i)] = multich_aa_array[:,i]
    return multichannel_autocor

def get_multich_aa_tdes(multich_aa, **kwargs):
    '''
    Calculates the positive autocorrelation peaks across multiple channels.

    Parameters
    ----------
    multich_aa : dict
    fs : int
    
    Keyword Arguments
    -----------------
    min_peak_diff, min_height, pctile_thresh
    
    Returns
    -------
    positive_multich_aapeaks : dict
        Keys are channel-ID (e.g. (1,1) - autocorr of channel 1).
        Entry is a list with tuples. Each tuple is a peak detection with 
        three peak properties:
        (peak position in samples, centred peak position in seconds, peak value)

    See Also
    --------
    timediffestim.get_multich_tdoas
    
    Notes
    -----
    The output of :code:`multich_aapeaks` mirrors that of :code:`get_multich_tdoas`.
    Ref the Notes there.
    '''
    multich_aapeaks = {}
    abs_thresh = kwargs.get('abs_thresh', False)

    for channel, aa in multich_aa.items():      
        peaks_raw = get_peaks(aa, **kwargs)
        peaks_sec = np.array(peaks_raw-int(aa.size/2.0), dtype=np.float64)
        peaks_sec /= np.float64(kwargs['fs'])
        peak_values = aa[peaks_raw]
        multich_aapeaks[channel] = []
       
        for peak_raw, peak_sec, peak_val in zip(peaks_raw, peaks_sec, peak_values):
            peak_tuple = (peak_raw, peak_sec, peak_val)
            multich_aapeaks[channel].append(peak_tuple)
    
    positive_multich_aapeaks = get_positive_aa_peaks(multich_aapeaks)
    return positive_multich_aapeaks

def get_multich_tdoas(multich_cc, **kwargs):
    '''

    Parameters
    ----------
    multich_cc : dict
        Dictionary with channel pair keys (tuples) and pairwise cross-correlation (np.array)
        entries
    Keyword Arguments
    -----------------
    fs : int
        Frequency of sampling in Hz
    min_height
    pctile_thresh
    min_peak_diff
    array_geom
    vsound : optional 
        Defaults to 343 m/s

    Returns
    -------
    multich_tdoas: dict
        Keys are channel pair (e.g. (1,0) ref to cross-corr bet 1 and 0 with 0 
        as the reference channel.). List with tuples. Each tuple is a peak
        detection with three peak properties:
        (peak position in samples, centred peak position in seconds, peak value)

    Notes
    -----
    The output from :code:`multich_tdoas` corresponds to :math:`P_{kl}`
    in Scheuing & Yang 2008

    For example the output of :code:`multich_tdoas` for 2 channels may look like

    .. code-block:: python
        
        {(1,0): [(203565, 5.23, 1044), (2345, -0.001, 2345)],
         (2,0): [(102,, 23e-5,  202)]} 

    Here we see the (1,0) pair has two cross-cor peaks. The sample peak 
    position is >=0. The centred peak position calculates the relative time
    difference in seconds, and the third entry provides the value
    :math:`r_{kl}(\eta_{\mu})` of the  peak :math:`\eta_{\mu}`.

    See Also
    --------
    get_peaks
    '''
    multich_tdoas = {}
    for ch_pair, crosscor in multich_cc.items():
        
        max_delay_s = max_interch_delay(ch_pair, kwargs['array_geom'],
                                        kwargs.get('vsound',343))
        max_delay_samples = int(kwargs['fs']*max_delay_s)
        midpoint = int(crosscor.size*0.5)
        start, stop = midpoint-max_delay_samples, midpoint + max_delay_samples
        relevant_cc = crosscor[start:stop]
        peaks_raw = np.array(get_peaks(relevant_cc, **kwargs))
        peaks_raw += start
        cc_delay_sec = np.array( peaks_raw - int(crosscor.size/2.0),
                                                              dtype=np.float64)
        cc_delay_sec /= np.float64(kwargs['fs']) # divide sample delay by sampling rate
        peak_values = crosscor[peaks_raw]
        
        multich_tdoas[ch_pair] = []
        for peak_raw, peak_sec, peakval in zip(peaks_raw, cc_delay_sec, peak_values):
            peak_tuple = (peak_raw, peak_sec, peakval)
            multich_tdoas[ch_pair].append(peak_tuple)
    return multich_tdoas

def get_peaks(X,  **kwargs):
    '''
    Uses scipy.find_peaks

    Parameters
    ----------
    X : np.array
        1D signal 
    fs : int
        Sampling rate in Hz. The default is 192000.
    min_height : float, optional
        Sets threshold to the value given in min_height
    pctile_thresh : 0<float<100, optional
        Sets threshold according to the percentile of the signal X. 

    Keyword Arguments
    -----------------
    min_peak_diff : float>0
        Distance between peaks in seconds. 
    min_height : float>0, optional 
        The absolute minimum height a peak needs to be. This sets the absolute
        threshold for peak detection. 
    pctile_thresh : 0<float<100, optional 
        The percentile of input signal X that is used to set the threshold.

    Returns
    -------
    np.array
        Array with sample indices of peaks.
    Notes
    -----
    Only one of ```min_height``` or ```pctile_thresh``` should be used - both
    can't be used in the same function call.
    '''
    abs_and_relative_thresh = [each in kwargs.keys() for each in ['min_height','pctile_thresh']]
    if np.sum(abs_and_relative_thresh)==2:
        raise ValueError('Both absolute and percentile thresholds given!')
    min_peak_diff = kwargs['min_peak_diff']
    if abs_and_relative_thresh[0]:
        min_height = kwargs['min_height']        
    elif abs_and_relative_thresh[1]:
        min_height = np.percentile(X, kwargs['pctile_thresh'])
    return signal.find_peaks(X, min_height, distance=int(kwargs['fs']*min_peak_diff))[0] 
      

def get_percentile_peaks(X, **kwargs):
    '''


    Parameters
    ----------
    X : np.array
    pctile_thresh : float>0
        Percentile threshold for peak detection. Defaults to 95
    min_peak_diff : float>0
        Minimum difstance in seconds between peaks. Defaults to 
        0.1 ms. 
    fs : int>0
        Sampling frequency. 

    Returns
    -------
    peak_indices : np.array
        Peak indices in sample.
    '''
    min_peak_diff = kwargs.get('min_peak_diff', 1e-4)
    pctile_thresh = kwargs.get('pctile_thresh', 95)
    min_height = np.percentile(X, pctile_thresh)
    return signal.find_peaks(X, min_height, distance=int(kwargs['fs']*min_peak_diff))[0] 

def geometrically_valid(multich_tdoas:dict, **kwargs):
    '''
    Checks that all time differences are <= max delay defined by inter-mic
    distance.

    Parameters
    ----------
    multich_tdoas : dict
        Dictionary with channel pair keys (e.g.(1,0)) and entry is a list
        with tuples. Each tuple is a peak detection with 
        peak sample position, peak time position, peak value.
    array_geom: (Mmics,3) np.array
        XYZ coordinates of M microphones
    v_sound: float, optional 
        speed of sound in m/s. Default is 340 m/s.

    Returns
    -------
    geom_valid_tdoas : dict
        Keys are channel pair number e.g. (0,2) and entries are lists with
        tuples for each peak detection. 
    
    See Also
    --------
    get_multich_tdoas
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
        for peak_details in pair_tdoas:
            peak_delay = peak_details[1]
            if abs(peak_delay)<=mic2mic_delay:
                geom_valid_tdoas[ch_pair].append(peak_details)
    return geom_valid_tdoas

def get_positive_aa_peaks(multich_aa):
    pos_multich_aa = {}
    for channel, aa_peaks in multich_aa.items():
        pos_multich_aa[channel] = []
        for each in aa_peaks:
            _, delay, _ = each
            if delay>0.0:
                pos_multich_aa[channel].append(each)
    return pos_multich_aa

if __name__ == '__main__':
    from simdata import simulate_1source_and_1reflector
    audio, _, _, _ = simulate_1source_and_1reflector()
    multich_cc = generate_multich_crosscorr(audio, use_gcc=True)
    multich_ac = generate_multich_autocorr(audio)
    
    cc_peaks = get_multich_tdoas(multich_cc, min_height=2, fs=192000)
    #%%
    multiaa = get_multich_aa_tdes(multich_ac, fs=192000, min_height=2)
    #%%
    # plot all the cross-cor peaks
    uu = {}
    for k, pks in cc_peaks.items():
        uu[k] = [each[1]*10**3 for each in pks]
    #%%
    multich_gcc = generate_multich_crosscorr(audio, gcc=True, gcc_variant='roth')
    multich_acc = generate_multich_autocorr(audio, gcc=False, gcc_variant='roth')
    get_percentile_peaks(multich_gcc[(1,0)], fs=192000)
    

    
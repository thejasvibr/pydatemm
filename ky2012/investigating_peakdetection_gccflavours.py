# -*- coding: utf-8 -*-
"""
Comparing GCC and flavours for good peak detection
==================================================
Getting good TDEs depends on having good peaks *and* having a good resolution
for peak detection. Standard cross-correlation doesn't seem to work so well, 
and here I'm going to try out the various GCC variants and compare their perfo-
rmance.

"""
from itertools import combinations
from investigating_tdes_in_reverb import *

def make_gcc_for_all_chpairs(multich_audio):
    '''
    Parameters
    ----------
    multich_audio : np.array
    gcc_flavour : str
        One of ['phat', 'ht', 'roth', 'cc']

    Returns
    -------
    multich_gcc : dict
        Dictionary with channel-pair as key and np.array as value.   
    '''
    ch_pairs = combinations(range(multich_audio.shape[1]),2)
    multich_gcc = {}
    for ch_pair in ch_pairs:
        b,a = sorted(ch_pair, reverse=True)
        multich_gcc[(b,a)] = GCC(multich_audio[:,b],multich_audio[:,a])
    return multich_gcc

def multich_gcc_w_max_delay(multich_audio, mic_geom):
    '''
    Combines the gcc object with the expected max delays given 
    distance between two microphones.
    '''
    multich_gcc = make_gcc_for_all_chpairs(multich_audio)
    multich_gcc_w_maxdelay = {}
    for ch_pair, gcc_obj  in multich_gcc.items():
        chpair_maxdelay = max_interch_delay(mic_geom, ch_pair)
        multich_gcc_w_maxdelay[ch_pair] = (gcc_obj, chpair_maxdelay)
    return multich_gcc_w_maxdelay

def multich_peak_detect(multich_gcc, peak_thresholds, **kwargs):
    '''
    See peak_detect_along_gcc_flavours
    '''
    multich_peaks_by_flavour = {}
    for ch_pair, gcc_w_maxdelay in multich_gcc.items():
        multich_peaks_by_flavour[ch_pair]  = peak_detect_along_gcc_flavours(gcc_w_maxdelay,
                                                                            peak_thresholds,
                                                                            **kwargs)
    return multich_peaks_by_flavour

def peak_detect_along_gcc_flavours(gcc_w_delay, peak_thresh, **kwargs):
    '''

    Parameters
    ----------
    gcc_w_delay : dict
        Dictionary with (chB, chA) as keys and values with a tuple of (GCC object,
                                                                       max_delay)
    peak_thresh : list
        List with various percentile threshold
    fs : int
        Freq. sampling, Hz
    distance : int
        Distance bw detected peaks

    Returns
    -------
    method_thresh_results : dict
        Keys are <method_pctile> and values are np.arrays with peak indices
    '''
    method_thresh_results = {}
    gcc_obj, max_delay = gcc_w_delay
    method_names = ['cc', 'phat', 'ht', 'roth', 'scot']
    fs = kwargs['fs']
    gcc_methods = [gcc_obj.cc(), gcc_obj.phat(), gcc_obj.ht(),
                   gcc_obj.roth(), gcc_obj.scot()]
    for (name, method) in zip(method_names,gcc_methods):
        for pctile in peak_thresh:
            halfway = int(method.sig.size*0.5)
            start,stop = halfway-int(max_delay*fs), halfway+int(max_delay*fs)
            relevant = method.sig[start:stop]
            peaks, _ = signal.find_peaks(relevant, height=np.percentile(relevant, pctile),
                                         distance=kwargs['distance'])
            peaks += start
            method_thresh_results[f'{name}_{str(pctile)}'] = peaks
    return method_thresh_results

def expected_peaks_one_chpair(chpair, sources, mic_geom, **kwargs):
    b,a = chpair
    fs = kwargs['fs']
    predicted_tdes  = expected_tdes(sources, mic_geom[b,:], mic_geom[a,:])
    pred_tdes_samples = kwargs['audiosize']+np.int64(np.array(predicted_tdes)*fs)
    return pred_tdes_samples
    
def multich_expected_peaks(input_audio, sources, mic_geom, **kwargs):
    '''
    Parameters
    ---------
    input_audio : np.array
    sources : list with sub-lists
    mic_geom : np.array
    Keyword Arguments
    fs : int
    
    Returns
    -------
    chpair_peaks : dict
        (ch_pair) key with np.array of predicted peak indices
    '''
    nchannels = input_audio.shape[1]
    chpairs = combinations(range(nchannels), 2)
    chpair_peaks = {}
    kwargs['audiosize'] = input_audio.shape[0]
    for chpair in chpairs:
        ch_pair = tuple(sorted(chpair, reverse=True))
        chpair_peaks[ch_pair] = expected_peaks_one_chpair(ch_pair, sources,
                                                 mic_geom, **kwargs)
    return chpair_peaks

min_rms_error = lambda X,Y : np.min(np.sqrt((Y-X)**2))

def min_peak_residual(expected_peaks, obtained_peaks):
    best_fits = []
    for expected in expected_peaks:
        residual = min_rms_error(obtained_peaks, expected)
        best_fits.append(residual)
    return best_fits 
        

if __name__ == "__main__":
    array_xyz = pd.read_csv('../pydatemm/tests/scheuing-yang-2008_micpositions.csv').to_numpy()
    fs = 192000
    kwargs = {'fs':fs, 'ray_trace':False}   
    simaudio_direct, sources = audiosim_room(0, array_xyz, **kwargs)
    kwargs['ray_trace'] = True
    simaudio_1ref, sources = audiosim_room(2, array_xyz, **kwargs)
    #%% 
    kwargs['distance'] = int(fs*1e-4)
    input_audio = simaudio_1ref
    all_gcc = multich_gcc_w_max_delay(input_audio, array_xyz)
    method_thresh_peaks = multich_peak_detect(all_gcc,  [95, 99, 99.5],
                                              **kwargs)
    #%% Now get the expected TDE peak indices for each channel pair, given 
    # the source and sensor positions.
    all_chpair_exppeaks = multich_expected_peaks(input_audio, sources, array_xyz, 
                                                 **kwargs)
    #%% Calculate the 
    df = pd.DataFrame(data=[], columns = ['ch_pair','method','thresh',
                                          'resid1','resid2', 'resid3', 'resid4'])
    method_thresh_resid = []
    for ch_pair, exp_peaks in all_chpair_exppeaks.items():
        for methods, peaks in method_thresh_peaks[ch_pair].items():
            key_name = str(ch_pair)+'-'+methods
            df = pd.DataFrame(data=[], index=[0], columns = ['ch_pair','method','thresh',
                                                  'total_npeaks',
                                                  'resid1','resid2', 'resid3', 'resid4',
                                                  ])
            gcc_variant, thresh = methods.split('_')
            df['ch_pair'] = str(ch_pair)
            df['method'] = gcc_variant
            df['thresh'] = thresh
            df['total_npeaks'] = peaks.size
            df.loc[0,'resid1':]= min_peak_residual(exp_peaks, peaks)
            method_thresh_resid.append(df)
    # Combine it all 
    peak_resid = pd.concat(method_thresh_resid).reset_index(drop=True)
    peak_resid['max_res'] = peak_resid.loc[:,'resid1':].apply(lambda X: np.max(X),1)
    peak_resid['mean_res'] = peak_resid.loc[:,'resid1':].apply(lambda X: np.mean(X),1)
    peak_resid['sum_res'] = peak_resid.loc[:,'resid1':].apply(lambda X: np.sum(X),1)
    #%% 
    plt.figure()
    plt.plot(peak_resid['sum_res'])
    plt.ylim(1400,0)
    plt.hlines(10,0,335,'r')
    #%% 
    # good_detections = peak_resid[peak_resid['sum_res']<=80]
    import sklearn 
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder
    
    y = peak_resid['sum_res'].to_numpy()
    x_df = peak_resid.loc[:, ['method', 'thresh']]
    x_labeled = pd.get_dummies(data=x_df, drop_first=True)
    x_train, x_test, y_train, y_test = train_test_split(x_labeled.to_numpy(), y, test_size = 0.1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    yy = pd.DataFrame(data=model.coef_).T
    yy.columns = x_labeled.columns
    #%% 
    # The (not so) great regression seems to indicate that the methods in order of 
    # performance are: PHAT, Roth, HT, CC
    # The thresholds in order of performance are: 95, 99, 99.5
    # Now let's check out the overall number of peaks we get when we have PHAT x 95 %ile
    for m in ['cc','scot','phat','roth','ht']:
        sub_rows = np.logical_and(peak_resid['method']==m, peak_resid['thresh']=='95')
        peak_subdf = peak_resid[sub_rows]
        print(f'method: {np.unique(peak_subdf["method"])}',peak_subdf['sum_res'].sum())
    #%% My own analysis broadly based on looking at sum of sum_residuals across
    # all detectiosn suggests that SCOT and PHAT perform with the same level (at 95%ile threshold)
    

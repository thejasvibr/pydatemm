'''
Raster matching 
===============
Implements raster matching as described in Scheuing & Yang 2008
In specific see description of raster matching in Section IV A & V. 

References
----------

* Scheuing, J., & Yang, B. (2008). Disambiguation of TDOA estimation for
  multiple sources in reverberant environments. IEEE transactions on audio,
  speech, and language processing, 16(8), 1479-1489.
'''
import numpy as np 
from itertools import combinations
from pydatemm.timediffestim import geometrically_valid

def multichannel_raster_matcher(multich_Pkl, multich_aa, **kwargs):
    '''
    Performs raster matching on multichannel crosscor/autocor input.
    Also eliminates geometrically invalid TDOAs which exceed max expected
    mic-to-mic straight-line delay.

    Parameters
    ----------
    multich_Pkl,multich_aa : dict
        Keys are channel pairs/channel IDs and entry is a list with tuples
        containing peaks.
    array_geom : np.array

    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    multich_Pprime_kl : dict
        Keys are channel pairs. Entries are lists with tuples.
        Each tuple holds one TDOA peak with the 4th entry being
        its quality score. 
    '''
    # keep only geometrically valid TDOAs across channels
    geomvalid_Pkl = geometrically_valid(multich_Pkl, **kwargs)

    multich_Pprime_kl = {}
    for ch_pair, Pkl in geomvalid_Pkl.items():
        ch1, ch2 = ch_pair
        Pkk, Pll = multich_aa[(ch1,ch1)], multich_aa[(ch2,ch2)]
        Pprime_kl = channel_pair_raster_matcher(Pkl, Pkk, Pll, **kwargs)
        multich_Pprime_kl[ch_pair] = Pprime_kl
    return multich_Pprime_kl

def channel_pair_raster_matcher(Pkl, Pkk, Pll, **kwargs):
    '''
    Compares TDOAs in a channel pair and filters out spurious TDOA peaks caused
    by echo paths.
    Cross-correlation pairs whose time-difference corresponds to an auto-corre-
    lation peak in either Pkk or Pll are evaluated further. If a peak is an 
    'arrow-tail' (the earlier cross-cor peak of a pair), then its score incre-
    ases. If a peak is an 'arrow-head' (the later cross-cor peak of a pair), 
    then its score decreases. Peaks whose score drops beyond their original 
    peak value are eliminated.

    Parameters
    ----------
    Pkl : list
        List with tuples. Each tuple 
        is a geometrically valid TDOA peak from   cross-correlations
    Pkk,Pll : List
        List with tuples containing valid (positive) autocorr peaks.
    twrm : float
        Tolerance width of raster matching in seconds. 

    Returns
    -------
    Pprime_kl : dict
        A filtered set of cross-correlation peaks. Keys are channel 
        pairs. Entries are TDOA tuples. Each tuple has 4 entries. 
        The 4th entry is the quality score. 
   peak_quality : 

    Notes
    -----
    The quality score is defined in eqn. 13 of Scheuing & Yang 2008. 

    See Also
    --------
    make_Pprime_kl
    
    '''
    Pprime_kk = make_Pprime(Pkk, Pkl, **kwargs)
    Pprime_ll = make_Pprime(Pll, Pkl, **kwargs)
    # Now calculate the quality score 
    peak_quality = calculate_quality_eta_mu(Pkl, Pprime_kk, Pprime_ll, **kwargs)   
    # 
    Pprime_kl = make_Pprime_kl(Pkl, peak_quality)
    return Pprime_kl

def make_Pprime(Paa, Pkl, **kwargs):
    '''
    Keeps all autocorrelation peaks of a channel corresponding to
    TDOA peak differences

    Parameters
    ----------
    Paa : list
        Autocorrelation peaks
        List with tuple entries for a channel. Each tuple identifies an autocorr peak.
    Pkl : list
        Cross-correlation peaks
        List with tuple entries for a channel pair.
    twrm : float
        Tolerance width of raster matching in seconds.

    Returns
    -------
    Pprime : list
        Autocorrelation peaks which are recaptured in the difference
        of cross-cor peaks. List with sub-lists. The list has the following str
        ucture :math:`P'_{kk/ll}` =
        [ [:math:`\eta_{\eta}`, (:math:`\eta_{\mu}, \eta_{\\nu}`) ], ....  ]
        Here :math:`\eta_{\mu},\eta_{\\nu}` are peaks with their associated
        data described by tuples.

    See Also
    --------
    pydatemm.timediffestim.get_multich_tdoas
    pydatemm.timediffestim.get_multich_aa_tdes
    '''
    # 
    peaks_s = [each[1] for each in Pkl] # peak locations in seconds
    cross_cor_combis_inds = list(combinations(range(len(peaks_s)),2))
    cross_cor_combis = [(peaks_s[p1], peaks_s[p2]) for (p1,p2) in cross_cor_combis_inds]
    crosscor_combi_diffs = []
    for (peak1,peak2) in cross_cor_combis:
        peak_diff = abs(peak1-peak2) # abs(eta_mu - eta_gamma)
        crosscor_combi_diffs.append(peak_diff)
    
    # keep all the aa peaks which approx. raster match
    Pprime = []
    for eta_eta in Paa:
        sample, aa_time, rkk = eta_eta
        # calculate all 
        eta_diff = abs(abs(aa_time) - abs(np.array(crosscor_combi_diffs)))
        for i, each_diff in enumerate(eta_diff):
            if each_diff < 0.5*kwargs['twrm']:
                peak1, peak2 = cross_cor_combis_inds[i]
                # sort the peaks by their locations.
                eta_mu = Pkl[peak1]
                eta_nu = Pkl[peak2]
                Pprime.append([eta_eta, (eta_mu, eta_nu)])
    return Pprime

def calculate_quality_eta_mu(Pkl, Pp_kk, Pp_ll, **kwargs):
    '''
    Implements :math:`q(\eta_{mu})` calculation as defined in eqn. 13
    
    Parameters
    ----------
    Pkl: list
        List with cross-correlation peaks
    Pp_kk, Pp_ll : list
        Lists with Pprime_k and Pprime_l 
    twrm : float>0
    
    Returns
    -------
    None.

    Notes
    -----
    
    '''
    eta_mu_w_q = [] # Each peak is still a tuple, but now with an additional 4th 
    # entry - the quality score
    for peak in Pkl:
        _, peak_s, rkl = peak
        quality = rkl
        quality += raster_match_score_v2(peak, Pp_kk, **kwargs)
        quality += raster_match_score_v2(peak, Pp_ll, reverse_order=True, **kwargs)
        peak_props = (_, peak_s, rkl, quality)
        eta_mu_w_q.append(peak_props)
    return eta_mu_w_q

def gamma_tfrm(eta, **kwargs):
    '''
    Parameters
    ----------
    eta : float
        I interpret it to mean the difference 
        between TDEs :math:`\eta= \eta_{\mu}-\eta_{\\nu}` (eqn. 12)
    twrm: float>0
        Tolerance width of raster matching in seconds.

    Returns
    -------
    tfrm_out
    
    Notes
    -----
    This function is defined in eqn. 14 of Scheuing & Yang 2008
    '''
    twrm = kwargs['twrm']
    if abs(eta) < 0.5*twrm:
        tfrm_out = 1 - (abs(eta))/(0.5*twrm)
    elif abs(eta)>= 0.5*twrm:
        tfrm_out = 0
    return tfrm_out

def raster_match_score(eta_mu, Pprimekk, reverse_order=False):
    '''
    Calculates what I call the raster-match score from eqn. 13
    I define raster-match score as
    :math:`\sum_{\eta_{\eta} \in P'_{kk}}^{} sign(\eta_{\mu}-\eta_{\\nu})|r_{kk}(\eta_{\eta})|`
    :math:`\\times \Gamma_{TFRM}(\eta_{\eta} - |\eta_{\mu}-\eta_{\\nu}|)`                 
    
    Parameters
    ----------
    eta_mu : tuple
        The tuple which identifies :math:`\eta_{\mu}`
    Pprimekk : List
        List with sub-lists of the following structure -
            [[(eta_eta), (eta_mu, eta_gamma)], ....]
    reverse_order : bool, optional 
        In equation 13, do :math:`\eta_{\\nu}-\eta_{\\mu}` instead of 
        :math:`\eta_{\mu}-\eta_{\\nu}`. Defaults to False, which then 
        performs :math:`\eta_{\mu}-\eta_{\\nu}`

    Returns
    -------
    raster_match_score : float
    
    See Also
    --------
    make_Pprime
    '''
    # first check if this eta_mu is part of a raster-match at all
    raster_match_score = 0
    for each in Pprimekk:
        eta_eta, (eta_Mu, eta_gamma) = each  # eta_Mu and eta_Nu are those peaks
        # that are raster matched
        eta_Mu_tdoa = eta_Mu[1]
        if eta_Mu_tdoa == eta_mu:
            # if eta_mu is raster-matched, then also include the auto-corr
            # peak coefficient and so on
            delay_mu, delay_gamma = eta_Mu[1], eta_gamma[1]
            # the sign and auto-corr coefficient part
            if reverse_order:
                part1 = np.sign(delay_gamma-delay_mu)*np.abs(eta_eta[-1])
            else:
                part1 = np.sign(delay_mu-delay_gamma)*np.abs(eta_eta[-1])
            # the gamma function part
            part2 = gamma_tfrm(eta_eta[1] - np.abs(delay_mu-delay_gamma))
            part12 = part1*part2
        else:
            part12 = 0
        raster_match_score += part12
    return raster_match_score

def raster_match_score_v2(eta_mu, Pprimekk, **kwargs):
    '''
    '''
    reverse_order = kwargs.get('reverse_order', False)
    _, etamu_s, rkl = eta_mu
    
    summation_score = 0
    for each_pprime in Pprimekk:
        eta_eta, (eta_mu_saved, eta_gamma) = each_pprime
        _, aa_s, rkk = eta_eta 
        if eta_mu_saved==eta_mu:
            etagamma_s = eta_gamma[1]
            peak_match_residual = aa_s - np.abs(etamu_s-etagamma_s)
            if not reverse_order:                
                value = np.sign(etagamma_s-etamu_s)*np.abs(rkk)
            else:
                value = np.sign(etamu_s-etagamma_s)*np.abs(rkk)
            value *= gamma_tfrm(peak_match_residual, **kwargs)
            summation_score += value
    return summation_score

        
    
    

def make_Pprime_kl(Pkl, peak_qualities):
    '''
    Generates subset of Pkl where the peak quality score is >= :math:`t_{kl}`
    :math:`t_{kl}` is defined as the minimum :math:`r_{kl}` value of the peaks
    in Pkl (see text just below eqn. 15)

    Parameters
    ----------
    Pkl : list
        List with tuples. Each tuple contains a TDOA.
    peak_qualities: list
        List with tuples. Each tuple has 4 entries. First 3 define a TDOA
        , the fourth entry is the TDOA's quality score.

    Returns
    -------
    Pprime_kl : list
        List with tuple. Each tuple is a quality filtered TDOA. 
    '''
    rkl_values = [rkl for (_,_,rkl) in Pkl]
    t_kl = np.min(rkl_values)
   
    # keep only those peaks with quality of at least tkl
    Pprime_kl = []
    for peak in peak_qualities:
        quality_score = peak[-1]
        if quality_score>=t_kl:
            Pprime_kl.append(peak)
    return Pprime_kl

if __name__ == '__main__':
    from simdata import simulate_1source_and_1reflector, simulate_1source_and_3reflector
    from pydatemm.timediffestim import *
    from itertools import permutations
    audio, distmat, arraygeom, _ = simulate_1source_and_3reflector()
    fs = 192000
    multich_cc = generate_multich_crosscorr(audio, use_gcc=True)
    multich_ac = generate_multich_autocorr(audio)
    
    cc_peaks = get_multich_tdoas(multich_cc, min_height=2, fs=192000)
    cc_geomvalid = geometrically_valid(cc_peaks, array_geom=arraygeom)
    #%%
    multiaa = get_multich_aa_tdes(multich_ac, fs=192000, min_height=2)
    
    #%%
    # make eta_mu
    ch1, ch2 = 2,1
    Pkl = cc_geomvalid[(ch1,ch2)]
    Paa = multiaa[(ch1,ch1)]
    Pprime_kk = make_Pprime(multiaa[(ch1,ch1)], Pkl, twrm=10/fs)
    Pprime_ll = make_Pprime(multiaa[(ch2,ch2)], Pkl, twrm=10/fs)
    
    # get true tDOA
    true_tdoa = (distmat[0,ch1]-distmat[0,ch2])/340
    print(f'True tDOA: {true_tdoa}')
    #%%
    b = channel_pair_raster_matcher(cc_geomvalid[(3,2)], multiaa[(3,3)], multiaa[(2,2)], twrm=10/fs)
    Q = multichannel_raster_matcher(cc_peaks, multiaa, twrm=10/fs, array_geom=arraygeom)
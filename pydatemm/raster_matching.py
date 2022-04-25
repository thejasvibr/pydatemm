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

def raster_matcher(Pkl, Pkk, **kwargs):
    '''
    Parameters
    ----------
    Pkl : dict
        Keys are channel pairs, entry is a list with tuples. Each tuple 
        is a geometrically valid TDOA peak from   cross-correlations
    Pkk : dict
        Keys are channel numbers repeated twice in a tuple. Entries  are lists
        with tuples containing valid autocorr peaks.

    Returns
    -------
    Pkl_prime : dict
        A filtered set of cross-correlation peaks. Keys are channel 
        pairs. Entries are TDOAs.
    q : dict
        Keys are channel pairs, entries hold the quality score for each 
        TDOA. 
    Notes
    -----
    The quality score is defined in eqn. 13 of Scheuing & Yang 2008. 
    
    '''
    # eqn. 11 defines mu_mu 
    # assemble the autocorr peaks Pkk and Pll into eta_mu
    eta_mu = []
    # keep only the positive peaks of the autocorrelation!
    
    
    # 
    
    
    
    
    
    
    Pkl_prime = []
    
    return Pkl_prime


def gamma_tfrm(eta, **kwargs):
    '''
    Parameters
    ----------
    eta : float
        I interpret it to mean the difference 
        between TDEs :math:`\eta= \eta_{\mu}-\eta_{\\nu}` (eqn. 12)
    tfrm: float>0
        twrm - tolerance width of raster matching

    Returns
    -------
    tfrm_out
    
    Notes
    -----
    This function is defined in eqn. 14 of Scheuing & Yang 2008
    '''
    
    twrm = kwargs.get('twrm', 5.21e-5) # 
    if abs(eta) < 0.5*twrm:
        tfrm_out = 1 - (abs(eta))/(0.5*twrm)
    elif abs(eta)>= 0.5*twrm:
        tfrm_out = 0
    return tfrm_out

def make_Pprime(Paa, Pkl, **kwargs):
    '''
    Parameters
    ----------
    Paa : list
        List with tuple entries for a channel. Each tuple identifies an autocorr peak.
    Pkl : list
        List with tuple entries for a channel pair.
    twrm : float
        Tolerance width of raster matching in seconds.

    Returns
    -------
    Pprime : list
        List with sub-lists. The list has the following structure
        :math:`P'_{kk/ll}` =
        [ [:math:`\eta_{eta}`, (:math:`\eta_{\mu}, \eta_{\\nu}`) ], ....  ]

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
                eta_mu = Pkl[peak1]
                eta_nu = Pkl[peak2]
                Pprime.append([eta_eta, (eta_mu, eta_nu)])
    return Pprime

def calculate_eta_mu_quality(Pkl, Pp_kk, Pp_ll):
    '''
    Implements :math:`q(\eta_{mu})` calculation as defined in eqn. 13

    Parameters
    ----------

    Returns
    -------
    None.

    Notes
    -----
    
    '''
    eta_mu_w_q = [] # Each peak is still a tuple, but now with an additional 4th 
    # entry - the quality score
    for peak in Pkl:
        _, eta_mu, rkl = peak
        quality = rkl + raster_match_score(eta_mu, Pp_kk) + raster_match_score(eta_mu, Pp_ll)
        peak_props = (_, eta_mu, rkl, quality)
        eta_mu_w_q.append(peak_props)
    return eta_mu_w_q

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
    Pprimekk
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
        in_rastermatch = False
        eta_eta, (eta_Mu, eta_Nu) = each
        if eta_mu==eta_Mu:
            in_rastermatch = True
            
        if not in_rastermatch:
            # if eta_mu is not raster-matched
            part12 = 0
        else:
            # if eta_mu is raster-matched, then also include the auto-corr
            # peak coefficient and so on
            delay_mu = eta_Mu[1]
            delay_nu = eta_Nu[1]
            if not reverse_order:
                part1 = np.sign(delay_nu-delay_mu)*np.abs(eta_eta[-1])
            else:
                part1 = np.sign(delay_mu-delay_nu)*np.abs(eta_eta[-1])
            part2 = gamma_tfrm(eta_eta[1] - np.abs(delay_mu-delay_nu))
            part12 = part1*part2
        raster_match_score += part12
    return raster_match_score

    


if __name__ == '__main__':
    from simdata import simulate_1source_and_1reflector, simulate_1source_and_3reflector
    from pydatemm.timediffestim import *
    from itertools import permutations
    audio, _, arraygeom, _ = simulate_1source_and_3reflector()
    fs = 192000
    multich_cc = generate_multich_crosscorr(audio, use_gcc=True)
    multich_ac = generate_multich_autocorr(audio)
    
    cc_peaks = get_multich_tdoas(multich_cc, min_height=2, fs=192000)
    cc_geomvalid = geometrically_valid(cc_peaks, array_geom=arraygeom)
    #%%
    multiaa = get_multich_aa_tdes(multich_ac, fs=192000, min_height=2)
    pos_multiaa = get_positive_aa_peaks(multiaa)
    #%%
    # make eta_mu
    ch1, ch2 = 2,1
    Pkl = cc_geomvalid[(ch1,ch2)]
    Paa = pos_multiaa[(ch1,ch1)]
    Pprime_kk = make_Pprime(pos_multiaa[(ch1,ch1)], Pkl, twrm=10/fs)
    Pprime_ll = make_Pprime(pos_multiaa[(ch2,ch2)], Pkl, twrm=10/fs)
    print(Pkl, Pprime_kk, Pprime_ll)
    
    #%% 
    # Now calculate the quality score 
    
    
    
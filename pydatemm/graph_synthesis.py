'''
Graph synthesis
===============
Builds up larger graphs using approximately consistent triplets.

Implements S2-8 of Table 1.

Reference
---------
* Scheuing & Yang 2008, ICASSP
'''








def triplet_quality(triplet, **kwargs):
    '''
    Calculates triplet quality score- which is the product of the 
    TFTM output and the sum of individual TDOA qualities.
    This metric is defined in eqn. 23
    '''
    triplet_name, t12, t23, t31 = triplet
    tdoa_quality_sum = t12[1] + t23[1] + t31[1]
    tdoa_tftm_score = gamma_tftm(t12[0],t23[0],t31[0], **kwargs)
    quality = tdoa_tftm_score*tdoa_quality_sum
    return quality 

def gamma_tftm(tdoa_ab, tdoa_bc, tdoa_ca,**kwargs):
    '''
    Calculates the tolerance width of triple match.
    
    Parameters
    ----------
    tdoa_ab,tdoa_bc,tdoa_ca: float
    twtm : float
        Tolerance width of triple match
    Returns
    -------
    twtm_out : float
        Final score
    '''
    residual = tdoa_ab + tdoa_bc + tdoa_ca
    twtm = kwargs['twtm']
    if abs(residual) < 0.5*twtm:
        twtm_out = 1 - (abs(residual))/(0.5*twtm)
    elif abs(residual)>= 0.5*twtm:
        twtm_out = 0
    return twtm_out

if __name__ == '__main__':
    from simdata import simulate_1source_and_3reflector
    from pydatemm.timediffestim import *
    from pydatemm.raster_matching import multichannel_raster_matcher
    from pydatemm.triple_generation import mirror_Pprime_kl, generate_consistent_triples
    from itertools import permutations
    audio, distmat, arraygeom, _ = simulate_1source_and_3reflector()
    fs = 192000
    multich_cc = generate_multich_crosscorr(audio, use_gcc=True)
    multich_ac = generate_multich_autocorr(audio)
    cc_peaks = get_multich_tdoas(multich_cc, min_height=2, fs=192000)
    multiaa = get_multich_aa_tdes(multich_ac, fs=192000, min_height=2) 
   
    tdoas_rm = multichannel_raster_matcher(cc_peaks, multiaa, twrm=10/fs, array_geom=arraygeom)
    tdoas_mirrored = mirror_Pprime_kl(tdoas_rm)
    true_tdoas = {}
    for chpair, _ in cc_peaks.items():
        ch1, ch2 = chpair
        # get true tDOA
        tdoa = (distmat[0,ch1]-distmat[0,ch2])/340
        true_tdoas[chpair] = tdoa
    #%%
    # Now get all approx consistent triples
    consistent_triples = generate_consistent_triples(audio.shape[1], tdoas_mirrored, twtm=1e-5)
    [triplet_quality(each, twtm=1e-5) for each in consistent_triples]
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigating the pre-processing of cross-correlations for peak detection
=========================================================================
I've been observing that a lot of my bad source localisations are due to missing
TDEs already from the first stage itself. If the correct TDE peaks are not 
detected - of course the correct graphs can't be combined - and thus we 
get a poor localisation. 

Till now I've been implementing a simple peak detection routine - where the 
percentile of the relevant cc portion (+/- max delay) is used to set the thres-
hold for peak detection. A minimum distance is defined between detected peaks -
and this actually works much of the time. However, the details make all the
difference!

Here I'll be comparing TDE peak detection with the current implementation and 
the 'noneg' version. 

The 'noneg' peak detector
~~~~~~~~~~~~~~~~~~~~~~~~~
Nothing special here really. The CC output often has -ve values. The -ve values
are of no interest to us - as we're looking for +ve samples, and preferrably high
positive samples.

The idea is to do two things: 
    1. Set all -ve samples to 0
    2. See which threshold values typically make sense (I doubt that 95 %ile would
                                                        still work)
"""
from ky2012_fullsim_chain import * 
from pydatemm.timediffestim import get_peaks
from pydatemm.timediffestim import max_interch_delay as maxintch
# print the expected TDE peaks given the sources and array geom
print(exp_tdes_multich)
# print the TDE min-residuals from the current method with just plain percentile
print(residual_chpairs)

def noneg_get_peaks(X, **kwargs):
    '''
    See pydatemm.timediffestim.get_peaks
    '''
    # replace all -ve values with 0s
    Y = X.copy()
    Y[Y<=0] = 0
    return get_peaks(Y, **kwargs)

edges = list(map(lambda X: tuple(sorted(X, reverse=True)),
                 combinations(range(sim_audio.shape[1]),2)))
for chpair in edges:
    max_sample = int(maxintch(chpair, kwargs['array_geom'])*fs)
    minmaxsample =  np.int64(sim_audio.shape[0] + np.array([-max_sample, max_sample]))
    relevant_cc = multich_cc[chpair][minmaxsample[0]:minmaxsample[1]]
    noneg_peaks = noneg_get_peaks(relevant_cc, **kwargs)
    noneg_peaks += minmaxsample[0]

# -*- coding: utf-8 -*-
"""
Utility functions to get pre-remphasis cross-cors

Created on Fri Aug  4 16:32:55 2023

@author: theja
"""
import numpy as np 
from scipy.interpolate import interp1d
import pandas as pd
from itertools import combinations
import sys
sys.path.append('../')
from source_traj_aligner import calculate_toa_channels



def make_crosscor_chpairs(nchannels):
    unique_pairs = list(combinations(range(nchannels), 2))
    unique_pairs = list(map(lambda X: sorted(X, reverse=True), unique_pairs))
    return unique_pairs

    

def upsample_xyz_to_ms_resolution(traj_df, final_res=1e-3):
    bybat = traj_df.groupby('batid')
    all_bats_interp = []
    for batid, subdf in bybat:
        tmin,tmax = np.percentile(subdf['t'], [0,100])
        t_highres = np.arange(tmin,tmax+final_res,final_res)
        t_highres = t_highres[np.logical_and(t_highres>=tmin, t_highres<=tmax)]
        fitted_fn = {axis: interp1d(subdf['t'], subdf[axis])for axis in ['x','y','z']}
        interp_data = pd.DataFrame(data=[], columns=['batid','x','y','z','t'])
        interp_data['t'] = t_highres
        interp_data['batid'] = batid
        for each in ['x','y','z']:
            interp_data[each] = fitted_fn[each](t_highres)
        all_bats_interp.append(interp_data)
    return pd.concat(all_bats_interp, axis=0).reset_index(drop=True)


def get_chpair_tdes_from_sources(call_pointsdf, highres_flighttraj, array_geom):
    '''
    Generates all expected TDEs from sources given a high-res flight trajectory 
    and call-point

    Parameters
    ----------
    call_pointsdf : pd.DataFrame
        With columns x,y,z,t,batid
    highres_flighttraj: pd.DataFrame
        With columns x,y,z,t,batid - and with high temporal resolution preferably
    array_geom : (Mmics,3) np.array
        xyz coordinates of the array
    
    Returns 
    -------
    expected_tdoas : dict
        Dictionary with channel pair as a tuple and np.array with expected time-delays
        in seconds. 
    '''
    nchannels = array_geom.shape[0]
    unique_pairs = list(combinations(range(nchannels), 2))
    unique_pairs = list(map(lambda X: sorted(X, reverse=True), unique_pairs))
    expected_tdoa = {tuple(chpair) : [] for chpair in unique_pairs}
    for idx, row in call_pointsdf.iterrows():
        toa_set = calculate_toa_channels(row['t'], highres_flighttraj, row['batid'], array_geom).flatten()
        for chpair, tdoas in expected_tdoa.items():
            chb,cha = chpair
            tde = toa_set[chb]-toa_set[cha] 
            tdoas.append(tde)
    return expected_tdoa
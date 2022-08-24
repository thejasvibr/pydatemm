# -*- coding: utf-8 -*-
"""
Kreissig-Yang CombineAll simulated data with full chain
=======================================================

Created on Wed Aug  3 10:28:28 2022

@author: theja
"""

from synthetic_data_generation import * 
from pydatemm.tdoa_quality import residual_tdoa_error
from pydatemm.timediffestim import generate_multich_autocorr, generate_multich_crosscorr
from pydatemm.timediffestim import geometrically_valid, get_multich_tdoas
from pydatemm.timediffestim import get_multich_aa_tdes
from pydatemm.raster_matching import multichannel_raster_matcher
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
import pandas as pd
import pyroomacoustics as pra
import scipy.signal as signal 
import time 
from pydatemm.localisation_mpr2003 import mellen_pachter_raquet_2003
from copy import deepcopy
# %load_ext line_profiler
#%% Generate simulated audio
array_geom = pd.read_csv('../pydatemm/tests/scheuing-yang-2008_micpositions.csv').to_numpy()
# from the pra docs
room_dim = [9, 7.5, 3.5]  # meters
fs = 192000
ref_order = 1

reflection_max_order = ref_order
# room = pra.ShoeBox(
#     room_dim, fs=fs, max_order=reflection_max_order
# )
room = pra.ShoeBox(
    room_dim, fs=fs, materials=pra.Material('rough_concrete'),
    max_order=ref_order,
    ray_tracing=False,
    air_absorption=True)

call_durn = 5e-3
t_call = np.linspace(call_durn, int(fs*call_durn))
batcall = signal.chirp(t_call, 9000, t_call[-1], 85000,'linear')
batcall *= signal.hamming(batcall.size)
batcall *= 0.6
sources = [[2.5, 5, 2.5],
           [4, 3, 1.5],
           [1, 4, 1.0],
           [2,4,2]]

delay = np.linspace(0,0.050,len(sources))
for each, emission_delay in zip(sources, delay):
    room.add_source(position=each, signal=batcall, delay=emission_delay)

room.add_microphone_array(array_geom.T)
room.compute_rir()
print('room simultation started...')
room.simulate()
print('room simultation ended...')
sim_audio = room.mic_array.signals.T
nchannels = array_geom.shape[0]

import soundfile as sf
sf.write(f'simaudio_reflection-order_{ref_order}.wav', sim_audio, samplerate=fs)

mic2sources = [mic2source(each, array_geom) for each in sources]    
delta_tdes = [np.zeros((nchannels, nchannels)) for each in range(len(mic2sources))]

for i,j in product(range(nchannels), range(nchannels)):
    for source_num, each in enumerate(delta_tdes):
        each[i,j] = mic2sources[source_num][i]-mic2sources[source_num][j] 
        each[i,j] /= vsound


paper_twrm = 32/fs
paper_twtm = 32/fs
kwargs = {'twrm': paper_twrm,
          'twtm': paper_twtm,
          'nchannels':nchannels,
          'fs':fs,
          'array_geom':array_geom}

# Estimate inter-channel TDES
multich_cc = generate_multich_crosscorr(sim_audio, use_gcc=True)
multich_ac = generate_multich_autocorr(sim_audio)
multiaa = get_multich_aa_tdes(multich_ac, fs=192000,
                              min_height=0.1) 
cc_peaks = get_multich_tdoas(multich_cc, min_height=0.15, fs=192000,
                             min_peak_diff=1e-4)

valid_tdoas = multichannel_raster_matcher(cc_peaks, multiaa,
                                       **kwargs)

# remove impossible pairwise TDOAs based on array geometry
# valid_tdoas = geometrically_valid(cc_peaks, array_geom=array_geom, vsound=343)

# choose only K=5 (top 5)
K = 10
top_K_tdes = {}
for ch_pair, tdes in valid_tdoas.items():
    descending_quality = sorted(tdes, key=lambda X: X[2], reverse=True)
    top_K_tdes[ch_pair] = []
    for i in range(K):
        try:
            top_K_tdes[ch_pair].append(descending_quality[i])
        except:
            pass


ch_pait = (7,0)
plt.figure()
plt.plot(multich_cc[ch_pait])
for i, tde in enumerate(top_K_tdes[ch_pait]):
    plt.plot(tde[0], tde[2], '*')
    plt.text(tde[0], tde[2]+0.001, str(i))
for possible in valid_tdoas[ch_pait]:
    plt.plot(possible[0], possible[2],'o')

#%%
cc_ch = multich_cc[(1,0)]
t_cc = np.linspace(-0.5*cc_ch.size/fs,0.5*cc_ch.size/fs,cc_ch.size)
plt.figure()
plt.plot(t_cc, cc_ch)
#%% create cFLs from TDES
# First get all Fundamental Loops
def make_consistent_fls(multich_tdes, nchannels, **kwargs):
    '''

    Parameters
    ----------
    multich_tdes : TYPE
        DESCRIPTION.
    nchannels : int>0
    max_loop_residual : float>0, optional 
        Defaults to 1e-6

    Returns
    -------
    cFLs : list
        List with nx.DiGraphs of all consistent FLs
    '''
    max_loop_residual = kwargs.get('max_loop_residual', 1e-6)
    all_edges_fls = make_edges_for_fundamental_loops(nchannels)
    all_cfls = []
    for fundaloop, edges in all_edges_fls.items():
        #print(fundaloop)
        a,b,c = fundaloop
        ba_tdes = multich_tdes[(b,a)]
        ca_tdes = multich_tdes[(c,a)]
        cb_tdes = multich_tdes[(c,b)]
        abc_combinations = product(ba_tdes, ca_tdes, cb_tdes)
        for (tde1, tde2, tde3)in abc_combinations:
            if abs(tde1[1]-tde2[1]+tde3[1]) < max_loop_residual:
                this_cfl = nx.ordered.Graph()
                for e, tde in zip(edges, [tde1, tde2, tde3]):
                    #print(e, tde)
                    this_cfl.add_edge(e[0], e[1], tde=tde[1])
                    all_cfls.append(this_cfl)
    return all_cfls


def get_compatibility(cfls, ij_combis):
    output = []
    for (i,j) in ij_combis:
        trip1, trip2  = cfls[i], cfls[j]
        cc_out = ccg_definer(trip1, trip2)
        output.append(cc_out)
    return output

def make_ccg_pll(cfls, **kwargs):
    '''Parallel version of make_ccg_matrix'''
    num_cores = kwargs.get('num_cores', joblib.cpu_count())
    num_cfls = len(cfls)
    cfl_ij_parts = [list(combinations(range(num_cfls), 2))[i::num_cores] for i in range(num_cores)]
    compatibility = Parallel(n_jobs=num_cores)(delayed(get_compatibility)(cfls, ij_parts)for ij_parts in cfl_ij_parts)
    ccg = np.zeros((num_cfls, num_cfls))
    for (ij_parts, compat_ijparts) in zip(cfl_ij_parts, compatibility):
        for (i,j), (comp_val) in zip(ij_parts, compat_ijparts):
            ccg[i,j] = comp_val
    # make symmetric
    ccg += ccg.T
    return ccg

if __name__ == '__main__':
    print('making the cfls...')
    cfls_from_tdes = make_consistent_fls(top_K_tdes, nchannels,
                                         max_loop_residual=0.5e-3)
    cfls_from_tdes = list(set(cfls_from_tdes))
    output = cfls_from_tdes[::]
    print(f'# of cfls in list: {len(output)}')
    start = time.perf_counter()
    if len(output) < 500:
        ccg_pll = make_ccg_matrix(output)
        #stop_time_normal = time.perf_counter()
    else:
        ccg_pll = make_ccg_pll(output)
        #stop_time_pll = time.perf_counter()
#        assert np.all(ccg_pll == ccg_matrix)
    print('done making the cfls...')
    #print(f'Normal run time: {stop_time_normal-start}, Pll run time: {stop_time_pll-stop_time_normal}')
    #%% generate CCG from cFLs
    ccg_matrix = make_ccg_matrix(cfls_from_tdes)
    print('..making the ccg matrix')
    smaller_cflset = cfls_from_tdes[::]
    # print(f'cflsets: {len(smaller_cflset)}')
    # smaller_ccg = make_ccg_matrix(smaller_cflset) 
    np.savetxt('flatA.txt',ccg_pll.flatten(),delimiter=',',fmt='%i')
    # print('..done making the ccg matrix')
    # #%%
    # Call the ui_combineall exe implemented in Cpp
    import os, platform
    if platform.system() == 'Windows':
        os.system('ui_combineall.exe flatA.txt')
    elif platform.system()=='Linux':
        os.system('./ui_combineall flatA.txt')

    # #%%
    # Load the 'jagged' csv file 
    output_file = 'combineall_solutions.csv'
    import csv
    comp_cfls = []
    with open(output_file, 'r') as ff:
        csvfile = csv.reader(ff, delimiter=',')
        for lines in csvfile:
            fmted_lines = [int(each) for each in lines if not each=='']
            comp_cfls.append(fmted_lines)
    #%% Solve CCG to get compatible cFL graphs
    # print(f'...solving the CCG ')
    # start = time.perf_counter_ns()
    # qq_combined = combine_all(smaller_ccg, set(range(len(smaller_ccg))), set([]), set([]))    
#     # durn = time.perf_counter_ns()-start
    # print(f'...done solving the CCG. time taken: {durn/1e9}')
    # comp_cfls = format_combineall(qq_combined)
    # #%%
    # # %lprun -f combine_all combine_all(smaller_ccg, set(range(len(smaller_ccg))), set([]), set([])) 
    # #%% Join compatible graphs and localise candidate sources
    import tqdm
    unique_positions = []
    tdoa_error = []
    only_one_cfl = 0
    less_than_5ch = 0
    leq_4ch_sources = {}
    for i, compat_cfls in enumerate(tqdm.tqdm(comp_cfls)):
        if not len(compat_cfls) <2:
            source_cfls = [smaller_cflset[each] for each in compat_cfls]
            s1_composed = combine_compatible_triples(source_cfls)
            s1c_tde = nx.to_numpy_array(s1_composed, weight='tde')
            channels = list(s1_composed.nodes)
            if len(channels) >=5:
                localised_source = spiesberger_wahlberg_solution(array_geom[channels,:],s1c_tde[1:,0]*343)
                if not np.sum(np.isnan(localised_source))>0:
                    
                    if list(localised_source) in unique_positions:
                        pass
                    else:
                        # print(localised_source)
                        unique_positions.append(list(localised_source))
                        error = residual_tdoa_error(s1_composed, localised_source, array_geom[channels,:])
                        tdoa_error.append(error)
                        #print('TDOA error', error)
            else:
                less_than_5ch += 1 
                try:
                    localised_source = mellen_pachter_raquet_2003(array_geom[channels,:], s1c_tde[1:,0]*343)
                    if localised_source.size >0 :
                        leq_4ch_sources[i] = localised_source
                except ValueError:
                    pass
        else:
            only_one_cfl += 1
    #%%
    import scipy.spatial as spl
    localised = pd.DataFrame(unique_positions, columns=['x','y','z'])
    # best fit for each of the sources
    for source in sources:
        error = localised.apply(lambda X: spl.distance.euclidean(X,source),1)
        best_row = np.argmin(error)
        print(f'best fix for {source} is {np.round(localised.loc[best_row,:],2).tolist()}, with: {error[best_row]}')
        print(f'TDOA residual error is: {tdoa_error[best_row]}')
    #%% also format the <5 channel results
    leq_4ch = []
    for idx, loc_source in leq_4ch_sources.items():
        if np.logical_or(loc_source.shape[0]==1,loc_source.shape[0]==3) :
            leq_4ch.append(loc_source.tolist())
        elif loc_source.shape[0]==2:
            [leq_4ch.append(loc_source[i,:].tolist()) for i in range(2)]
        else:
            raise ValueError
    #%%
    fourch_localised = []
    for key, localised in leq_4ch_sources.items():
        if localised.shape == (3,):
            fourch_localised.append(localised.tolist())
        else:
            for i in range(localised.shape[0]):
                fourch_localised.append(localised[i,:].tolist())
    #%%
    leq_4ch_df = pd.DataFrame(data=fourch_localised)
    # best fit for each of the sources
    for i,source in enumerate(sources):
        error = leq_4ch_df.apply(lambda X: spl.distance.euclidean(X,source),1)
        best_row = np.argmin(error)
        print(f'best fix for {source} is {np.round(leq_4ch_df.loc[best_row,:],2).tolist()}, with: {error[best_row]}')
        #print(f'TDOA residual error is: {tdoa_error[best_row]}')
                            
                

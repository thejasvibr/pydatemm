# -*- coding: utf-8 -*-
"""
Guessing the number of calling bats in audio using Bayesian priors
==================================================================
Here I'll aim to recreate the simplest possible scenario. And also write a 
function which simulates the various possibilities. 

Created on Sun Jul  2 11:57:15 2023

@author: theja
"""
from itertools import combinations
import scipy.signal as signal 
from scipy.spatial import distance_matrix
import numpy as np 
import pymc as pm
import matplotlib.pyplot as plt 

#%% Create mock bat scenarios
# Genreate fake cross-correlation scenarios 
def generate_scenario_number(nbats):
    '''
    Includes a 
    '''
    
    all_scenarios = []
    scenario_number = 0
    for subset in range(1, nbats+1):
        subscenario_combis = combinations(range(1,nbats+1),subset)
        
        for each in subscenario_combis:
            binary_scenario = np.zeros(nbats)
            batinds = np.array((each))-1
            binary_scenario[batinds] = 1 
            all_scenarios.append(binary_scenario )
            scenario_number += 1 
    all_scenarios = np.array(all_scenarios)
    return all_scenarios

def calculate_toa(source, micarray, vsound=343.0):
    '''
    TDOAs calculated wrt to channel 0 in seconds.
    '''
    mic_to_source = distance_matrix(source.reshape(-1,3), micarray).flatten()
    return mic_to_source/vsound
    
def make_mock_audio(scenariodata, vsound):
    '''
    '''
    micarray = scenariodata[-1]
    emittedsound, fs = scenariodata[-2]
    
    mbat_xyz = scenariodata[:-2]
    all_toas = []
    for thisbat in mbat_xyz:
        all_toas.append(calculate_toa(thisbat, micarray, vsound))
    all_toas = np.array(all_toas)
    # create a mock audio 
    max_toa = all_toas.max()
    
def make_sim_cc(sim_audio):
    '''
    Takes in a MsamplesxNchannel np.array and returns 
    a (NxN-1/2) X 2*Msamples-1 np.arrray with row-wise
    cross-correlations of all channel pairs. 
    Each audio channel is normalised so that the max/min value is 1/-1. 
    '''
    norm_audio = np.apply_along_axis(lambda X: X/np.abs(X).max(), 1, sim_audio)
    chpairs = list(combinations(range(norm_audio.shape[1]), 2))
    all_ccs = np.zeros((len(chpairs), norm_audio.shape[0]*2 - 1))
    for i,(chb, cha) in enumerate(chpairs):
        all_ccs[i,:] = signal.correlate(norm_audio[:,chb],norm_audio[:,cha], 'full')
    return all_ccs
       

def generate_scenario_crosscor(all_input_data):
    '''
    Parameters
    ----------
    all_input_data : list
        A list with all relevant info to recreate the necessary simulated
        cross-correlation
    
    Nbats
    scenarionumber
    scenariodict
    scenariodata : list with data for the scenario in order of bat id.
        [bat1xyz, ...batNxyz, emitted_sound, mic_xyz]
    
    Returns
    -------
    sim_crosscor:
    '''
    # batscenario = scenariodict[scenarionumber]
    # emitted_sound, fs = scenariodata[-2]
    
    
    
   
    pass




if __name__ == "__main__":
    import os
    from pytensor.tensor.subtensor import inc_subtensor
    os.environ['PYTHONIOENCODING']='utf-8'    
    #%%
    import pytensor as pt
    nbats = 4
    allscenarios_asnp = generate_scenario_number(nbats)
    num_scenarios = allscenarios_asnp.shape[0]
    p_categories = np.tile(1/num_scenarios, num_scenarios)
    p_categories = p_categories/p_categories.sum()
    scenario_data = {}
    xyz_ranges = np.ones((nbats,6))
    for i in range(6):
        xyz_ranges[:,i] *= i+1
    allscenario_pt = pt.shared(allscenarios_asnp, 'allscenario_pt', shape=allscenarios_asnp.shape)
    
    fs = 192000
    micxyz = np.random.normal(0,1,30).reshape(-1,3)
    
    durn = 5e-3 # seconds
    #emitted_sound = np.random.normal(0,1e-5,5*192)
    emitted_sound = signal.chirp(np.linspace(0,durn,int(fs*durn)),10000,
                                             durn, 90000,'linear')
    emitted_sound *= signal.windows.hann(emitted_sound.size)
    
    
    def generate_fake_cc(rng, scenario_number, xyz_emissions, size=None):
        
        #calling_bats = allscenario_pt[scenario_number,:]        
        print(type(xyz_emissions), xyz_emissions.shape,
              type(scenario_number), scenario_number)
        calling_bats = allscenarios_asnp[scenario_number,:]
        print(calling_bats)
        #
        #inds = np.where(calling_bats)
        #print('HELLO!',scenario_number, type(calling_bats), inds)
        #inds = calling_bats[0].nonzero()[0]
        #emission_hypotheses = pt.tensor.zeros((inds.size,3))
        # emission_hypotheses = inc_subtensor(emission_hypotheses[inds,:],
        #                                     xyz_emissions[inds,: ])
        #print(emission_hypotheses.eval())
        #print(emission_hypotheses.eval(), scenario_number.eval())
        #print( xyz_emissions, emission_hypotheses.shape)
        #return np.array(emission_hypotheses.shape).reshape(-1,2)
        return np.array([5,2]).reshape(-1,2)

    # def generate_fake_np_based(rng, scenario_number, xyz_emissions,  size=None):
    #     '''
    #     '''
    #     calling_bats = allscenarios_asnp[scenario_number,:]
    #     inds = calling_bats.nonzero()[0]
        
    #     # now generate the TOAs of the various emission points
    #     toas = distance_matrix(emission_hypotheses, micxyz)/343.0 # each source has a row. 
    #     audio_samples = int(fs*toas.max())  + emitted_sound.size
    #     audio_data = np.zeros((audio_samples, micxyz.shape[0]))
    #     for row in range(toas.shape[0]):
    #         for ch_num, channel_toa in enumerate(toas[row,:]):
    #             toa_sample = int(fs*channel_toa)
    #             toa_end = toa_sample + emitted_sound.size
    #             audio_data[toa_sample:toa_end, ch_num] += emitted_sound
    #     return audio_data

   
    
    with pm.Model() as mo:
        # choose which scenario is being tested
        current_scenario = pm.Categorical('current_scenario',
                                              p_categories,
                                              shape=(1,))
        #scenario_data[0] = current_scenario
        # generate the hypothesised x,y,z for the emitting bats
        x_hyp = pm.Uniform('x_hyp', lower=xyz_ranges[:,0],
                              upper=xyz_ranges[:,1], shape=(nbats,))
        y_hyp = pm.Uniform('y_hyp', lower=xyz_ranges[:,2],
                              upper=xyz_ranges[:,3],shape=(nbats,))
        z_hyp = pm.Uniform('z_hyp', lower=xyz_ranges[:,4],
                              upper=xyz_ranges[:,5],shape=(nbats,))
        xyz_stack = pm.math.stack([x_hyp, y_hyp, z_hyp], axis=1)
        print('xyz_stack:',xyz_stack.shape)
        # #scenario_data[1] = xyz_stack
        # # recreate the simulated audio and cross-correlations
        sim = pm.Simulator('sim', generate_fake_cc, params=[current_scenario, xyz_stack],
                                                    observed=np.array([3,3]).reshape(-1,2),
                                                    )
    with mo:
        idata = pm.sample_smc(draws=100)
        idata.extend(pm.sample_posterior_predictive(idata))
    #az.plot_trace(idata, kind="rank_vlines");
    
    # #%%
    # # What if I deal with the bat choosing problem differently?
    # def make_audio_from_distmat(distmat,vsound=343.0):
    #     '''
    #     in distmat - each row refers to one source
        
    #     TODO : implement distance based spherical spreading of levels
        
    #     '''
    #     fs = 192000
    #     durn = 5e-3 # seconds
    #     #emitted_sound = np.random.normal(0,1e-5,5*192)
    #     emitted_sound = signal.chirp(np.linspace(0,durn,int(fs*durn)),10000,
    #                                              durn, 90000,'linear')
    #     emitted_sound *= signal.windows.hann(emitted_sound.size)
    #     toas = distmat/vsound
    #     audio_samples = int(fs*toas.max())  + emitted_sound.size
    #     audio_data = np.zeros((audio_samples, micxyz.shape[0]))
    #     for row in range(toas.shape[0]):
    #         for ch_num, channel_toa in enumerate(toas[row,:]):
    #             toa_sample = int(fs*channel_toa)
    #             toa_end = toa_sample + emitted_sound.size
    #             audio_data[toa_sample:toa_end, ch_num] += emitted_sound
        
    #     #normalise audio data by channel
    #     audio_data = np.apply_along_axis(lambda X: X/np.max(np.abs(X)), 0, audio_data,)
        
    #     return audio_data
    
    # def make_ccset_with_refch0(audio):
    #     '''
    #     audio : Msamples x Nchannels
    #     output:
    #         all_ccs : 2*M-1 samples x (audiochannels-1) cross-cors
        
    #     '''
    #     nchannels = audio.shape[1]
    #     relevant_pairs = [(i,0) for i in range(1,nchannels)]
    #     all_ccs = np.zeros((2*audio.shape[0]-1, nchannels-1))
    #     for i,(chb,cha) in enumerate(relevant_pairs):
    #         all_ccs[:,i] = signal.correlate(audio[:,chb], audio[:,cha], 'full')
    #     return all_ccs

    # def generate_fake_cc_sq(rng, emittingbats,xyz_emissions, mic_xyz, obs_cc_shape=None, size=None):
    #     '''
    #     Outputs the squared and flattened cross-correlations across 
    #     all channels with ref to channel 0
    #     '''
    #     num_ccs = mic_xyz.shape[0] - 1
        
        
    #     whichbats = np.bool_(emittingbats)
    #     if np.sum(whichbats)==0:
    #         return np.zeros(obs_cc_shape)
    #     else:            
    #         distmat = distance_matrix(xyz_emissions, mic_xyz)
    #         relevant_distmat = distmat[whichbats,:]
    #         simaudio = make_audio_from_distmat(relevant_distmat)
    #         sim_cc = make_ccset_with_refch0(simaudio)
    #         sim_cc = sim_cc**2
    #     if obs_cc_shape is not None:
    #         obs_cc_origshape = (int(np.max(obs_cc_shape)/num_ccs), num_ccs)
    #         size_diff = sim_cc.shape[0] - obs_cc_origshape[0]
    #         halfsizediff = int(size_diff*0.5)
    #         # generated crosscor is bigger than observed
    #         if size_diff > 0:
    #             sim_cc = sim_cc[halfsizediff:-halfsizediff,:]
    #         elif size_diff <0:
    #         # generated crosscor is smaller than observed
    #             zero_pad = np.zeros((abs(halfsizediff), sim_cc.shape[1]))
    #             sim_cc = np.row_stack([zero_pad, sim_cc, zero_pad])
    #         elif size_diff ==0:
    #             pass
    #         else:
    #             raise ValueError(f'Unknown sizediff: {size_diff}')
    #         return sim_cc.flatten().reshape(-1,1)
    #     else:
    #         return sim_cc.flatten().reshape(-1,1)

    # #%%
    
    # nbats = 3
    # xyz_ranges = np.ones((nbats,6))
    # for batnum in range(nbats):
    #     xyz_ranges[batnum,:] = batnum + np.linspace(0,1,6)
    # np.random.seed(82319)
    # def gen_batpositions(xyzranges):
    #     batpositions = []
    #     for each in range(nbats):
    #         xyz = []
    #         for j in range(nbats):
    #             current_cood = np.random.choice(np.linspace(xyzranges[each,j*2],xyzranges[each,2*j+1],10), 1)
    #             xyz.append(current_cood)
    #         batpositions.append(np.array(xyz).flatten())
    #     batpositions = np.array(batpositions)
    #     return batpositions
    # bat_positions = gen_batpositions(xyz_ranges)
        
    
    # #%%
    # import scipy.stats as stats
    # wasserstein = stats.wasserstein_distance
    # calling_bats = [0,1,1]
    # fake_obs_data = generate_fake_cc_sq(10, calling_bats, bat_positions, micxyz, None)
    # batpositionsclose = bat_positions + np.random.normal(0,1e-3,bat_positions.size).reshape(bat_positions.shape)
    # fake_obs_dataclose = generate_fake_cc_sq(10, calling_bats, batpositionsclose, micxyz, fake_obs_data.shape)
    
    # all_dists = []
    # for i in range(1000):
    #     callingbats = np.random.choice([0,1], 3)
    #     fd = generate_fake_cc_sq(i, callingbats, gen_batpositions(xyz_ranges), micxyz, fake_obs_data.shape)
    #     #print(i, callingbats)
    #     wassserstein_dist = wasserstein(fd.flatten(), fake_obs_data.flatten())
    #     all_dists.append(wassserstein_dist)
    
    #%%
    # import time 
    # import datetime as dt
    # call_prob = 0.6
    # with pm.Model() as mo2:
    #     emittingbats = pm.math.stack([pm.Binomial(f'bat{i}',1, p=call_prob) for i in range(nbats)])
    #     x_hyp = pm.Uniform('x_hyp', lower=xyz_ranges[:,0],
    #                           upper=xyz_ranges[:,1], shape=(nbats,))
    #     y_hyp = pm.Uniform('y_hyp', lower=xyz_ranges[:,2],
    #                           upper=xyz_ranges[:,3],shape=(nbats,))
    #     z_hyp = pm.Uniform('z_hyp', lower=xyz_ranges[:,4],
    #                           upper=xyz_ranges[:,5],shape=(nbats,))
    #     xyz_stack = pm.math.stack([x_hyp, y_hyp, z_hyp], axis=1)
    #     sim2 = pm.Simulator('sim2', generate_fake_cc_sq, params=[emittingbats, xyz_stack, micxyz, fake_obs_data.shape],
    #                                                 observed=fake_obs_data,
    #                                                 distance="laplace",sum_stat="sort",
    #                                                 epsilon = 30
    #                                                 )
    # print(f'STarting now: {dt.datetime.now()}')
    # with mo2:
    #     idata = pm.sample_smc()
    #     #idata.extend(pm.sample_posterior_predictive(idata))
    # import arviz as az
    # az.plot_density(idata)
    # stoptime = dt.datetime.now()
    # plt.savefig(f'{nbats}_estimation_{stoptime.strftime("%Y-%m-%d_%H-%M-%S")}.png')
    # print(f'Stopping now: {dt.datetime.now()}')
    
    # #%%
    # def generate_fake_data(rng, emittingbats, xyz_stack, micxyz, fake_obs_datashape, size=None):
    #     return np.random.normal(0,1,fake_obs_datashape[0]).reshape(-1,1)
    # fakemo3 = np.random.normal(0,1,100).reshape(-1,1)

    # with pm.Model() as mo3:
    #     scenarionumber = pm.Categorical('emittingbats', p=p_categories)
    #     emit_scenarios = pm.Deterministic('emit_scenario', allscenario_pt[scenarionumber,:])
        
    #     x_hyp = pm.Uniform('x_hyp', lower=xyz_ranges[:,0],
    #                           upper=xyz_ranges[:,1], shape=(nbats,))
    #     y_hyp = pm.Uniform('y_hyp', lower=xyz_ranges[:,2],
    #                           upper=xyz_ranges[:,3],shape=(nbats,))
    #     z_hyp = pm.Uniform('z_hyp', lower=xyz_ranges[:,4],
    #                           upper=xyz_ranges[:,5],shape=(nbats,))
    #     xyz_stack = pm.math.stack([x_hyp, y_hyp, z_hyp], axis=1)
    #     sim2 = pm.Simulator('sim2', generate_fake_data, params=[emit_scenarios, xyz_stack, micxyz, fakemo3.shape],
    #                                                 observed=fakemo3,
    #                                                 distance="laplace",sum_stat="sort",
    #                                                 epsilon = 30
    #                                                 )
    
    # with mo3:
    #     idata3 = pm.sample_smc(draws=100, chains=1)
    #     #idata.extend(pm.sample_posterior_predictive(idata))
    
    # az.plot_density(idata3)
    # stoptime = dt.datetime.now()
    # #plt.savefig(f'{nbats}_estimation_{stoptime.strftime("%Y-%m-%d_%H-%M-%S")}.png')
    # print(f'Stopping now: {dt.datetime.now()}')
    
    
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
import matplotlib.pyplot as plt 

#%% Create mock bat scenarios
# Genreate fake cross-correlation scenarios 
def generate_scenario_number(nbats):
    '''
    Includes a 
    '''
    
    all_scenarios = {}
    scenario_number = 0
    for subset in range(1, nbats+1):
        subscenario_combis = combinations(range(1,nbats+1),subset)
        
        for each in subscenario_combis:
            binary_scenario = np.zeros(nbats)
            batinds = np.array((each))-1
            binary_scenario[batinds] = 1 
            all_scenarios[scenario_number] = np.bool_(binary_scenario )
            scenario_number += 1 
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
    batscenario = scenariodict[scenarionumber]
    emitted_sound, fs = scenariodata[-2]
    
    
    
   
    pass




if __name__ == "__main__":
    import os
    os.environ['PYTHONIOENCODING']='utf-8'

    

    obs = np.random.normal(0,1,1000)
    import arviz as az
    import pymc as pm
    
    def dummy_func(rng, indata, size=None):
        mu = indata[-1]
        sd = indata[-2]
        return np.normal(mu,sd, 1000)
    
    with pm.Model() as mm:
        muest = pm.Uniform('muest',-1,1)
        sdest = pm.Uniform('sdest',0.8,1.5)
        inputvariable = [1,2,sdest,muest]
        ss = pm.Simulator('ss', dummy_func, inputvariable,
                          sum_stat="sort", distance='laplace',
                          observed=obs, epsilon=50)
        idata = pm.sample_smc(draws=1000)
        idata.extend(pm.sample_posterior_predictive(idata))
    az.plot_trace(idata, kind="rank_vlines");
    az.plot_posterior(idata);
    
    #%%
    nbats = 3
    scenarios = generate_scenario_number(nbats)
    num_scenarios = len(scenarios.keys())
    p_categories = np.tile(1/num_scenarios, num_scenarios)
    p_categories = p_categories/p_categories.sum()
    scenario_data = {}
    xyz_ranges = np.ones((nbats,6))
    for i in range(6):
        xyz_ranges[:,i] *= i+1
    
    allscenarios_asnp = np.array([callingbats for i, callingbats in scenarios.items()])
    micxyz = np.random.normal(0,1,30).reshape(-1,3)

    
    def generate_fake_cc(rng, scenario_number, xyz_emissions , size=None):
        
        calling_bats = allscenarios_asnp[scenario_number,:]
        #print('HELLO!',scenario_number, calling_bats)
        emission_hypotheses = xyz_emissions[calling_bats.flatten(),:]
       
        return np.array(emission_hypotheses.shape).reshape(-1,2)
        #return np.array([5, 2]).reshape(-1,2)

    #%%
    scenario_data = []
    scenario_data.append([]) # 0 current scenario
    scenario_data.append([]) # 1 all emission hypotheses
    
    
    
    with pm.Model() as mo:
        # choose which scenario is being tested
        current_scenario = pm.Categorical('current_scenario',
                                              p_categories,
                                              shape=1)
        #scenario_data[0] = current_scenario
        # generate the hypothesised x,y,z for the emitting bats
        x_hyp = pm.Uniform('x_hyp', lower=xyz_ranges[:,0],
                              upper=xyz_ranges[:,1])
        y_hyp = pm.Uniform('y_hyp', lower=xyz_ranges[:,2],
                              upper=xyz_ranges[:,3])
        z_hyp = pm.Uniform('z_hyp', lower=xyz_ranges[:,4],
                              upper=xyz_ranges[:,5])
        xyz_stack = pm.math.stack([x_hyp, y_hyp, z_hyp], axis=1)
        # #scenario_data[1] = xyz_stack
        # # recreate the simulated audio and cross-correlations
        sim = pm.Simulator('sim', generate_fake_cc, params=(current_scenario, xyz_stack),
                                                   observed=np.array([4,3]).reshape(-1,2),
                                                   )
        idata = pm.sample_smc()
        idata.extend(pm.sample_posterior_predictive(idata))
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:22:40 2023

@author: theja
"""
from itertools import combinations
import numpy as np 
import pymc as pm 
import os 
os.environ['PYTHONIOENCODING']='utf-8'    

def generate_all_possible_scenarios(nbats):
    '''
    Generates all possible 2^nbats - 1  possible combinations where
    at least 1 bat is calling
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

nbats = 3
# generate various possible scenarios individual called (1=call, 0=silent)
allscenarios_asnp = generate_all_possible_scenarios(nbats)
# set probability of all scenarios to be the same 
num_scenarios = allscenarios_asnp.shape[0]
p_categories = np.tile(1/num_scenarios, num_scenarios)
p_categories = p_categories/p_categories.sum() # 're' normalise everything to be sure it adds to 1


# x,y,z ranges of possible emission points for each individual per row
# ordered as xmin, xmax, ymin, ymax, zmin, zmax
xyz_ranges = np.ones((nbats,6))
for batnum in range(nbats):
    xyz_ranges[batnum,:] = batnum + np.linspace(0,1,6)

def do_sound_propagation(call_positions):
    ''' dummy function to mimic sound propagation. 
    Parameters
    ----------
    call_positions : (Bbats, 3) np.array
        Where Bbats >= 1.
    Returns
    -------
    synthetic_audio : (Msamples,Nchannels) np.array
    '''
    # sound propagated
    synthetic_audio =  np.random.normal(0,1,100).reshape(20,5)
    return synthetic_audio

def generate_fake_audio(rng, scenario_number, xyz_emissions, size=None):
    '''
    generates audio based on potential call positions
    
    Parameters
    ----------
    rng :  np.random.Generator
    scenario_number : int
    xyz_emissions : (Bbats,3)
        Bbats is the max number of bats
    size : int
    
    Returns
    -------
    emitted_sounds : (Msamples, Nchannels) np.array
    '''
    # which bats are calling 
    current_scenario = allscenarios_asnp[scenario_number,:]
    # select xyz positions of calling bats only
    actual_callpositions = xyz_emissions[np.bool_(current_scenario),:]
    # do audio propagation stuff  here
    emitted_sounds = do_sound_propagation(actual_callpositions)
    return emitted_sounds


if __name__ == "__main__":
    observed_audio =  np.random.normal(0,1,100).reshape(20,5)
    with pm.Model() as mo:
        # choose which scenario is being tested
        current_scenario = pm.Categorical('current_scenario',
                                              p_categories,
                                              shape=(1,))
        
        # generate the hypothesised x,y,z for the emitting bats
        x_hyp = pm.Uniform('x_hyp', lower=xyz_ranges[:,0],
                              upper=xyz_ranges[:,1], shape=(nbats,))
        y_hyp = pm.Uniform('y_hyp', lower=xyz_ranges[:,2],
                              upper=xyz_ranges[:,3],shape=(nbats,))
        z_hyp = pm.Uniform('z_hyp', lower=xyz_ranges[:,4],
                              upper=xyz_ranges[:,5],shape=(nbats,))
        xyz_stack = pm.math.stack([x_hyp, y_hyp, z_hyp], axis=1)
        
        sim = pm.Simulator('sim', generate_fake_audio, params=(current_scenario, xyz_stack),
                                                   observed=observed_audio,
                                                   )
    with mo:
        idata = pm.sample_smc(draws=100, chains=2)
        idata.extend(pm.sample_posterior_predictive(idata))
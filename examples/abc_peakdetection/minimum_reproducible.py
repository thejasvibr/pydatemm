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

nbats = 5
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


def generate_fake_cc(rng, scenario_number, xyz_emissions, size=None):
    '''
    Parameters
    ----------
    rng : 
    scenario_number : int? or pytensor?
        The row number to choose from the binary scenario matrix
    xyz_emissions : np.array? or pytensor?
        The set of drawn potential emission points
    size : 
        
    '''
    
    print(type(xyz_emissions), xyz_emissions.shape,
          type(scenario_number), scenario_number)
    
    # PROBLEM POINT when uncommented
    
    # calling_bats = allscenarios_asnp[scenario_number,:]
    
    

    
    # dummy output just to get the function to run
    return np.array([5, 2]).reshape(-1,2)
if __name__ == "__main__":
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
        print('xyz_stack:',xyz_stack.shape)
        # #scenario_data[1] = xyz_stack
        # # recreate the simulated audio and cross-correlations
        sim = pm.Simulator('sim', generate_fake_cc, params=(current_scenario, xyz_stack),
                                                   observed=np.array([3,3]).reshape(-1,2),
                                                   )
    with mo:
        idata = pm.sample_smc(draws=100)
        idata.extend(pm.sample_posterior_predictive(idata))
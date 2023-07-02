# -*- coding: utf-8 -*-
"""
Prior-based TDE peak detection 
------------------------------

Created on Fri Jun 30 18:56:07 2023

@author: theja
"""
import arviz as az
import subprocess
import numpy as np
try:
    import pymc as pm 
except:
    pass
import pandas as pd
import math
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.signal as signal 
import os 
roomdim = '4,9,3'

os.chdir('..//')
inputfolder = os.path.join('abc_peakdetection','nbat8')
try:
    os.makedirs(inputfolder)
except:
    pass

print(f'CURRENT CWD: {os.getcwd()}')
# If need be run the simulated audio once again 
subprocess.run(f"python  multibatsimulation.py -nbats 8 -ncalls 5 -all-calls-before 0.1 -room-dim {roomdim} -seed 82319 -input-folder {inputfolder} -ray-tracing False -ref-order 0",
                shell=True)

subprocess.run(f"python  multibatsimulation.py -nbats 8 -ncalls 5 -all-calls-before 0.1 -room-dim {roomdim} -seed 82319 -input-folder {inputfolder} -ray-tracing True -ref-order 1",
                shell=True)

os.chdir('abc_peakdetection//')

try:
    bat_xyz = pd.read_csv('nbat8/multibatsim_xyz_calling.csv')
    # upsample the data by interpolation 
except:
    raise ValueError('No trajectory data found!')




#%%
nbats = 5
# all possible scenarios 
batcall_scenarios = {}
scenario_num = 0
for k in range(1, nbats+1):
    scenarios = list(combinations(range(1,nbats+1), k))
    for each in scenarios:
        batcall_scenarios[scenario_num] = each
        scenario_num += 1
#%%


def gen_mock_observed_data(seednum=82319):
    fs = 192000
    np.random.seed(seednum)
    mock_audio = np.random.normal(0,1e-6,25000*4).reshape(25000,4)
    sound = np.random.normal(0,1e-4,192)
    for i in range(4):
        start = int(np.random.choice(np.arange(100,12500),1)[0])
        mock_audio[start:start+192,i] +=  sound
    
    ch_combis = list(combinations(range(4),2))
    num_combis = len(ch_combis)
    all_ccs = np.zeros((num_combis, int(mock_audio.shape[0]*2)-1))
    for i, chpair in enumerate(ch_combis):
        all_ccs[i,:] = signal.correlate(mock_audio[:,chpair[0]], mock_audio[:,chpair[1]])
    return np.sqrt(all_ccs.flatten()**2)

def generate_parambased_cc(rng, categorytype, size=None):
    if categorytype<= 20:
        return gen_mock_observed_data(82319)
    else:
        return gen_mock_observed_data(categorytype)

observed_cc = gen_mock_observed_data(82319)


def run_model():
    with pm.Model() as mm:
        category = pm.Categorical(name='category',
                                      p=np.tile(1/(scenario_num+1), scenario_num+1))
        simsim = pm.Simulator('simsim', generate_parambased_cc, category,
                              epsilon=5e-10, distance='laplace',
                              sum_stat='sort', observed=observed_cc)
        idata = pm.sample_smc()
        idata.extend(pm.sample_posterior_predictive(idata))
    az.plot_trace(idata, kind="rank_vlines");
    plt.savefig('miaowmiaow.png')
    plt.show()
    return mm, idata
#%%
if __name__ == "__main__":
    run_model()



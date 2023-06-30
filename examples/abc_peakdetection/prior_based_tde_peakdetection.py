# -*- coding: utf-8 -*-
"""
Prior-based TDE peak detection 
------------------------------

Created on Fri Jun 30 18:56:07 2023

@author: theja
"""
import subprocess
import numpy as np
try:
    import pymc as pm 
except:
    pass
import pandas as pd
import math
from itertools import combinations

# If need be run the simulated audio once again 

# subprocess.run(f"python  multibatsimulation.py -nbats 8 -ncalls 5 -all-calls-before 0.1 -room-dim {roomdim} -seed 82319 -input-folder {inputfolder} -ray-tracing False -ref-order 0",
#                shell=True)

# subprocess.run(f"python  multibatsimulation.py -nbats 8 -ncalls 5 -all-calls-before 0.1 -room-dim {roomdim} -seed 82319 -input-folder {inputfolder} -ray-tracing True -ref-order 1",
#                shell=True)

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

# def generate_sim_cc(rng, xyz_bounds, micxyz, size=None):
#     '''
#     Parameters
#     ----------
    
#     Returns 
#     -------
    
#     '''
#     exp_tdes = generate_tdes()
#     sim_audio = implement_tdes(exp_tdes)
#     sim_cc = do_multich_cc(sim_audio)
#     return sim_cc


#%%
if __name__ == "__main__"
with pm.Model() as mm:
    category = pm.Categorical(name='category',
                                 p=np.array([0.25,0.75]))
    trace = pm.sample(20)
print(trace['category'])






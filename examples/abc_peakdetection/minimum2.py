# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:43:24 2023

@author: theja
"""
import arviz as az
import numpy as np 
import pymc as pm 
import pytensor.tensor as at
import scipy.spatial as spl
import scipy.signal as signal
import random
import os 
os.environ['PYTHONIOENCODING']='utf-8'    
np.random.seed(78464)
nbats = 3
fs = 192000
calldurn = 5e-3
# all possible scenarios whare >=1 individual called (1=call, 0=silent)
allscenarios_asnp = np.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.],
                               [1., 1., 0.],
                               [1., 0., 1.],
                               [0., 1., 1.],
                               [1., 1., 1.]], dtype=np.int64)

allscenarios_boolnp = np.bool_(allscenarios_asnp)

# set probability of all scenarios to be the same 
num_scenarios = allscenarios_asnp.shape[0]
p_categories = np.tile(1/num_scenarios, num_scenarios)
p_categories = p_categories/p_categories.sum() # 're' normalise everything to be sure it adds to 1

micxyz = np.random.choice(np.arange(-10,10,0.5), 24).reshape(-1,3)
mic2mic_tof = spl.distance_matrix(micxyz, micxyz).max()/343.0 
max_delay = np.ceil(mic2mic_tof*1e3)*1e-3  + calldurn # round up the max delay in s



# x,y,z ranges of possible emission points for each individual per row
# ordered as xmin, xmax, ymin, ymax, zmin, zmax
xyz_ranges = np.ones((nbats,6))
for batnum in range(nbats):
    xyz_ranges[batnum,:] = batnum + np.linspace(0,1,6)

t = np.linspace(0,calldurn,int(fs*calldurn))
synth_call = signal.chirp(t,90000,t[-1],30000,'linear')
synth_call *= signal.windows.tukey(synth_call.size, alpha=0.9)
#%%

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
    synth_audio = np.zeros((int(fs*max_delay), micxyz.shape[0]))
    # time of flights 
    
    if call_positions.ndim==1:
        call_positions = call_positions.reshape(1,-1)
    ncalls = call_positions.shape[0]
    tofs = []
    distances = []
    for i in range(ncalls):
        distmat = spl.distance_matrix(call_positions[i,:].reshape(-1,3), micxyz)
        distances.append(distmat)
        tofmat = (distmat/343.0).flatten() # in seconds
        tofs.append(tofmat)
   
    # Now append the calls at the times of flight measured.
    # If any TOAs are -ve, then add the abs(min(TOAs)) to all the TOAS
    # to get an overal +ve TOA while also maintaining the TDOAs
    for dist, tof_callpos in zip(distances, tofs):
        
        t_sample = np.int64(fs*tof_callpos)
        randomch = random.randint(0,synth_audio.shape[1]-1)
        rel_t_sample = t_sample - t_sample[randomch]
        rel_t_sample = adjust_toas_postitive(rel_t_sample)
        for j, (radial_dist, tof_sample) in enumerate(zip(dist.flatten(),rel_t_sample.flatten())):
            
            dB_attenuation = -20*np.log10(radial_dist) # ref level @ 1 m
            rel_level = 10**(dB_attenuation/20)
            #print(j, radial_dist, tof_sample, dB_attenuation, rel_level)
            synth_audio[tof_sample:tof_sample+synth_call.size,j] += synth_call*rel_level
    
    return synth_audio

def adjust_toas_postitive(toas):
    if not np.any(toas<0):
        return toas
    else:
        altered_toas = toas.copy()
        altered_toas += abs(toas.min())
    return altered_toas

def generate_synthetic_crosscor(rng, scenario_number, xyz_emission, allscenarios, size=None):
    synth_audiodata = generate_fake_audio(rng, scenario_number, xyz_emission, allscenarios) 
    # Now make the cross-correlation between 1:Nch and 0th channel. 
    synth_cc = make_multich_crosscor(synth_audiodata)
    return synth_cc


def generate_fake_audio(rng, scenario_number, xyz_emissions,allscenarios, size=None):
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
    currentscenario = allscenarios[scenario_number]
    inds = currentscenario.nonzero()[0]
    # select xyz positions of calling bats only
    actual_callpositions = xyz_emissions[inds,:]
    # do audio propagation stuff  here
    emitted_sounds = do_sound_propagation(actual_callpositions)
    return emitted_sounds
    #return np.random.normal(0,1,100).reshape(20,5)


def make_multich_crosscor(synthaudio, refch=0):
    nsamples = -1+synthaudio.shape[0]*2
    nchannels = synthaudio.shape[1]
    cc_array = np.zeros((nsamples, nchannels-1))
    ref_audio = synthaudio[:,refch]
    ref_audio /= np.abs(ref_audio).max()
    other_ch = list(set(range(nchannels)) - set([refch]))
    
    for j in range(0,nchannels-1):
        cc_array[:,j] = signal.correlate(synthaudio[:,j]/np.abs(synthaudio[:,j]).max(),
                                         ref_audio,'full')
    return cc_array

#%%



if __name__ == "__main__":
    
    #weird_p = np.concatenate((np.tile(1/7,7), np.zeros(3))).flatten()
    prob_scenarios = np.tile(1/7,7)
    prob_scenarios = prob_scenarios/prob_scenarios.sum()
    np.random.seed(82319)
    choose = np.random.choice
    linspace = np.linspace
    # This fails
    #observed_audio =  np.random.normal(0,1,100).reshape(20,5)
    scenario_num = 6
    actual_scenario = allscenarios_asnp[scenario_num]
    batids = actual_scenario.nonzero()[0]
    call_points = np.zeros((nbats,3))
    for bat in batids:
        bat_limits = xyz_ranges[bat,:]
        
        call_points[bat,:] = [float(choose(linspace(bat_limits[j*2],bat_limits[j*2+1],100),1)) for j in range(3)]
            
        
    
    #%%
    observed_audio = generate_fake_audio(10,
                                         scenario_num, call_points, allscenarios_asnp)
    
    
    observed_cc = generate_synthetic_crosscor(10, scenario_num, 
                                              call_points, allscenarios_asnp)
    
    with pm.Model() as mo:
        allscenarios_aspt =  pm.ConstantData('allscenarios_aspt',
                                             allscenarios_asnp)
        # choose which scenario is being tested
        current_scenario = pm.Categorical('current_scenario',
                                              prob_scenarios,
                                              )
       
        currscenario = pm.math.clip(current_scenario, 0, -2+2**nbats)

        # generate the hypothesised x,y,z for the emitting bats
        x_hyp = pm.Uniform('x_hyp', lower=xyz_ranges[:,0],
                              upper=xyz_ranges[:,1], shape=(nbats,))
        y_hyp = pm.Uniform('y_hyp', lower=xyz_ranges[:,2],
                              upper=xyz_ranges[:,3],shape=(nbats,))
        z_hyp = pm.Uniform('z_hyp', lower=xyz_ranges[:,4],
                              upper=xyz_ranges[:,5],shape=(nbats,))
        xyz_stack =   pm.math.stack([x_hyp, y_hyp, z_hyp], axis=1)
        
        sim = pm.Simulator('sim', generate_synthetic_crosscor, params=[currscenario, 
                                                                       xyz_stack,
                                                                       allscenarios_aspt],
                                                    observed=observed_cc, epsilon=10,
                                                    sum_stat='median'
                                                    )
        idata = pm.sample_smc(draws=500, chains=4)        
    az.summary(idata, var_names=sorted(map(str, mo.free_RVs)))
        #%% What is a good distance measure ? 

        
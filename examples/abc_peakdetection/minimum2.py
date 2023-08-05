# -*- coding: utf-8 -*-
"""
Using Approximate Bayesian Computation to figure out the best fitting scenario
==============================================================================

Broad lessons
-------------
* Using the 1D 1 Wassertstein distance works well. 
* 


Created on Fri Jul  7 15:43:24 2023

@author: theja
"""
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from itertools import product 
import pymc as pm 
import pytensor.tensor as at
import scipy.spatial as spl
import soundfile as sf
import scipy.signal as signal
import random
import os 
os.environ['PYTHONIOENCODING']='utf-8'    
np.random.seed(78464)
nbats = 4
fs = 192000
calldurn = 5e-3
# all possible scenarios whare >=1 individual called (1=call, 0=silent)

def generate_all_scenarios(nbats):
    all_possible = list(product(*[[0,1] for i in range(nbats)]))
    return np.array(all_possible, dtype=np.int64)
    
    
allscenarios_asnp = generate_all_scenarios(nbats)

allscenarios_boolnp = np.bool_(allscenarios_asnp)

# set probability of all scenarios to be the same 
num_scenarios = allscenarios_asnp.shape[0]
p_categories = np.tile(1/num_scenarios, num_scenarios)
p_categories = p_categories/p_categories.sum() # 're' normalise everything to be sure it adds to 1

# micxyz = np.random.choice(np.arange(-10,10,0.5), 24).reshape(-1,3)
# mic2mic_tof = spl.distance_matrix(micxyz, micxyz).max()/343.0 
# max_delay = np.ceil(mic2mic_tof*1e3)*1e-3  + calldurn # round up the max delay in s




t = np.linspace(0,calldurn,int(fs*calldurn))
synth_call = signal.chirp(t,90000,t[-1],30000,'linear')
synth_call *= signal.windows.tukey(synth_call.size, alpha=0.9)


#%%
# Load the simulated audio with 4 bats and 
t_start = 0
t_stop = 0.04
fname = '../multibat_stresstests/nbat4/4-bats_trajectory_simulation_1-order-reflections.wav'
audiosnip, fs = sf.read(fname, start=t_start, stop=int(fs*t_stop))

max_delay = audiosnip.shape[0]/fs
farray = '../multibat_stresstests/nbat4/mic_xyz_multibatsim.csv'
micxyz_df = pd.read_csv(farray)
micxyz = micxyz_df.loc[:,'x':'z'].to_numpy()




# x,y,z ranges of possible emission points for each individual per row
# ordered as xmin, xmax, ymin, ymax, zmin, zmax
traj_data = pd.read_csv('../multibat_stresstests/nbat4/multibatsim_xyz_calling.csv')
valid_rows = np.logical_and(traj_data['t']>=t_start,traj_data['t']<=t_stop)
traj_data_within = traj_data.loc[valid_rows,:]
batids = traj_data['batnum'].unique()

trajwithin_by_batid = traj_data_within.groupby('batnum')

xyz_ranges = np.ones((nbats,6))
for i,batnum in enumerate(batids):
    subdf = trajwithin_by_batid.get_group(batnum)
    xyz_ranges[i,:] = [subdf['x'].min(), subdf['x'].max(),
                            subdf['y'].min(), subdf['y'].max(),
                            subdf['z'].min(), subdf['z'].max(),]




#%%
import pyroomacoustics as pra
import time 
room_dim = [4.0,9.0,3.0]

def get_first_True(X):
    '''
    X is a bool np.array
    '''
    if X.sum()>0:
        return np.where(X)[0][0]
    else:
        return np.nan

#e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
def find_common_above_threshold(multich_audio, audio_length=0.04, fs=192000, threshold=-40):
    envelope = np.apply_along_axis(lambda X: np.abs(signal.hilbert(X)), 1, multich_audio)
    linear_thresh = 10**(threshold/20.0)
    geq_thresh = envelope>= linear_thresh
    # find the earliest arriving sounds across all channels
    first_sounds = [get_first_True(geq_thresh[i,:]) for i in range(geq_thresh.shape[0])]
    first_toa = np.nanmin(first_sounds)
    common_start, common_stop = first_toa , first_toa+int(fs*audio_length)
    return multich_audio[:,common_start:common_stop]

def do_detailed_soundprop(sources):
    room = pra.ShoeBox(
        room_dim, fs=192000, materials=pra.Material(0.7), max_order=2,
        use_rand_ism = True, max_rand_disp = 0.05)
    room.add_microphone(micxyz.T)
    delays = choose(np.linspace(0,0.015,100), 4)
    [room.add_source(each, signal=synth_call, delay=calldelay)for each, calldelay in zip(sources, delays)]
    room.compute_rir()
    room.simulate()
    simaudio = room.mic_array.signals
    commonaudio = find_common_above_threshold(simaudio)
    return commonaudio.T


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
            try:
            #print(j, radial_dist, tof_sample, dB_attenuation, rel_level)
                synth_audio[tof_sample:tof_sample+synth_call.size,j] += synth_call*rel_level
            except:
                pass
    
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
    synth_cc = make_multich_crosscor_FLAT(synth_audiodata)
    return synth_cc


def generate_synthetic_crosscorv2(rng, scenario_number, xyz_emission, allscenarios, size=None):
    synth_audiodata = generate_fake_audio(rng, scenario_number, xyz_emission, allscenarios) 
    # Now make the cross-correlation between 1:Nch and 0th channel. 
    synth_cc = make_multich_crosscor_FLAT(synth_audiodata)
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
    if currentscenario.sum() == 0:
        return np.random.normal(0,1e-2,int(fs*max_delay)*micxyz.shape[0]).reshape(int(fs*max_delay), micxyz.shape[0])
    
    inds = currentscenario.nonzero()[0]
    # select xyz positions of calling bats only
    actual_callpositions = xyz_emissions[inds,:]
    # do audio propagation stuff  here
    emitted_sounds = do_sound_propagation(actual_callpositions)
    return emitted_sounds
    #return np.random.normal(0,1,100).reshape(20,5)


        
    def generate_fake_audio_roomsim(rng, scenario_number, xyz_emissions,allscenarios, size=None):
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
        if currentscenario.sum() == 0:
            return np.random.normal(0,1e-2,int(fs*max_delay)*micxyz.shape[0]).reshape(int(fs*max_delay), micxyz.shape[0])
        
        inds = currentscenario.nonzero()[0]
        # select xyz positions of calling bats only
        actual_callpositions = xyz_emissions[inds,:]
        # do audio propagation stuff  here
        emitted_sounds = do_sound_propagation(actual_callpositions)
        return emitted_sounds

def pick_callpoints(xyz_ranges):
    nrows = xyz_ranges.shape[0]
    all_chosen = []
    for i in range(nrows):
        xyz_chosen = []
        for minind, maxind in zip(range(0,6,2),range(1,6,2)):
            minval, maxval = xyz_ranges[i,minind], xyz_ranges[i,maxind]
            xyz_chosen.append(float(choose(np.linspace(minval, maxval,100),1)))
        all_chosen.append(xyz_chosen)
    return np.array(all_chosen)

def make_multich_crosscor_FLAT(synthaudio, refch=0):
    nsamples = -1+synthaudio.shape[0]*2
    nchannels = synthaudio.shape[1]
    cc_array = np.zeros((nsamples, nchannels-1))
    ref_audio = synthaudio[:,refch]
    ref_audio /= np.abs(ref_audio).max()
    #other_ch = list(set(range(nchannels)) - set([refch]))
    
    for j in range(0,nchannels-1):
        cc_array[:,j] = signal.correlate(synthaudio[:,j]/np.abs(synthaudio[:,j]).max(),
                                         ref_audio,'full')
        cc_array[:,j] /= np.abs(cc_array[:,j]).max()
    return cc_array.flatten('F').reshape(1,-1)


if __name__ == "__main__":
    
    #weird_p = np.concatenate((np.tile(1/7,7), np.zeros(3))).flatten()
    num_scenarios = allscenarios_asnp.shape[0]
    prob_scenarios = np.tile(1/num_scenarios, num_scenarios)
    prob_scenarios = prob_scenarios/prob_scenarios.sum()
    
    choose = np.random.choice
    linspace = np.linspace
    
    #%%
    def get_call_points(scenarionumber):
        actual_scenario = allscenarios_asnp[scenarionumber]
        batids = actual_scenario.nonzero()[0]
        call_points = np.zeros((nbats,3))
        for bat in batids:
            bat_limits = xyz_ranges[bat,:]
            
            call_points[bat,:] = [float(choose(linspace(bat_limits[j*2],bat_limits[j*2+1],100),1)) for j in range(3)]
        return call_points
    
    
    #%%
    np.random.seed(560018)
    # scenario_num = int(choose(np.arange(allscenarios_asnp.shape[0]),1))
    # print(scenario_num)
    # call_points = get_call_points(scenario_num)
    # # observed_audio = generate_fake_audio(10,
    # #                                      scenario_num, call_points, allscenarios_asnp)
    
    
    # observed_cc = generate_synthetic_crosscor(10, scenario_num, 
    #                                           call_points, allscenarios_asnp)
    
    observed_cc = make_multich_crosscor_FLAT(audiosnip, )
    simcc = []
    for i in range(allscenarios_asnp.shape[0]):
        ccsim = generate_synthetic_crosscor(10, i, 
                                                  get_call_points(i),
                                                  allscenarios_asnp)
        simcc.append(ccsim)    
    #%% Timing it all 

    #%% What if I split the x,y,z into discrete points - reduces the parameter exploration 
    # space - and makes the parameter estimation somewhat quicker perhaps?
    
    xyz_ranges_cm = np.int64(xyz_ranges*100)
    
    with pm.Model() as mo1:
        allscenarios_aspt =  pm.ConstantData('allscenarios_aspt',
                                              allscenarios_asnp)
        # choose which scenario is being tested
        current_scenario = pm.Categorical('current_scenario',
                                              prob_scenarios,
                                              )
       
        currscenario = pm.math.clip(current_scenario, 0, -1+2**nbats)

        # generate the hypothesised x,y,z for the emitting bats
        x_hyp = pm.DiscreteUniform('x_hyp', lower=xyz_ranges_cm[:,0],
                              upper=xyz_ranges_cm[:,1], shape=(nbats,))
        y_hyp = pm.DiscreteUniform('y_hyp', lower=xyz_ranges_cm[:,2],
                              upper=xyz_ranges_cm[:,3],shape=(nbats,))
        z_hyp = pm.DiscreteUniform('z_hyp', lower=xyz_ranges_cm[:,4],
                              upper=xyz_ranges_cm[:,5],shape=(nbats,))
        # bring it back to metres
        xyz_stack =   pm.math.stack([x_hyp, y_hyp, z_hyp], axis=1)*0.01
        
        sim = pm.Simulator('sim', generate_synthetic_crosscor, params=[currscenario, 
                                                                        xyz_stack,
                                                                        allscenarios_aspt],
                                                    observed=observed_cc,
                                                    distance='laplace',
                                                    sum_stat='sort',     
                                                    epsilon=2                                                                                                 
                                                    )
        idata1 = pm.sample_smc(draws=100, chains=4)
        #idata.extend(pm.sample_posterior_predictive(idata))
    print(az.summary(idata1, kind='stats', hdi_prob=0.2))
    print(az.summary(idata1, kind='diagnostics'))
    az.plot_trace(idata1, kind="rank_vlines");
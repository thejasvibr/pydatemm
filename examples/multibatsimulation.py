# -*- coding: utf-8 -*-
"""
Simulating what is recorded in the cave by the microphone array
===============================================================
Here I will simulate 1-N bats flying around the cave and emitting 
calls at =10 Hz call rate.

Simulation setup
~~~~~~~~~~~~~~~~
N bats flying in the space with a =0.1 s IPI (with variation). The bats fly
with a curving trajectory towards the microphone array. 

"""
import pandas as pd
import pyroomacoustics as pra
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np 
import os
import soundfile as sf
from scipy.spatial import distance
choose = np.random.choice
np.random.seed(78464)

# make the folder for all the input files
input_folder = 'simaudio_input/'
if not os.path.exists(input_folder):
	os.mkdir(input_folder)


nbats = 3
ncalls = 3
room_dims = [4,9,3]
#emission_times = make_emission_times(nbats, ncalls, ipi_range=np.linspace(0.01,0.03,50))
vbat = choose(np.linspace(2,4,20), nbats)

full_timespan = np.arange(0, 0.3,0.01)

def choose_emission_times(timepoints, nbats, ncalls, **kwargs):
    ipi = kwargs.get('ipi', 0.1)
    bat_emission_times = []
    for bat in range(nbats):
        match_not_found = True
        while match_not_found:
            start = np.random.choice(timepoints, 1)
            emission_times = np.array([start + i*ipi for i in range(ncalls)]).flatten()
            if np.all(emission_times<np.max(timepoints)):
                match_not_found = False
                bat_emission_times.append(emission_times)
    return bat_emission_times
                
emission_times = choose_emission_times(full_timespan, nbats, ncalls, ipi=0.05)
trajectories = []
f = 0.7
height = np.linspace(1, 2.0, 50)
bat1xyz = 4*np.sin(2*np.pi*f*full_timespan)+0.2, 6*np.cos(2*np.pi*full_timespan) + 1.5, np.tile(choose(height,1), full_timespan.size)
bat2xyz = 4*np.sin(2*np.pi*f*full_timespan)+0.2, 3*np.cos(2*np.pi*full_timespan) + 1, np.tile(choose(height,1), full_timespan.size)
bat3xyz = 4*np.sin(2*np.pi*f* full_timespan)+0.5, 3+3*np.cos(2*np.pi* full_timespan),  np.tile(choose(height,1), full_timespan.size)

#%%
plt.figure()
a0  = plt.subplot(111, projection='3d')
plt.plot(bat1xyz[0], bat1xyz[1], bat1xyz[2], )
plt.plot(bat2xyz[0], bat2xyz[1], bat2xyz[2], )
plt.plot(bat3xyz[0], bat3xyz[1], bat3xyz[2], )
a0.set_xlim(0,4); a0.set_ylim(0,9); a0.set_zlim(0,3)
a0.set_xlabel('x'); a0.set_ylabel('y'); a0.set_zlabel('z')
#%%

batxyz_dfs = [pd.DataFrame(each).T for each in [bat1xyz, bat2xyz, bat3xyz]]
for i, each in enumerate(batxyz_dfs):
    each.columns = ['x', 'y', 'z']
    each['t'] = full_timespan
    each['batnum'] = i+1
    each['emission_point'] = each['t'].isin(emission_times[i])

allbat_xyz = pd.concat(batxyz_dfs)
#%%


fs = 192000
ref_order = 1
reflection_max_order = ref_order
ray_tracing = False

rt60_tgt = 0.2  # seconds
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dims)

room = pra.ShoeBox(
    room_dims, fs=fs, materials=pra.Material(e_absorption),
    max_order=ref_order,
    ray_tracing=ray_tracing,
    air_absorption=True)

array_geom = np.array(([3, 8.9, 1.5],
                      [2.5, 8.9, 1],
                      [2, 8.9, 1.5],
                      [1.5, 8.9,1],
                      [1.0, 8.9, 1.5],
                      [0.01, 8, 2.0],
                      [0.01, 8, 1.5],
                      [0.01, 7, 2.0],
                      )
                      )
# add some noise to the array - this is so that none of the mics are 
# co-planar.
array_geom += np.random.choice(np.linspace(-0.01,0.01,20), array_geom.size).reshape(array_geom.shape)

#%%
# Go crazy and make each call emission the same type of call.
call_points = allbat_xyz[allbat_xyz['emission_point']]

# design the synthetic bat call
call_durn = float(choose(np.linspace(5e-3,7e-3,5),1))
minf, maxf = float(choose([15000,20000,22000],1)), float(choose([88000, 89000, 92000],1))
t_call = np.linspace(0,call_durn, int(fs*call_durn))
call_type = str(choose(['logarithmic','linear','hyperbolic'], 1)[0])
batcall = signal.chirp(t_call, maxf, t_call[-1], minf,call_type)
batcall *= signal.hamming(batcall.size)
batcall *= 1/nbats

for rownum, row in call_points.iterrows():
    x,y,z,t,_,_ = row
    call_position = np.array([x, y, z])
    room.add_source(position=call_position, signal=batcall, delay=t)

room.add_microphone_array(array_geom.T)
print('...computing RIR...')
room.compute_rir()
print('room simultation started...')
room.simulate()
print('room simultation ended...')
sim_audio = room.mic_array.signals.T

#%%
# Write down the data
if ray_tracing:
    final_path = os.path.join(input_folder,
								f'{nbats}-bats_trajectory_simulation_raytracing-{ref_order}.wav')
    sf.write(final_path, sim_audio, fs)
else:
    final_path = os.path.join(input_folder,
							f'{nbats}-bats_trajectory_simulation_{ref_order}-order-reflections.wav')
    sf.write(final_path, sim_audio, fs)


pd.DataFrame(array_geom, columns=['x','y','z']).to_csv(os.path.join(input_folder,'mic_xyz_multibatsim.csv'))
allbat_xyz.to_csv(os.path.join(input_folder,'multibatsim_xyz_calling.csv'))
#%%
# also create a 'noisy' mic xyz file to mimic the effect of having array geom data
# with noise in it (e.g. from camera reconstruction)
overall_euclidean_error = [5e-3, 0.01, 0.025, 0.05, 0.1] # m

def generate_noisy_micgeom(micxyz, overall_error):
    noisy_xyz = micxyz.copy()
    for i,each in enumerate(micxyz):
        not_achieved = True
        while not_achieved:
            noise = np.random.normal(0,overall_error,3)
            xyz = each.copy()
            xyz+= noise
            if distance.euclidean(xyz, each) <= overall_error:
                noisy_xyz[i,:] = xyz
                not_achieved = False

    return noisy_xyz
for error_level in overall_euclidean_error:
    noisy_micarray = generate_noisy_micgeom(array_geom, error_level)
    pd.DataFrame(noisy_micarray, columns=['x','y','z']).to_csv(os.path.join(input_folder,f'mic_xyz_multibatsim_noisy{error_level}m.csv'))
                
        
        
        
    




# -*- coding: utf-8 -*-
"""
Simulating what is recorded in the cave by the microphone array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here I will simulate 1-N bats flying around the cave and emitting 
calls at ~10 Hz call rate.

Simulation setup
================
N bats flying in the space with a ~0.1 s IPI (with variation). The bats fly
with a curving trajectory towards the microphone array. 


Created on Sun Sep 11 09:47:41 2022

@author: theja
"""
import pandas as pd
import pyroomacoustics as pra
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np 
import soundfile as sf
choose = np.random.choice
np.random.seed(78464)

def make_emission_times(nbats, ncalls, **kwargs):    
    ipi_range = kwargs.get('ipi_range', np.linspace(0.07,0.1,50))
    ipi_variation = kwargs.get('ipi_variation', np.linspace(0,0.01,50))
    possible_start_times = kwargs.get('possible_start_times', np.linspace(0,0.05,50))
    
    call_emission_times = []
    for each in range(nbats):
        start = choose(possible_start_times, 1)
        ipi_range = choose(ipi_range, ncalls-1)
        emission_times = np.concatenate(([0],ipi_range.copy()))
        emission_times[0] += start
        emission_times = np.cumsum(emission_times)
        call_emission_times.append(emission_times)
    return call_emission_times

nbats = 3
ncalls = 3
room_dims = [4,9,3]
emission_times = make_emission_times(nbats, ncalls)
vbat = choose(np.linspace(2,4,20), nbats)
trajectories = []
f = 0.1
bat1x, bat1y = 4*np.sin(2*np.pi*f*emission_times[0])+0.5, 7*np.cos(2*np.pi*emission_times[0])
bat2x, bat2y = 4*np.sin(2*np.pi*f*emission_times[0])+2, 3*np.cos(2*np.pi*emission_times[0])
bat3x, bat3y = 4*np.sin(2*np.pi*f*emission_times[0])+2, 3+3*np.cos(2*np.pi*emission_times[0])

bat_xyz = []
height = np.linspace(1, 2.0, 50)
for xy in [ [bat1x,bat1y], [bat2x,bat2y], [bat3x, bat3y]]:
    bat_height = choose(height,1 )
    bat_heights = np.tile(bat_height, ncalls) + choose([0,0.1,0.2],ncalls)
    xy_arr = np.column_stack(xy)
    xyz_arr = np.column_stack([xy_arr, bat_heights])
    bat_xyz.append(xyz_arr)

batxyz_df = pd.DataFrame(data = np.array(bat_xyz).reshape(-1,3), columns=['x','y','z'])
batxyz_df['t'] = [ every for each in emission_times for every in each]
batxyz_df['batnum'] = [ batnum for batnum in range(nbats) for ee in range(ncalls)]
#%%


fs = 192000
ref_order = 2
reflection_max_order = ref_order
ray_tracing = True

rt60_tgt = 0.2  # seconds
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dims)

room = pra.ShoeBox(
    room_dims, fs=fs, materials=pra.Material(e_absorption),
    max_order=ref_order,
    ray_tracing=ray_tracing,
    air_absorption=True)

array_geom = np.array(([3, 9, 1.5],
                      [2.5, 9, 1],
                      [2, 9, 1.5],
                      [1.5, 9,1],
                      [1.0, 9, 1.5],
                      [0, 8, 2.0],
                      [0, 8, 1.5],
                      [0, 7, 2.0],
                      )
                      )
array_geom += np.random.choice(np.linspace(-0.01,0.01,20), array_geom.size).reshape(array_geom.shape)
num_sources = int(np.random.choice(range(5,7),1)) # or overruled by the lines below.
random = False

for batnum in range(nbats):
    call_durn = float(choose(np.linspace(5e-3,7e-3,20),1))
    minf, maxf = float(choose([15000,20000,22000],1)), float(choose([88000, 89000, 92000],1))
    t_call = np.linspace(0,call_durn, int(fs*call_durn))
    call_type = str(choose(['logarithmic','linear','hyperbolic'], 1)[0])
    batcall = signal.chirp(t_call, maxf, t_call[-1], minf,call_type)
    batcall *= signal.hamming(batcall.size)
    batcall *= 1/nbats
    for each, emission_delay in zip(bat_xyz[batnum], emission_times[batnum]):
        room.add_source(position=each, signal=batcall, delay=emission_delay)

room.add_microphone_array(array_geom.T)
room.compute_rir()
print('room simultation started...')
room.simulate()
print('room simultation ended...')
sim_audio = room.mic_array.signals.T


#%% 
plt.figure()
a0 = plt.subplot(111, projection='3d')
plt.plot(array_geom[:,0], array_geom[:,1], array_geom[:,2], '*')
for each in bat_xyz:
    plt.plot(each[:,0], each[:,1], each[:,2], '*')

plt.xlim(0,room_dims[0])
plt.ylim(0,room_dims[1])

a0.set_zlim(0,room_dims[2])

#%%
if ray_tracing:
    sf.write(f'{nbats}-bats_trajectory_simulation_raytracing-{ref_order}.wav', sim_audio, fs)
else:
    sf.write(f'{nbats}-bats_trajectory_simulation_{ref_order}-order-reflections.wav', sim_audio, fs)

batxyz_df.to_csv('multibat_xyz_emissiontime.csv')

pd.DataFrame(array_geom, columns=['x','y','z']).to_csv('multibat_sim_micarray.csv')



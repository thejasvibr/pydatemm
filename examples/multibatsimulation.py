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
import soundfile as sf
choose = np.random.choice
np.random.seed(78464)

# def make_emission_times(nbats, ncalls, **kwargs):    
#     ipi_range = kwargs.get('ipi_range', np.linspace(0.07,0.1,50))
#     ipi_variation = kwargs.get('ipi_variation', np.linspace(0,0.01,50))
#     possible_start_times = kwargs.get('possible_start_times', np.linspace(0,0.05,50))
    
#     call_emission_times = []
#     for each in range(nbats):
#         start = choose(possible_start_times, 1)
#         ipi_range = choose(ipi_range, ncalls-1)
#         emission_times = np.concatenate(([0],ipi_range.copy()))
#         emission_times[0] += start
#         emission_times = np.cumsum(emission_times)
#         call_emission_times.append(emission_times)
#     return call_emission_times

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
                
emission_times = choose_emission_times(full_timespan, nbats, ncalls, ipi=0.08)
trajectories = []
f = 1
height = np.linspace(1, 2.0, 50)
bat1xyz = 4*np.sin(2*np.pi*f*full_timespan)+0.2, 7*np.cos(2*np.pi*full_timespan), np.tile(choose(height,1), full_timespan.size)
bat2xyz = 4*np.sin(2*np.pi*f*full_timespan)+0.2, 3*np.cos(2*np.pi*full_timespan), np.tile(choose(height,1), full_timespan.size)
bat3xyz = 4*np.sin(2*np.pi*f* full_timespan)+0.5, 3+3*np.cos(2*np.pi* full_timespan),  np.tile(choose(height,1), full_timespan.size)

#%%

bat_xyz = []

for xy in [ [bat1x,bat1y], [bat2x,bat2y], [bat3x, bat3y]]:
    bat_height = choose(height,1 )
    bat_heights = np.tile(bat_height, ncalls) + choose([0,0.1,0.2],ncalls)
    xy_arr = np.column_stack(xy)
    xyz_arr = np.column_stack([xy_arr, bat_heights])
    bat_xyz.append(xyz_arr)

batxyz_df = pd.DataFrame(data = np.array(bat_xyz).reshape(-1,3), columns=['x','y','z'])
batxyz_df['t'] = [ every for each in emission_times for every in each]
batxyz_df['batnum'] = [ batnum for batnum in range(nbats) for ee in range(ncalls)]


raise NotImplementedError('UNDER CONSTRUCTION - NEED A CLEAN IMPLEMENTATION TO CHOOSE EMISSION POINTS AND TIMES')

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
array_geom += np.random.choice(np.linspace(-0.01,0.01,20), array_geom.size).reshape(array_geom.shape)
num_sources = int(np.random.choice(range(5,7),1)) # or overruled by the lines below.
random = False

#%%

all_calldurns = []
for batnum in range(nbats):
    call_durn = float(choose(np.linspace(5e-3,7e-3,5),1))
    minf, maxf = float(choose([15000,20000,22000],1)), float(choose([88000, 89000, 92000],1))
    t_call = np.linspace(0,call_durn, int(fs*call_durn))
    call_type = str(choose(['logarithmic','linear','hyperbolic'], 1)[0])
    batcall = signal.chirp(t_call, maxf, t_call[-1], minf,call_type)
    batcall *= signal.hamming(batcall.size)
    batcall *= 1/nbats
    for each, emission_delay in zip(bat_xyz[batnum], emission_times[batnum]):
        room.add_source(position=each, signal=batcall, delay=emission_delay)
        all_calldurns.append(call_durn)

room.add_microphone_array(array_geom.T)
room.compute_rir()
print('room simultation started...')
room.simulate()
print('room simultation ended...')
sim_audio = room.mic_array.signals.T




#%%
if ray_tracing:
    sf.write(f'{nbats}-bats_trajectory_simulation_raytracing-{ref_order}.wav', sim_audio, fs)
else:
    sf.write(f'{nbats}-bats_trajectory_simulation_{ref_order}-order-reflections.wav', sim_audio, fs)

batxyz_df['call_durn'] = all_calldurns
batxyz_df.to_csv('multibat_xyz_emissiontime.csv')

pd.DataFrame(array_geom, columns=['x','y','z']).to_csv('multibat_sim_micarray.csv')

if __name__ == "__main__":
   
    plt.figure()
    plt.plot(bat1x, bat1y, 'r*')
    plt.plot(bat2x, bat2y, 'g*')
    plt.plot(bat3x, bat3y, 'k*')
    plt.plot(array_geom[:,0],array_geom[:,1],'^')
    plt.ylim(0,10)
    plt.xlim(0,6)

    pass
    #%% 
    # plt.figure()
    # a0 = plt.subplot(111, projection='3d')
    # plt.plot(array_geom[:,0], array_geom[:,1], array_geom[:,2], '*')
    # for each in bat_xyz:
    #     plt.plot(each[:,0], each[:,1], each[:,2], '*')

    # plt.xlim(0,room_dims[0])
    # plt.ylim(0,room_dims[1])

    # a0.set_zlim(0,room_dims[2])

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
import argparse
import pandas as pd
import pyroomacoustics as pra
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np 
import os
import soundfile as sf
from scipy.spatial import distance
import scipy.interpolate as si
choose = np.random.choice

def parse_room_dims(roomdims):
    split = [float(each) for each in roomdims.split(',')]
    return split 
def parse_raytracing_bool(bool_in):
    if bool_in == 'True':
        return True
    elif bool_in == 'False':
        return False
    else:
        raise ValueError(f'Cant parse {bool_in} for ray-tracing. Must be True or False')

args = argparse.ArgumentParser()
args.add_argument('-nbats', type=int)
args.add_argument('-ncalls', type=int)
args.add_argument('-all-calls-before', type=float, default=0.1)
args.add_argument('-ipi', type=float, default=0.05)
args.add_argument('-room-dim', type=parse_room_dims,)
args.add_argument('-seed', type=int, default=78464)
args.add_argument('-input-folder', type=str)
args.add_argument('-ray-tracing', type=parse_raytracing_bool, default=False)
args.add_argument('-samplerate', type=int, default=192000)




param = args.parse_args()

np.random.seed(param.seed)

# make the folder for all the input files
input_folder = param.input_folder
if not os.path.exists(input_folder):
 	os.mkdir(input_folder)


nbats = param.nbats
ncalls = param.ncalls
room_dims = param.room_dim


timestep = 2e-3
potential_call_times = np.arange(0,param.all_calls_before, timestep)

def choose_emission_times(timepoints, nbats, ncalls, **kwargs):
    ipi = kwargs.get('ipi', 0.1)
    bat_emission_times = []
    for bat in range(nbats):
        #match_not_found = True
        #while match_not_found:
        start = choose(timepoints, 1)
        emission_times = np.array([start + i*ipi for i in range(ncalls)]).flatten()
        #if np.all(emission_times<np.max(timepoints)):
        #match_not_found = False
        bat_emission_times.append(emission_times)
    return bat_emission_times
                
emission_times = choose_emission_times(potential_call_times, nbats, ncalls, ipi=param.ipi)
max_emission_time = np.concatenate(emission_times).flatten().max() + 0.01

full_timespan = np.arange(0, max_emission_time,timestep)

def make_bat_trajectoryv2(time_span, room_dims):
    all_points_not_in_room = True
    while all_points_not_in_room:
        # choose 4 random points in the room
        seed_xyz = [choose(np.linspace(0.1,axlim-0.1,1000), 6) for axlim in room_dims]
        seed_xyz = np.array(seed_xyz).T
        time_inds = np.array(time_span.size*np.linspace(0,1,seed_xyz.shape[0]), dtype=np.int64)
        time_inds[-1] -= 1 
        spline = [si.interp1d(time_span[time_inds], seed_xyz[:,i], 'cubic') for i in range(3)]
        interp_spline = np.zeros((time_span.size, 3))
        for i in range(3):
            interp_spline[:,i] = spline[i](time_span)
        
        bound_satisfied = []
        for i,axlim in enumerate(room_dims):
            in_bounds = np.logical_and(np.all(interp_spline[:,i]>=0.1),
                                        np.all(interp_spline[:,i]<=axlim-0.1))
            bound_satisfied.append(in_bounds)
        if np.all(bound_satisfied):
            all_points_not_in_room = False
            
    return interp_spline

batxyzs = [make_bat_trajectoryv2(full_timespan, room_dims) for i in range(nbats)]


#%%
plt.figure()
a0  = plt.subplot(111, projection='3d')
for i,batxyz in enumerate(batxyzs):
    plt.plot(batxyz[:,0], batxyz[:,1], batxyz[:,2], '-', label=str(i))
a0.set_xlim(0,room_dims[0]); a0.set_ylim(0,room_dims[1]); a0.set_zlim(0,room_dims[2])
a0.set_xlabel('x'); a0.set_ylabel('y'); a0.set_zlabel('z')
plt.legend()
plt.savefig(os.path.join(input_folder,f'{nbats}_{ncalls}_{room_dims}-roomsize.png'))
#%%

batxyz_dfs = [pd.DataFrame(each) for each in batxyzs]
for i, each in enumerate(batxyz_dfs):
    each.columns = ['x', 'y', 'z']
    each['t'] = full_timespan
    each['batnum'] = i+1
    each['emission_point'] = each['t'].apply(lambda X: np.sum(np.abs(X-emission_times[i])<1e-5)>0)

allbat_xyz = pd.concat(batxyz_dfs)

print(allbat_xyz[allbat_xyz['emission_point']])
#%%


fs = param.samplerate
ref_order = 1
reflection_max_order = ref_order
ray_tracing = param.ray_tracing

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
array_geom += choose(np.linspace(-0.01,0.01,20), array_geom.size).reshape(array_geom.shape)

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
    noise_range = np.arange(-overall_error,overall_error+1e-3,1e-3)
    for i,each in enumerate(micxyz):
        not_achieved = True
        
        while not_achieved:
            
            #noise = np.random.normal(0,overall_error,3)
            noise = choose(noise_range, 3)
            xyz = each.copy()
            xyz+= noise
            if distance.euclidean(xyz, each) <= overall_error:
                noisy_xyz[i,:] = xyz
                not_achieved = False

    return noisy_xyz
for error_level in overall_euclidean_error:
    noisy_micarray = generate_noisy_micgeom(array_geom, error_level)
    pd.DataFrame(noisy_micarray, columns=['x','y','z']).to_csv(os.path.join(input_folder,f'mic_xyz_multibatsim_noisy{error_level}m.csv'))
                
        
        
        
    




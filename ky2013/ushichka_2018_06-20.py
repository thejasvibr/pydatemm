"""
Trying out acoustic tracking of multi-bat audio recorded in Orlova Chuka
========================================================================

Source audio file timestamp: 1529434881 (pre-synchronised, see 'audio_prep.py')
12 microphones split over a Fireface UC and Fireface UCX.

Microphones laid out in the following order

Channels 0-5: SMP9, SMP10, SMP11, SMP12, SMP1, SMP2
Channel 6: empty
Channel 7: SYNC
Channel 8-13: SMP3, SMP4, SMP5, SMP6, SMP7, SMP8
Channel 14: empty
Channel 15: SYNC

Microphone positions 
~~~~~~~~~~~~~~~~~~~~
Microphone positions given in XYZ as measured by Asparuh Kamburov on 2018-06-20
morning with a Total Station. The Total Station measurements have an accuracy
of a few mm. 

Recording
~~~~~~~~~
The current audio recording has what seems to be multiple bats in it!
"""
import soundfile as sf
from ky2013_fullsim_chain import *

array_geometry = pd.read_csv('../examples/array_geom_2018-06-19_onlymics.csv')
array_geom = array_geometry.loc[:,'x':'z'].to_numpy()
multich_audio, fs = sf.read('../examples/first_twoseconds_1529434881_2018-06-19_synced.wav')
# take out only the mic relevant channels 
array_audio = multich_audio[:,[0,1,2,3,4,5,8,9,10,11,12,13]]

#%%
nchannels = array_audio.shape[1]
kwargs = {'nchannels':nchannels,
          'fs':fs,
          'array_geom':array_geom,
          'pctile_thresh': 95,
          'use_gcc':True,
          'gcc_variant':'phat', 
          'min_peak_diff':0.35e-4, 
          'vsound' : 343.0, 
          'no_neg':False}
kwargs['max_loop_residual'] = 0.25e-4
kwargs['K'] = 5
dd = np.max(distance_matrix(array_geom, array_geom))/343  
dd_samples = int(kwargs['fs']*dd)

start_samples = np.arange(0,array_audio.shape[0], 192)
end_samples = start_samples+dd_samples
max_inds = 100
start = time.perf_counter_ns()
all_candidates = []
for (st, en) in tqdm.tqdm(zip(start_samples[:max_inds], end_samples[:max_inds])):
    audio_chunk = array_audio[st:en,:]
    candidates = generate_candidate_sources(audio_chunk, **kwargs)
    #refined = refine_candidates_to_room_dims(candidates, 0.5e-4, room_dim)
    all_candidates.append(candidates)
stop = time.perf_counter_ns()
durn_s = (stop - start)/1e9
print(f'Time for {max_inds} ms of audio analysis: {durn_s} s')
#%%
plt.figure()
a0 = plt.subplot(111, projection='3d')
for i, data in enumerate(all_candidates):
    x,y,z = [data.loc[:,ax]  for ax in ['x','y','z']]
    a0.plot(x,y,z,'*')
    a0.plot(array_geom[:,0],array_geom[:,1], array_geom[:,2],'^')
    a0.set_xlim(-7, 1)
    a0.set_ylim(-1, 4)
    a0.set_zlim(-3, 3)
    plt.savefig(f'miaow{i}.png')
    a0.clear()


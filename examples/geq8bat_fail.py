# -*- coding: utf-8 -*-
"""
Understanding why CCG fails with 8 bats
=======================================
Here I will investigate in detail why one batid 6's calls are recovered so well, 
while none of the other bats' calls are recovered so completely.


Observations
~~~~~~~~~~~~
Bat 6 calls are likely being detected so well because min-max TOA
windows do not overlap with any of the other TOAs from other calls.


Things I tried
~~~~~~~~~~~~~~

Make compact arrays - fail
--------------------------
Reduce the number of channels to reduce the overall TOA spread of the same
sound across channels. Didn't really work as the localisation failed rather 
badly!

It actually seems like more channels helps in recovering the sources better.

Change the 'MAIN NODE' while making CFLs
----------------------------------------
See page 101 of Kreissig's thesis. 
The 'root' node for making FLs is by default set to 0. Kreissig reports
that my iterating through the root node to make FLs the source detection rate
increases.
The root node may possibly lead to some triples being formed or not??

Increase K
----------
Increasing K from 3-6 didn't really ahve an effect, while keeping epsilon rather
generous (and with crude TDOA). K=8 ran out of 20 GB memory. 


Reduce \math{\epsilon}
----------------------
Kreissig (Phd Thesis) is using a range of 0.3-0.5 sampling interval! And here
I am sitting and using very generous values of 10-50 sampling interval!! 


Interpolation of cross-correlations to reduce TDOA temporal errors
------------------------------------------------------------------
Kreissig interpolates the cross-correlation to a much higher sampling rate and
and then performs peak detection on the interpolated CC. 



TODO
~~~~
* Follow up with counting which calls 


Created on Sat Jun 17 23:28:29 2023

@author: theja
"""
import matplotlib.pyplot as plt
#from pydatemm import generate_candidate_sources
#import  pydatemm.timediffestim as timediff
import numpy as np 
import os 
import pandas as pd
import soundfile as sf
import yaml 
from source_traj_aligner import calculate_toa_channels

paramfile = 'multibat_stresstests/nbat8/nbats8outdata/paramset_nbats8-raytracing823_0.yaml'
with open(paramfile, 'r') as file:
    kwargs = yaml.safe_load(file)
arraygeom = pd.read_csv(kwargs['arraygeompath']).loc[:,'x':'z'].to_numpy()
#arraygeom = arraygeom[,:]
#%% original call points and traj
origdata = pd.read_csv('multibat_stresstests/nbat8/multibatsim_xyz_calling.csv')
callpoints = origdata[origdata['emission_point']==True].sort_values('t')
callpoints = callpoints.rename(columns={'batnum':'batid'}).reset_index(drop=True)
callpoints['min_toa'] = np.nan
callpoints['max_toa'] = np.nan
for idx, rowdata in callpoints.iterrows():
    _,x,y,z,t,batid,em_point,mintoa,maxtoa = rowdata
    all_toa = calculate_toa_channels(t, callpoints, batid, arraygeom)
    callpoints.loc[idx,'min_toa'] = np.min(all_toa)
    callpoints.loc[idx,'max_toa'] = np.max(all_toa)
 

bybatid = callpoints.groupby('batid')


#%%
# Let's quantify the overlaps of the min-max TOA windows across calls
with_overlap = []
for batid, batdf in bybatid:
    batdf['overlaps'] = 0
    otherbatdata = callpoints[callpoints['batid']!=batid]
    other_intervals = [pd.Interval(each['min_toa'], each['max_toa'], closed='both')for i, each in otherbatdata.iterrows()]
    for i, rowdata in batdf.iterrows():
        this_interval = pd.Interval(rowdata['min_toa'], rowdata['max_toa'], closed='both')
        
        for interval in other_intervals:
            if this_interval.overlaps(interval):
                batdf.loc[i,'overlaps'] += 1
    with_overlap.append(batdf)
withoverlap = pd.concat(with_overlap).reset_index(drop=True)
        

#%%
num_bats = 8
fig, axs = plt.subplots(ncols=1, nrows=num_bats,
                        figsize=kwargs.get('figsize',(5,20)),
                        layout="tight", sharex=True)

for i, batid in enumerate(withoverlap.groupby('batid').groups.keys()):
    subdf = withoverlap.groupby('batid').get_group(batid)
    subdf['mean_toa'] = np.tile(np.nan,subdf.shape[0])
    plt.sca(axs[i])
    subdf['mean_toa'] = (subdf.loc[:,'min_toa']+subdf.loc[:,'max_toa'])/2
    plt.vlines(subdf['min_toa'],0,1, label='batid '+str(batid))
    plt.vlines(subdf['max_toa'],0,1)
    
    for idx,row in subdf.iterrows():
        each = row['mean_toa']
        overlaps = row['overlaps']
        if each > 0:
            y_points = np.linspace(0.1,0.9,overlaps)
            plt.plot(np.tile(each, overlaps),y_points,'r*')
        plt.text(row['min_toa']-0.01,0.5,'Tem \n'+str(row['t']), rotation=90)

    plt.legend()

print(f"\n \n \n Total overlaps: {withoverlap['overlaps'].sum()}")
#%%
plt.figure()
a0 = plt.subplot(111, projection='3d')
by_batid = callpoints.groupby('batid')
focal_batid = 3
# plot the array
plt.plot(arraygeom[:,0],arraygeom[:,1],arraygeom[:,2],'k')
x,y,z = [by_batid.get_group(focal_batid).loc[:,ax] for ax in ['x','y','z']]
plt.plot(x,y,z, '*')
plt.plot(x[x.index[0]],y[y.index[0]],z[z.index[0]],'r^')
a0.set_xlim(0,4);a0.set_ylim(0,9);a0.set_zlim(0,3)
# plot all emission points 
subdf = by_batid.get_group(focal_batid)
call_points = subdf[subdf['emission_point']]
a0.plot(call_points.loc[:,'x'],call_points.loc[:,'y'],call_points.loc[:,'z'],'o')


#%%
# sim_audio, fs = sf.read()

# num_cores = kwargs.get('num_cores', os.cpu_count())
# multich_cc = timediff.generate_multich_crosscorr(sim_audio, **kwargs )
# kwargs['nchannels'] = sim_audio.shape[1]
# cc_peaks = timediff.get_multich_tdoas(multich_cc, **kwargs)
# K = kwargs.get('K',5) # number of peaks per channel CC to consider
# top_K_tdes = {}
# for ch_pair, tdes in cc_peaks.items():
#     descending_quality = sorted(tdes, key=lambda X: X[-1], reverse=True)
#     top_K_tdes[ch_pair] = []
#     for i in range(K):
#         try:
#             top_K_tdes[ch_pair].append(descending_quality[i])
#         except:
#             pass

# cfls_from_tdes = gramanip.make_consistent_fls_cpp(top_K_tdes, **kwargs)

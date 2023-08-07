# -*- coding: utf-8 -*-
"""
Testing the actual effect of pre-emphasis.
==========================================
Here I'll run a command-line script with different time-stamps and see
how the error in peak detection varies in comparison to the ground truth
across all channel pairs.
Created on Sun Aug  6 13:51:08 2023

@author: theja
"""
import os
import glob
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import subprocess
import tqdm
for tsta in np.arange(0,0.34,0.01):
    tsto = tsta + 10e-3
    os.popen(f"python pre-emphasis-pt2.py -tstart {tsta} -tstop {tsto} -runname miaow")

#%%
minpeakdist = 20-6
runname = 'hihi'
for tsta in tqdm.tqdm(np.arange(0,0.34,0.01)):
    tsto = tsta + 10e-3
    # os.popen(f"python pre-emphasis-pt2.py -tstart {tsta} -tstop {tsto} -runname bow -minpeakdist {minpeakdist}")
    process = subprocess.Popen(f"python pre-emphasis-pt2.py -tstart {tsta} -tstop {tsto} -runname {runname} -minpeakdist {minpeakdist}")
    process.wait()
    print(process.returncode)

dfs = [pd.read_csv(each) for each in glob.glob(f'{runname}*.csv')]
all_data = pd.concat(dfs)

plt.figure()
plt.boxplot([all_data['rawpeaks'], all_data['preemppeaks']])
plt.title(f'{runname, minpeakdist} min-peak-diff')

#%%
minpeakdist = 50e-6
runname = 'byebye'
for tsta in tqdm.tqdm(np.arange(0,0.34,0.01)):
    tsto = tsta + 10e-3
    # os.popen(f"python pre-emphasis-pt2.py -tstart {tsta} -tstop {tsto} -runname bow -minpeakdist {minpeakdist}")
    process = subprocess.Popen(f"python pre-emphasis-pt2.py -tstart {tsta} -tstop {tsto} -runname {runname} -minpeakdist {minpeakdist}")
    process.wait()
    print(process.returncode)

dfs = [pd.read_csv(each) for each in glob.glob(f'{runname}*.csv')]
all_data = pd.concat(dfs)

plt.figure()
plt.boxplot([all_data['rawpeaks'], all_data['preemppeaks']])
plt.title(f'{runname, minpeakdist} min-peak-diff')


#%%
minpeakdist = 100e-6
runname = 'byebye'
for tsta in tqdm.tqdm(np.arange(0,0.34,0.01)):
    tsto = tsta + 10e-3
    # os.popen(f"python pre-emphasis-pt2.py -tstart {tsta} -tstop {tsto} -runname bow -minpeakdist {minpeakdist}")
    process = subprocess.Popen(f"python pre-emphasis-pt2.py -tstart {tsta} -tstop {tsto} -runname {runname} -minpeakdist {minpeakdist}")
    process.wait()
    print(process.returncode)

dfs = [pd.read_csv(each) for each in glob.glob(f'{runname}*.csv')]
all_data = pd.concat(dfs)

plt.figure()
plt.boxplot([all_data['rawpeaks'], all_data['preemppeaks']])
plt.title(f'{runname, minpeakdist} min-peak-diff')


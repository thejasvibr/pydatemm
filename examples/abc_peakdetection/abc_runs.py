# -*- coding: utf-8 -*-
"""
Peak ABC
========
Created on Mon Jun 26 23:00:29 2023

@author: theja
"""
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
import scipy.signal as signal 
import numpy as np 
import pickle 
import sys 


# sys.stdin.reconfigure(encoding='utf-8')
# sys.stdout.reconfigure(encoding='utf-8')


with open('peakdetect_abc.pkl','rb') as f:
    alldata = pickle.load(f)

exp_tdoas, multichcc, envelopes = alldata
#%%

def generate_fake_data(orig_cc, exp_tdoas, maxvals, fs, peakbreadth=np.array([5e-5])):
    '''
    Parameters
    ----------
    orig_cc : np.array
        The cross-correlation
    exp_tdoas : np.array
        Array with indices where predicted peaks are. 
    maxvals : np.array
        Max peak height
    fs : int>0
        Sampling frequency in Hz
    peakbreadth : list-like, optional 
        Defaults to 50 micro secs. This is the window size of the 
        cosine window in seconds.
    
    Returns 
    -------
    predicted_cc : np.array
        Same size as orig_cc
    '''
    predicted_cc = np.zeros(orig_cc.size)
    ntdoas = len(exp_tdoas)
    if len(peakbreadth)==1:
        peakbreadth = np.tile(peakbreadth, ntdoas)     
    #print(f'peakbreadth: {peakbreadth}')
    peakbreadth *= fs
    peakbreadth = np.int16(peakbreadth)
    #maxvals = np.tile(maxvals, 3).flatten()
    for winsize, maxval, tdoa in zip(peakbreadth, maxvals, exp_tdoas):
        start_ind = tdoa-int(winsize*0.5)
        end_ind = start_ind+winsize
        this_window = signal.windows.cosine(winsize)*maxval
        predicted_cc[start_ind:end_ind] += this_window
    return predicted_cc



peakbreadth = [5e-5]
chpair = (4,3)
global fixedvars
fixedvars =  [peakbreadth, envelopes[chpair], exp_tdoas[chpair], 192000]

def sim_peak_generator(rng, maxvals,size=None):
    peakbreadth, origcc, exp_tdoas, fs = fixedvars
    return generate_fake_data(origcc, exp_tdoas, maxvals, fs, peakbreadth)

def main():
    with pm.Model() as model_abc:
        mm = pm.Uniform("mm", lower=0,upper=1,shape=(3,))
        sim = pm.Simulator('sim',
                            sim_peak_generator,
                            mm,                         
                            observed=envelopes[chpair],
                            epsilon=1e-1)
        idata_lv = pm.sample_smc()
    az.plot_trace(idata_lv, kind="rank_vlines");
    plt.savefig('miaowmiaow.png')
    return model_abc, idata_lv
#%%
if __name__ =="__main__":
        main()

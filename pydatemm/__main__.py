#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line interface to run multiple audio segments together
==============================================================
Input is a YAML file that contains all the necessary parameters. 


TODO: 
    * implement the possibility of adding multiple timewindows to the computation.
    Right now one-timewindow per main call is a real bottleneck as the startup+compilation
    of multiple c++ files takes about 6-8 s each time.

"""
import argparse
import datetime as dt
import numpy as np 
import os
import pandas as pd
from pydatemm import generate_candidate_sources
import scipy.signal as signal 
import soundfile as sf
import traceback
import yaml 
from sklearn.cluster import DBSCAN

help_text = """Run generate_candidate_sources for a given time snippet of an audio file"
    Parameters to input in -paramfile:
        * audiopath : path to input audio file
        * arraygeompath : path to csv file with (Nmics,3) xyz data. 
            Headers must be organised as 'x','y','z'
        * start_time : string with start times of the audio in seconds. e.g. 0,0.003,0.006
        * window_size: Window size of the chosen audio in seconds 0.010. 
        * remove_lastchannel : bool. Whether the last channel is to be removed. Defaults to True
        * thresh_tdoaresidual : Threshold in seconds to filter out localisations with higher\
                                TDOA residual. 
        * dest_folder : Path to the destination folder. New folder will be made if not present.
        * K : int. NUmber of peaks to choose per cross-correlation profile. 
        * min_peak_dist : float. Minimum distance between peaks in the cross-correlation in seconds
        * maxloopres : float. Maximum loop residual tolerated. 
        * channels : str. comma-separated zero-indexed channel numbers which are to be used for this 
            particular analysis run e.g. 0,1,4,5,6 
        * vsound : float>0. Speed of sound in m/s. Defaults to 343 m/s
        * num_cores : int. Number of cores to use per run to perform the candidate source
            localisations calculations after TDOA graphs are generated. Defaults to 2.
        * run_name : str. Name of the run. Defaults to the wav-file name.
        * highpass_ : str. If provided then the audio snippet will be highpassed
            with the given Butterworth filter order and at the given cutoff frequency
            in Hz.
            e.g. highpass: 2,10000 means a 2nd order filter with cutoff at 10 kHz.

            """

parser = argparse.ArgumentParser(description=help_text,)
parser.add_argument('-paramfile', type=str, help="Path to parameter YAML file")

args = parser.parse_args()
with open(args.paramfile, 'r') as file:
    params = yaml.safe_load(file)

if not os.path.exists(params['dest_folder']):
    os.mkdir(params['dest_folder'])

now = dt.datetime.now()
#%%
# Utility functions
def conv_to_numpy(pydatemm_out):
    return np.array([np.array(each) for each in pydatemm_out]).reshape(-1,4)
# Run DBSCAN to simplify these datapoints!
def get_3d_median_location(xyz):
    return np.apply_along_axis(np.median, 0, xyz)

def main():
    #%%
    # Setting up tracking
    raw_array_geom = pd.read_csv(params['arraygeompath']).loc[:,'x':'z'].to_numpy()
    vsound = params.get('vsound', 343.0) # m/s
    
    #%%
    # 
    fs = sf.info(params['audiopath']).samplerate
    #%% parse time windows 
    timewindows = list(map(lambda X: float(X), params['start_time'].split(",")))
    timewindows = [float(each) for each in timewindows]
    window_size = float(params['window_size']) # seconds
    for start_t in timewindows:
        try:
            stop_t = start_t + window_size
            start_ind, stop_ind = int(start_t*fs), int(stop_t*fs)
            audio, fs = sf.read(params['audiopath'], start=start_ind, stop=stop_ind+1)
            if params.get('remove_lastchannel') is None:
            	mic_audio = audio[:,:-1]
            elif params.get('remove_lastchannel')=='False':
                mic_audio = audio.copy()
            
            
            kwargs = {'fs':fs,
                      'pctile_thresh': 95,
                      'use_gcc':True,
                      'gcc_variant':'phat', 
                      'min_peak_diff': float(params['min_peak_dist']), 
                      'vsound' : vsound}
            kwargs['max_loop_residual'] = float(params['maxloopres']) 
            tdoa_resid_threshold = float(params['thresh_tdoaresidual'])
            
            kwargs['K'] = params['K']
            kwargs['num_cores'] = params.get('num_cores', 2)
            #%% parse channels to use
            if params.get('channels') is not None:
            	channels = [int(each) for each in params['channels'].split(',')]
            	mic_audio = mic_audio[:,channels]
            	array_geom = raw_array_geom[channels,:]
            else:
            	array_geom = raw_array_geom.copy()
            
            kwargs['nchannels'] = mic_audio.shape[1]
            kwargs['array_geom'] = array_geom

            hp_order =  params.get('highpass_order')
            if hp_order is not None:
                order, cutoff = [float(each) for each in hp_order.split(',')]
                order = int(order)
                b,a = signal.butter(order, cutoff/(2*fs), 'highpass')
                mic_audio = np.apply_along_axis(lambda X: signal.filtfilt(b,a,X), 0, mic_audio)
            
            #%%
            # Keep things simple for now, save some time and check only specific chunks of 
            # audio that have the playbacks. 
            
            output = generate_candidate_sources(audio, **kwargs)
            wavfilename = os.path.split(params['audiopath'])[-1].split('.')[0]
            
            run_name = params.get('run_name', wavfilename)
            csv_fname = os.path.join(params['dest_folder'], f'{run_name}_{start_t}-{stop_t}.csv')
            
            if len(output.sources)>0:
                posns = conv_to_numpy(output.sources)
                # get rid of -999 entries
                no_999 = np.logical_and(posns[:,0]!=-999, posns[:,1]!=-999)
                posns_filt = posns[no_999,:]
                posns_filt = posns_filt[posns_filt[:,-1]<tdoa_resid_threshold]
            else:
                posns_filt = np.array([])
                
            if posns_filt.shape[0]>0:
                # There're a lot of 'repeat' localisations - simplify and bring down the massive 
                # localisations to the minimum unique set. 
            
                posns_filt_str = np.char.array(posns_filt)
                spacer = np.char.array(np.tile(-999,posns_filt.shape[0]))
                all_rows_combined = posns_filt_str[:,0] +spacer+ posns_filt_str[:,1] + spacer+posns_filt_str[:,2] + spacer+posns_filt_str[:,3]
            
                unique_elements, unique_inds, counts= np.unique(all_rows_combined, return_index=True, return_counts=True)
                unique_posns_filt = posns_filt[unique_inds,:]
                # perform DBSCAN only if the # of particles is very high (e.g. > 20000)
                if unique_posns_filt.shape[0]>20000:
                    
                    clustered = DBSCAN(eps=0.1).fit(unique_posns_filt)
                    # clustered = DBSCAN(eps=0.3, min_samples=20, algorithm='ball_tree').fit(posns_filt)
                    labels = clustered.labels_
                    valid_labels = np.unique(labels)
                    all_centres = []
                    for each in valid_labels :
                        rows = clustered.labels_ == each
                        centre = get_3d_median_location(unique_posns_filt[rows,:])
                        all_centres.append(np.append(centre, each))
                    df = pd.DataFrame(all_centres)
                else:
                    df = pd.DataFrame(unique_posns_filt)
                    df['label'] = np.nan
                df['t_start'] = start_t
                df['t_end'] = stop_t
                
                df.columns = ['x','y','z','tdoa_res','label','t_start','t_end']
                df['paramfile'] = args.paramfile
                df.to_csv(csv_fname)
            
                print(f'COMPLETED RUN FOR {start_t, stop_t}')
            else:
                df = pd.DataFrame(data={}, columns = ['x','y','z','tdoa_res','label','t_start','t_end'])
                df['x'] = [np.nan]
                df['paramfile'] = args.paramfile
                print(f'No sources found in {start_t, stop_t}...')
            df.to_csv(csv_fname)
        except Exception:
            print(f'CANT PROCESS AUDIO FOR {start_t, stop_t}')
            print(traceback.format_exc())
            

if __name__ == "__main__":
    main()
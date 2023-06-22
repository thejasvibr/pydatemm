# -*- coding: utf-8 -*-
"""
Code that generates proximity profiles
======================================

Created on Fri Jun  9 23:36:12 2023

@author: theja
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
import scipy.signal as signal 

def get_close_points(batid, input_df, nearish_sources, coarse_thresh):
    '''
    Get all sources that are <= an initial 'coarse threshold' distance 
    of the flight trajectories.

    Parameters
    ----------
    batid : int
    input_df : pd.DataFrame
        Trajectory information with columns t, batid, x,y,z
    nearish_sources : (Npoints,6) np.array
        Columns hold x,y,z,tdoa residual,tstart,tend
    coarse_thresh : float>0
        The coarse threshold over which the nearby sources are chosen
    
    Returns 
    -------
    close_points : (M, 6) np.array
        Sub set of all source points that are close enough to the flight
        trajectory points.

    '''
    points_to_traj = distance_matrix(nearish_sources[:,:3],
                                     input_df.loc[:,'x':'z'].to_numpy())
    close_point_inds = np.where(points_to_traj<coarse_thresh)
    close_points = nearish_sources[np.unique(close_point_inds[0]),:]
    return close_points

def get_source_proximity_counts(close_points,fine_threshold, input_df, arraygeom, vsound, topx):
    '''
    Parameters
    ----------
    close_points : (M,6) np.array
        Set of sources that are <= the coarse threshold for proximity
    input_df : pd.DataFrame
        Video trajectory data for that particular bat id 
    arraygeom : (N,3) np.array
        xyz data for the microphone array
    vsound : float>0
        The speed of sound in m/s
    
    Returns
    -------
    counts_by_batid : (timesteps,) np.array
        
    
    '''
    
    t_em = np.zeros(input_df.shape[0])
    counts_by_batid = np.zeros(t_em.size)

    i = 0
    for k,candidate in enumerate(close_points):
        xyz, timewindow = candidate[:3], candidate[-2:]
        potential_tof = distance_matrix(xyz.reshape(-1,3),
                                        arraygeom)/vsound
        # get the widest window possible for the video trajs
        wide_timewindow = [np.min(timewindow[0]-potential_tof), 
                           np.max(timewindow[1]-potential_tof)]
        
        rows_inwindow = input_df['t'].between(wide_timewindow[0],
                                           wide_timewindow[1], inclusive='both')
        subset_df = input_df.loc[rows_inwindow,:]
        
        dist_to_clust = distance_matrix(xyz.reshape(-1,3),
                                        subset_df.loc[:,'x':'z'].to_numpy()).flatten()
    
        inds_close = np.where(dist_to_clust<fine_threshold)[0]

        if len(inds_close)>0:
            relevant_inds = subset_df.index[np.argsort(dist_to_clust)][:topx]
            corrected_inds = relevant_inds - input_df.index[0]
            counts_by_batid[corrected_inds[0]] += 1
            counts_by_batid[corrected_inds] += 1                   
            i += 1 
    return counts_by_batid

def generate_proximity_profile(*kwargs):
    '''
    Wrapper around get_close_points and get_source_proximity_counts
    '''
    batid, batdf, sources_nearish, coarse_threshold, fine_threshold, array_geom, vsound, topx = kwargs
    
    closepoints = get_close_points(batid,
                                   batdf,
                                   sources_nearish,
                                   coarse_threshold)
    proximity_counts = get_source_proximity_counts(closepoints,fine_threshold,
                                                   batdf,
                                                   array_geom,
                                                   vsound, topx)
    return proximity_counts


# Diagnostic plots below
    
def make_colorset(**kwargs):
    '''
    Generates a set of colors wthin the viridis colormap
    num_bats : int
        Number of bats
    
    '''
    cmap = matplotlib.cm.get_cmap('viridis')
    fractions = np.linspace(0,1,kwargs['num_bats'])
    colors = [cmap(frac)[:-1] for frac in fractions]
    return colors

def plot_diagnostic(audio_data_pack, proximity_data_pack, arraygeom, flighttraj, **kwargs):
    '''
    Parameters
    ----------
    audio_data_pack : tuple
        A tuple with (multichannel audio np.array, sampling frequency (Hz), duration (s))
        The audio data is a (Msamples,Nchannel) np.array.
    proximity_data_pack : tuple
        Tuple with (proximity_profile, proximity_profile_peaks)
        Proximity_profile is a dict with keys are bat-ids and entries are
        np.arrays with proximity counts
        where the # of sources close to the flight trajectory are very close.
    arraygeom : (Nmics,3) np.array
        xyz data in rows for each mic
    flighttraj : pd.DataFrame
        Flight trajectory data.
    vis_channels : list, optional 
        The channels to be visualised on the spectrogram. 
        Defaults to [0,-1]
    sim_traj : pd.DataFrame, optional
        If present then the groundtruth times of emission will be plotted. 
    
    Returns 
    -------
    fig,axs : plt.figure, plt.axes
        Figure and axes of the plot
        
    '''
    n_bats = len(flighttraj.groupby('batid').groups)
    colorset = kwargs.get('colors', make_colorset(num_bats=n_bats))
                          
    audio, fs, duration = audio_data_pack
    proximity_profiles, proximity_peaks = proximity_data_pack
    num_bats = len(proximity_profiles.keys())
    vis_channels = kwargs.get('vis_channels', [0,-1])
    nchannels = len(vis_channels)
    # Make the broad plot 
    fig, axs = plt.subplots(ncols=1, nrows=num_bats+len(vis_channels),
                            figsize=kwargs.get('figsize',(5,20)),
                            layout="tight", sharex=True)
    
    # plot the proximity profiles for each bat
    
    for i,(batid, source_prof) in enumerate(proximity_profiles.items()):
        t_batid = flighttraj.groupby('batid').get_group(batid)['t'].to_numpy()
        
        plt.sca(axs[i])
        plt.plot(t_batid, source_prof, label='bat id ' + str(batid),
                                                 color=colorset[int(batid)-1])
        plt.xlim(-0.001,audio.shape[0]/fs)
        plt.xticks([])
        plt.legend()
        # on the way also plot the detected peaks too. 
        peak_height_inds = [int(np.argwhere(t_batid==each)[0]) for each in proximity_peaks[batid]]
        peak_heights = source_prof[peak_height_inds]
        plt.plot(proximity_peaks[batid], peak_heights, 'k*')
    
    # plot the expected TOA for all of the detected peaks
    for batid in proximity_peaks.keys():
        plotted_toa_from_peaks(fig, axs[-nchannels:], flighttraj, batid,
                               proximity_peaks[batid],
                                               vis_channels, arraygeom, colorset)
    # if known, plot the known TOE of all calls
    if kwargs.get('sim_traj') is not None:
        groundtruth = kwargs['sim_traj']
        actual_emission_points = groundtruth[groundtruth['emission_point']].groupby('batid')
        for batid, subdf in actual_emission_points:
            plt.sca(axs[batid-1])
            for i, row in subdf.iterrows():
                _,x,y,z,t,*uu = row
                if np.max(proximity_profiles[batid])>0:
                    plt.vlines(t, 0, np.max(proximity_profiles[batid]), color='r')
                else:
                    plt.vlines(t, 0, 1, color='r')
                    
    
    # plot the spectrogram of the visualised audio channels and the expected
    # TOAs of the sources
    for i, ch in enumerate(vis_channels):
        plt.sca(axs[num_bats+i])
        plt.specgram(audio[:,ch], Fs=fs, xextent=[0, duration], cmap='cividis')
    return fig, axs

def plotted_toa_from_peaks(fig, specgram_axes, flighttraj, batid, peak_times,
                           target_channels, arraygeom, colorset):
    for t_emission in peak_times:
        toa = calculate_toa_channels(t_emission, flighttraj, batid, 
                                     arraygeom[target_channels,:]).flatten()
        for channel_toa, axid in zip(toa, specgram_axes):
            plt.sca(axid)
            plt.vlines(channel_toa, 20000,96000, linestyle='dotted', color=colorset[batid-1])
    fig.canvas.draw()

def calculate_toa_channels(t_source, flighttraj, sourceid, arraygeom):
    '''
    Parameters
    ----------
    t_source : float
        Time point of acoustic source emission
    flighttraj: pd.DataFrame
        With at least batid, x,y,z columns
    sourceid : int
        The numeric batid 
    arraygeom : (N,3) np.array
        xyz positions of array
    '''
    flight_traj = flighttraj.groupby('batid').get_group(sourceid).reset_index(drop=True)
    # get closest point in time 
    nearest_ind = abs(flight_traj['t']-t_source).argmin()
    emission_point = flight_traj.loc[nearest_ind, 'x':'z'].to_numpy().reshape(-1,3)
    tof = distance_matrix(emission_point, arraygeom)/343.0 
    toa = t_source + tof
    return toa

# audio_channels = [0,1,]


# traj_data = upsampled_flighttraj.groupby('batid')


# def calculate_toa_channels(t_source, sourceid, arraygeom):
#     flight_traj = traj_data.get_group(sourceid).reset_index(drop=True)
#     # get closest point in time 
#     nearest_ind = abs(flight_traj['t']-t_source).argmin()
#     emission_point = flight_traj.loc[nearest_ind, 'x':'z'].to_numpy().reshape(-1,3)
#     tof = distance_matrix(emission_point, arraygeom)/343.0 
#     toa = t_source + tof
#     return toa

# proximity_peaks = {}

# for i,(batid, source_prof) in enumerate(counts_by_batid.items()):
#     t_batid = upsampled_flighttraj.groupby('batid').get_group(batid)['t'].to_numpy()
    
#     plt.sca(axs[i])
#     plt.plot(t_batid, source_prof, label='bat id ' + str(batid), color=colors[int(batid)-1])
#     plt.xticks([])
#     plt.legend()
    
    
#     plt.plot(t_batid[pks], source_prof[pks],'g*')


# for i, ch in enumerate(audio_channels):
#     plt.sca(axs[num_bats+i])
#     plt.specgram(audio[:,ch], Fs=fs, xextent=[0, sf.info(audiofile).duration], cmap='cividis')


# def plotted_toa_from_peaks(specgram_axes, batid, peak_times, target_channels, window_halfwidth):
#     for t_emission in peak_times:
#         toa = calculate_toa_channels(t_emission, batid, 
#                                      array_geom[target_channels,:]).flatten()
#         toa_2 = calculate_toa_channels(t_emission-window_halfwidth, batid, 
#                                      array_geom[target_channels,:]).flatten()
    

#         for channel_toa,channel_toa2, axid in zip(toa, toa_2, specgram_axes):
#             plt.sca(axid)
#             plt.vlines(channel_toa, 20000,96000, linestyle='dotted', color=colors[batid-1])
#             #plt.vlines(channel_toa2, 20000,96000, linestyle='dashed', color=colors[batid-1])
#     fig.canvas.draw()

# nchannels = len(audio_channels)
# for batid in list(counts_by_batid.keys()):
#     plotted_toa_from_peaks(axs[-nchannels:], batid, proximity_peaks[batid], audio_channels, 16e-3)

# plt.gca().set_xticks(np.arange(0, duration, .05))
# plt.gca().set_xticks(np.linspace(0, duration, 100), minor=True)

# from matplotlib.lines import Line2D
# actual_emission_points = flight_traj[flight_traj['emission_point']].groupby('batid')
# for batid, subdf in actual_emission_points:
#     plt.sca(axs[batid-1])
#     for i, row in subdf.iterrows():
#         _,x,y,z,t,*uu = row
#         plt.vlines(t, 0, np.max(counts_by_batid[batid]), color='r')

# # access legend objects automatically created from data
# plt.sca(axs[0])
# handles, labels = axs[0].get_legend_handles_labels()
# line = Line2D([0],[0],label='original emission times', color='r')
# handles.extend([line])
# plt.legend(handles=handles)


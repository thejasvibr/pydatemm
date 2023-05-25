#!/usr/bin/env pyth

# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:33:20 2023

@author: thejasvi
"""
import glob
import matplotlib
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import pyvista as pv
from scipy.spatial import distance_matrix, distance
import soundfile as sf
import scipy.signal as signal 
euclidean = distance.euclidean
import scipy.interpolate as si

posix = '1529546136'
output_folder = f'{posix}_output/'
arraygeom_file = f'{posix}_input/Sanken9_centred_mic_totalstationxyz.csv'
audiofile = f'{posix}_input/video_synced10channel_first15sec_{posix}.WAV'
array_geom = pd.read_csv(arraygeom_file).loc[:,'x':'z'].to_numpy()
vsound = 340.0  # m/s
#%%
# load all the results into a dictionary
result_files = natsorted(glob.glob(output_folder+'/*.csv'))
# keep only those with the relevant time-window size
def get_start_stop_times(file_name):
    times = file_name.split('_')[-1].split('.csv')[0]
    start_t, stop_t = [float(each) for each in times.split('-')]
    durn = stop_t - start_t
    return durn
def window_length_is_correct(file_name, expected, tolerance=1e-15):
    durn = get_start_stop_times(file_name)
    if abs(durn-expected)<tolerance:
        return True
    else:
        return False

    
all_results = []
for i,fname in enumerate(result_files):
    if window_length_is_correct(fname, 0.016):
        all_results.append(pd.read_csv(fname))

all_sources = pd.concat(all_results).reset_index(drop=True)
all_posns = all_sources.loc[:,['x','y','z','tdoa_res','t_start','t_end']].to_numpy()

#%%
# Now load the video flight trajectories and transform them from the 
# camera to TotalStation coordinate system
start_time = 0
end_time = 1.5
flight_traj = pd.read_csv(f'{posix}_input/xyz/P00_22000_bat_trajs_round1_sanken9_centred.csv')
transform_matrix = pd.read_csv(f'{posix}_input/xyz/Sanken9_centred-video-to-TotalStation_transform.csv')
transform = transform_matrix.to_numpy()[:,1:]
bring_to_totalstation = lambda X: np.matmul(transform, X)
flight_traj_conv = flight_traj.loc[:,'x':'z'].to_numpy()
flight_traj_conv = np.apply_along_axis(bring_to_totalstation, 1, flight_traj_conv)
flight_traj_conv = pd.DataFrame(flight_traj_conv, columns=['x','y','z'])
flight_traj_conv['batid'] = flight_traj['batid'].copy()
flight_traj_conv['frame'] = flight_traj['frame'].copy()

flight_traj_conv['t'] = flight_traj_conv['frame']/25 
# filter out the relevant flight trajectory for the window
timewindow_rows = np.logical_and(flight_traj_conv['t'] >= start_time, 
                                        flight_traj_conv['t'] <= end_time) 
flighttraj_conv_window = flight_traj_conv.loc[timewindow_rows,:].dropna()

#%% Now assemble the high-temporal resolution version of the same
t_raw = flighttraj_conv_window['t'].to_numpy()
t_highres = np.arange(t_raw.min(), t_raw.max(), 1e-3)
traj_splines = {i : si.interp1d(t_raw, flighttraj_conv_window.loc[:,i], 'quadratic') for i in ['x','y','z']}

traj_interp = { i: traj_splines[i](t_highres) for i in ['x','y','z']}
flighttraj_interp = pd.DataFrame(data=traj_interp)
flighttraj_interp['t'] = t_highres
flighttraj_interp['batid'] = 1

#%% Keep only those that are within a few meters of any bat trajectory positions
distmat = distance_matrix(flighttraj_interp.loc[:,'x':'z'].to_numpy(), all_posns[:,:3])
nearish_posns = np.where(distmat<0.75) # all points that are at most 8 metres from any mic
sources_nearish = all_posns[np.unique(nearish_posns[1]),:]

#%%
# Run a DBSCAN on the nearish sources to get call centres
from sklearn.cluster import DBSCAN
clusters = DBSCAN(eps=0.1).fit(sources_nearish[:,:3])

cluster_centres = []
for lab in np.unique(clusters.labels_):
    if not lab == -1:
        inds = np.where(clusters.labels_==lab)
        cluster_points = sources_nearish[inds, :3].reshape(-1,3)
        centroid = np.median(cluster_points,axis=0)
        #print(lab, centroid)
        cluster_centres.append(centroid)
cluster_centres = np.array(cluster_centres)

mic_video_xyz = pd.read_csv('1529543496_input/xyz_data/Sanken9_centred_mic_videoxyz.csv')
mic_video_xyz.columns = [0,'x','y','z','micname']



#%%
box = pv.Box(bounds=(0,5,0,9,0,3))
plotter = pv.Plotter()

def mouse_callback(x):
    print(f'camera position: {plotter.camera.position})')
    print(f'\n az, roll, el: {plotter.camera.azimuth, plotter.camera.roll, plotter.camera.elevation}')
    print(f'\n view angle, focal point: {plotter.camera.view_angle, plotter.camera.focal_point}')
plotter.track_click_position(mouse_callback)

#plotter.add_mesh(box, opacity=0.3)
# include the mic array
for each in array_geom:
    plotter.add_mesh(pv.Sphere(0.03, center=each), color='g')

for i in [1,2,3]:
    plotter.add_mesh(pv.lines_from_points(array_geom[[0,i],:]), line_width=5)
plotter.add_mesh(pv.lines_from_points(array_geom[4:,:]), line_width=5)

points_to_traj = distance_matrix(sources_nearish[:,:3], flighttraj_interp.loc[:,'x':'z'].to_numpy())
close_point_inds = np.where(points_to_traj<0.5)
close_points = sources_nearish[np.unique(close_point_inds[0]),:3]

plotter.add_points(close_points, render_points_as_spheres=True, point_size=10)


# One idea is to histogram all the flight traj points that have an acoustic source
# as their closest point. This way we can create a 'proximity histogram'

counts = np.zeros(flighttraj_interp.shape[0])
for i,each in enumerate(np.unique(close_point_inds[0])):
    #get min distance 
    try:
        index = np.nanargmin(points_to_traj[each,:])
        counts[index] += 1
    except:
        pass



# get diff points of diff size https://github.com/pyvista/pyvista-support/issues/171
pointmesh = pv.PolyData(flighttraj_interp.loc[:,'x':'z'].to_numpy())
# pointmesh["radius"] = np.log10(counts/sum(counts) +1)*20

# trajpoint = pv.Sphere(theta_resolution=15, phi_resolution=8)
# glyphed = pointmesh.glyph(scale="radius", geom=trajpoint, progress_bar=True)
plotter.add_mesh(pointmesh, color='white', opacity=0.25)

plotter.camera.position = (-2.29, -11.86, 1.50)
plotter.camera.azimuth = 0.0
plotter.camera.roll = 50
plotter.camera.elevation = 0.0
plotter.camera.view_angle = 30.0
plotter.camera.focal_point = (0.42, 1.59, 0.68)


cmap = matplotlib.cm.get_cmap('Spectral')
fractions = np.linspace(0,1,np.unique(flight_traj_conv['batid']).size)
colors = [cmap(frac)[:-1] for frac in fractions]

# plot the flight trajectories and call emission points
for key, subdf in flighttraj_conv_window.groupby('batid'):
    traj_line = pv.lines_from_points(subdf.loc[:,'x':'z'].to_numpy())
    plotter.add_mesh(traj_line, line_width=7, color=colors[int(key)-1])

plotter.show()

#%%
camera_positions = np.array([[-1.0, -8.18, 2.08],
                    [-5.27, -7.03, 2.24],
                    [-9.12, -3.12, 2.03],
                    ])
t = np.linspace(0,1,camera_positions.shape[0])
t_interp = np.linspace(0,1,25)
roll_values = [50, 76, 84]
roll_spline = si.interp1d(t,roll_values)
xyz_splines = [ si.interp1d(t, camera_positions[:,i], 'quadratic') for i in range(3)]
xyz_interp = []
for i in range(3):
    xyz_interp.append(xyz_splines[i](t_interp))
roll_interp = roll_spline(t_interp)
camera_traj = np.column_stack(xyz_interp)
#%%
# Now update the camera position 

box = pv.Box(bounds=(0,5,0,9,0,3))
plotter = pv.Plotter()

plotter.camera.position = tuple(camera_traj[0,:])
plotter.camera.azimuth = 0.0
plotter.camera.roll = roll_values[0]
plotter.camera.elevation = 0.0
plotter.camera.view_angle = 30.0
plotter.camera.focal_point = (-0.8600637284803319, 1.0764831143834046, 0.5677780189761863)
# include the mic array
for each in array_geom:
    plotter.add_mesh(pv.Sphere(0.03, center=each), color='g')

for i in [1,2,3]:
    plotter.add_mesh(pv.lines_from_points(array_geom[[0,i],:]), line_width=5)
plotter.add_mesh(pv.lines_from_points(array_geom[4:,:]), line_width=5)


# filter out cluster centres based on how far they are from a trajectory
cluster_centre_traj = distance_matrix(cluster_centres, flighttraj_interp.loc[:,'x':'z'].to_numpy())
valid_centres = np.where(cluster_centre_traj<0.3)

valid_clusters = cluster_centres[np.unique(valid_centres[0]),:]

#plotter.add_points(valid_clusters)
for xyz  in valid_clusters:
    plotter.add_mesh(pv.Sphere(0.1, center=xyz), color='w', opacity=0.9)

for row,_ in zip(*valid_centres):
    plotter.add_mesh(pv.Sphere(0.1, center=cluster_centres[row,:]), color='w', opacity=0.9)

cmap = matplotlib.cm.get_cmap('Spectral')
fractions = np.linspace(0,1,np.unique(flight_traj_conv['batid']).size)
colors = [cmap(frac)[:-1] for frac in fractions]

# plot the flight trajectories and call emission points
for key, subdf in flighttraj_conv_window.groupby('batid'):
    traj_line = pv.lines_from_points(subdf.loc[:,'x':'z'].to_numpy())
    plotter.add_mesh(traj_line, line_width=7, color=colors[int(key)-1])

# plotter.open_gif('1529546136-0-1.5_singlebat.gif', fps=5, )

# for roll,cam_posn in zip(roll_interp, camera_traj):
#     plotter.camera.position = tuple(cam_posn)
#     plotter.camera.roll = roll
#     plotter.write_frame()
plotter.show(auto_close=True)  
plotter.close()    

#%%
# Get the timestamps for the flight trajs closest to the valid clusters
#%%
counts_by_batid = {}

for batid, batdf in flighttraj_interp.groupby('batid'):
    t_em = np.zeros(batdf.shape[0])
    counts_by_batid[batid] = np.zeros(t_em.size)
    print('bow')
    points_to_traj = distance_matrix(sources_nearish[:,:3],
                                     batdf.loc[:,'x':'z'].to_numpy())
    print('miaow')
    close_point_inds = np.where(points_to_traj<0.5)
    close_points = sources_nearish[np.unique(close_point_inds[0]),:]
    
    topx = 16
    video_audio_pairs = [[],[]]
    i = 0
    for candidate in close_points:
        xyz, timewindow = candidate[:3], candidate[-2:]
        potential_tof = distance_matrix(xyz.reshape(-1,3),
                                        array_geom)/vsound
        minmax_tof = np.percentile(potential_tof, [0,100])
        # get the widest window possible for the video trajs
        wide_timewindow = [np.min(timewindow[0]-potential_tof), 
                           np.max(timewindow[1]-potential_tof)]
        
        rows_inwindow = batdf['t'].between(wide_timewindow[0],
                                           wide_timewindow[1], inclusive=True)
        subset_df = batdf.loc[rows_inwindow,:]
        
        dist_to_clust = distance_matrix(xyz.reshape(-1,3),
                                        subset_df.loc[:,'x':'z'].to_numpy()).flatten()
        # dist_to_clust = distance_matrix(xyz.reshape(-1,3),
        #                                 relevant_traj.loc[:,'x':'z'].to_numpy()).flatten()
        inds_close = np.where(dist_to_clust<0.3)[0]
        
        if len(inds_close)>0:
            # if i>700:
            #raise ValueError('miaow')
            counts_by_batid[batid][np.argmin(dist_to_clust)] += 1 
            
            # closest_ind = np.argmin(dist_to_clust) # choose the closest
            # counts_by_batid[batid][closest_ind] +=1 
            #choose the top 5 points that are closest
            relevant_inds = subset_df.index[np.argsort(dist_to_clust)][:topx]
            #top_x_inds = np.argsort(dist_to_clust).flatten()[:topx]
            counts_by_batid[batid][relevant_inds] += 1   
                    
        i += 1 
        
#%%
fs = sf.info(audiofile).samplerate
audio, fs = sf.read(audiofile, start=int(fs*0), stop=int(fs*1.5))
num_bats = len(counts_by_batid.keys())


    
# def calculate_toa_across_mics(time_click, array_geom):
#     closest_traj_point = 
#     tof_mat = distance_matric(time_cli)
    

audio_channels = [0,2,3,4,5]

fig, axs = plt.subplots(ncols=1, nrows=num_bats+len(audio_channels),
                        figsize=(5, 10.0),
                        layout="tight", sharex=True)
traj_data = flighttraj_interp.groupby('batid')

# here I'll also establish an interactive workflow to get the bat calls 

def get_clicked_point_location(event):
    if event.button is MouseButton.LEFT:
        event_axes = [ax.in_axes(event) for ax in axs ]
        if sum(event_axes) == 0:
            return None, None
        else:
            axes_id = int(np.where(event_axes)[0])
            
            tclick, _ =  event.xdata, event.ydata
            curr_plot = axs[axes_id]
            num_lines = len(curr_plot.lines)
            plt.sca(curr_plot)
            if num_lines >2:
                for i in range(2,num_lines)[::-1]:
                    curr_plot.lines[i].remove()
            
            plt.plot(event.xdata, event.ydata,'r*')
#            plt.vlines(event.xdata-0.008, 0, 50, colors=colors[axes_id])
            return tclick, int(axes_id)
    else:
        return None, None

def calculate_toa_channels(t_source, sourceid, arraygeom):
    flight_traj = traj_data.get_group(sourceid).reset_index(drop=True)
    # get closest point in time 
    nearest_ind = abs(flight_traj['t']-t_source).argmin()
    emission_point = flight_traj.loc[nearest_ind, 'x':'z'].to_numpy().reshape(-1,3)
    tof = distance_matrix(emission_point, arraygeom)/343.0 
    toa = t_source + tof
    return toa

def draw_expected_toa(event, target_channels, window_halfwidth):
    start = time.time()
    t_emission, ax_id = get_clicked_point_location(event)
    if t_emission is not None:
        batid = ax_id + 1 
        toa = calculate_toa_channels(t_emission, batid, 
                                     array_geom[target_channels,:]).flatten()
        toa_2 = calculate_toa_channels(t_emission-window_halfwidth, batid, 
                                     array_geom[target_channels,:]).flatten()


        # get last few axes - that have specgrams
        specgram_axes = axs[-len(target_channels):]
        for channel_toa,channel_toa2, axid in zip(toa, toa_2, specgram_axes):
            plt.sca(axid)
            plt.vlines(channel_toa, 20000,96000, linestyle='dotted', color=colors[ax_id])
            plt.vlines(channel_toa2, 20000,96000, linestyle='dotted', color=colors[ax_id])
            
        fig.canvas.draw()
        #fig.canvas.draw_idle()



proximity_peaks = {}

for i,(batid, source_prof) in enumerate(counts_by_batid.items()):
    t_batid = flighttraj_interp.groupby('batid').get_group(batid)['t'].to_numpy()
    
    plt.sca(axs[i])
    plt.plot(t_batid, source_prof, label='bat id ' + str(batid), color=colors[int(batid)-1])
    plt.xticks([])
    plt.legend()
    
    pks, _ = signal.find_peaks(source_prof, distance=30,  height=15)
    proximity_peaks[batid] = t_batid[pks]
    plt.plot(t_batid[pks], source_prof[pks],'g*')


for i, ch in enumerate(audio_channels):
    plt.sca(axs[num_bats+i])
    plt.specgram(audio[:,ch], Fs=fs, xextent=[0,1.5], cmap='cividis')


def plotted_toa_from_peaks(specgram_axes, batid, peak_times, target_channels, window_halfwidth):

    for t_emission in peak_times:
        toa = calculate_toa_channels(t_emission, batid, 
                                     array_geom[target_channels,:]).flatten()
        toa_2 = calculate_toa_channels(t_emission-window_halfwidth, batid, 
                                     array_geom[target_channels,:]).flatten()
    

        for channel_toa,channel_toa2, axid in zip(toa, toa_2, specgram_axes):
            plt.sca(axid)
            plt.vlines(channel_toa, 20000,96000, linestyle='dotted', color=colors[batid-1])
            #plt.vlines(channel_toa2, 20000,96000, linestyle='dashed', color=colors[batid-1])
    fig.canvas.draw()

nchannels = len(audio_channels)
for batid in range(1,7):
    plotted_toa_from_peaks(axs[-nchannels:], batid, proximity_peaks[batid], audio_channels, 2e-3)

plt.gca().set_xticks(np.arange(0,1.5,0.05))
plt.gca().set_xticks(np.linspace(0,1.5,100), minor=True)


#%%
fs = sf.info(audiofile).samplerate
audio_clip, fs = sf.read(audiofile, stop=int(fs*time_tem.max()))

def fwdbkwd_avg(X, customwin=signal.windows.tukey(5)):
    fwd_pass = signal.convolve(X,customwin,'same')
    bkwd_pass = signal.convolve(X[::-1],customwin,'same')
    avg = (fwd_pass+bkwd_pass[::-1])/2
    return avg

peak_dist = 20
counts_smooth_trans = np.sqrt(fwdbkwd_avg(counts))
peaks_counts = signal.find_peaks(counts_smooth_trans, distance=peak_dist)[0]
counts_topx_smooth_trans = np.sqrt(fwdbkwd_avg(counts_cons))
peaks_counts_cons = signal.find_peaks(counts_topx_smooth_trans, distance=peak_dist)[0]

plt.figure()
a0 = plt.subplot(411)
plt.plot(time_tem, counts_smooth_trans)
plt.plot(time_tem[peaks_counts], counts_smooth_trans[peaks_counts], '*')

a01 = plt.subplot(412, sharex=a0)
plt.plot(time_tem, counts_topx_smooth_trans)
plt.plot(time_tem[peaks_counts_cons], counts_topx_smooth_trans[peaks_counts_cons], '*')

a1 = plt.subplot(413, sharex=a0)
plt.specgram(audio_clip[:,0], Fs=fs)
a1 = plt.subplot(414, sharex=a0)
plt.specgram(audio_clip[:,3], Fs=fs)

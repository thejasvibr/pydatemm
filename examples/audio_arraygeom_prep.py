"""
Something something here
========================
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import soundfile as sf
import scipy.signal as signal 
import glob

# find the file 
posix_timestamp = '1529434881'
folder_path = '/media/thejasvi/THEJASVI_DATA_BACKUP/'
subfolder_path = 'fieldwork_2018_001/actrackdata/wav/2018-06-19_001/'

file_path = glob.glob(folder_path+subfolder_path+'/*'+posix_timestamp+'.WAV', recursive=True)[0]

# cut out only the first 2 seconds 
fs = sf.info(file_path).samplerate
part_audio, fs = sf.read(file_path, start=0, stop=int(fs*2.0))

# align the audio using the sync channels 
delay_cc = signal.correlate(part_audio[:int(0.25*fs),7], part_audio[:int(0.25*fs),15], 'full')
# plt.plot(delay_cc)
delay_peak = int(np.argmax(delay_cc) -  delay_cc.size*0.5)

# fix delay of Fireface UC # 1 
synced_audio = np.column_stack((part_audio[delay_peak:,:8], part_audio[:-delay_peak,8:]))
# compare now 
plt.figure()
plt.plot(synced_audio[:int(0.25*fs),7])
plt.plot(synced_audio[:int(0.25*fs),15])

sf.write(f'first_twoseconds_{posix_timestamp}_2018-06-19_synced.wav', synced_audio, fs)

#%% Also prepare the csv file to make sense
array_geometry = pd.read_csv('Cave.csv', header=None)
array_geometry.columns = ['micname','x','y','z']
only_mic = array_geometry.loc[[1,2,3,4,5,6,7,8,12,13,14,15],:]
only_mic = only_mic.loc[[12,13,14,15,5,4,3,2,1,6,7,8],:]
only_mic['x'] = -only_mic['x']

plt.figure()
a0 = plt.subplot(111, projection='3d')
a0.set_box_aspect(aspect = (1,1,1))
a0.plot(only_mic['x'], only_mic['y'], only_mic['z'], '*')

only_mic.to_csv('array_geom_2018-06-19_onlymics.csv')

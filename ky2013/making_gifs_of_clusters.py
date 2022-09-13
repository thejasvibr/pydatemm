# -*- coding: utf-8 -*-
"""
Making GIF from multiple png files
===================================
Activate  ushichkawebsite environment in Win10, and otherwise
install moviepy and natsort. 

Created on Thu Sep  8 16:59:56 2022

@author: theja
"""
import moviepy
from moviepy.editor import ImageSequenceClip
import natsort
import glob

image_files = natsort.natsorted(glob.glob('*.png'))
clip = ImageSequenceClip(image_files, fps=1.5)
clip.write_gif('clustered_positions.gif', fps=1.5)
clip.close()




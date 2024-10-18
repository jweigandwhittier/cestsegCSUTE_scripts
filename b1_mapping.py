#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:51:19 2024

@author: jonah
"""
import os
import sys
if 'BART_TOOLBOX_PATH' in os.environ and os.path.exists(os.environ['BART_TOOLBOX_PATH']):
	sys.path.append(os.path.join(os.environ['BART_TOOLBOX_PATH'], 'python'))
elif 'TOOLBOX_PATH' in os.environ and os.path.exists(os.environ['TOOLBOX_PATH']):
	sys.path.append(os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
else:
	raise RuntimeError("BART_TOOLBOX_PATH is not set correctly!")
from bart import bart
import scripts.BrukerMRI as bruker
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

main_dir = '/Users/jonah/Documents/MRI_Data/Berkeley/HCM_New/'
animal_id = '20241011_141940_M1913_1_1'

directory = main_dir + animal_id

num_ref = 13

num_60deg = 10
num_120deg = 9

exp_ref = bruker.ReadExperiment(directory, num_ref)

exp_60deg = bruker.ReadExperiment(directory, num_60deg)
exp_120deg = bruker.ReadExperiment(directory, num_120deg)

ref_traj = exp_ref.traj
ref_ksp = exp_ref.GenerateKspace()
ref_ksp = np.squeeze(ref_ksp)
ref_ksp = np.expand_dims(ref_ksp, 0)
data_ref = bart(1, 'nufft -i', ref_traj, ref_ksp)
data_ref = bart(1, 'rss 8', data_ref)
data_ref = data_ref[:,:,0,0,0]
data_ref = np.squeeze(data_ref)
data_ref = np.abs(data_ref)
data_ref = np.rot90(data_ref, k=2)

data_60deg = exp_60deg.proc_data
data_60deg = np.flip(data_60deg,1)
data_120deg = exp_120deg.proc_data
data_120deg = np.flip(data_120deg,1)

plt.imshow(data_60deg[:,:,0])

b1 = np.arccos(data_120deg/(2*data_60deg))
b1 = np.nan_to_num(b1)
b1 = np.squeeze(b1)

b1 *= 180/np.pi
b1 = 60 - b1

b1_interp = ndimage.zoom(b1, zoom=2, order=1)

plt.imshow(data_ref, cmap='gray')
plt.imshow(b1_interp, cmap='plasma', alpha = 0.4)
plt.colorbar(label='$\Delta\Theta$ (°)')
# plt.title('Radial, ungated', fontsize = 16)
plt.title('$B_1$ Map')
plt.axis('off')
plt.show()
plt.savefig('b1_map_radial_gating.png', bbox_inches = 'tight', dpi=300)
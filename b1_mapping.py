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
from mpl_toolkits.axes_grid1 import make_axes_locatable

main_dir = '/Users/jonah/Documents/MRI_Data/Berkeley/Liver/'
animal_id = '20241103_165423_Liver_Test_1_3'

mask = np.load(main_dir + animal_id + '/Data/3uT_Liver_Radial/mask.npy')

directory = main_dir + animal_id

num_ref = 5

num_60deg = 7
num_120deg = 6

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
data_ref = np.rot90(data_ref, k=3)

data_60deg = exp_60deg.proc_data
data_60deg = np.flip(data_60deg,1)
data_120deg = exp_120deg.proc_data
data_120deg = np.flip(data_120deg,1)

y_indices, x_indices = np.where(mask)
x_min, x_max = max(np.min(x_indices) - 20, 0), min(np.max(x_indices) + 20, mask.shape[1])
y_min, y_max = max(np.min(y_indices) - 20, 0), min(np.max(y_indices) + 20, mask.shape[0])

plt.imshow(data_60deg[:,:,0])

b1 = np.arccos(data_120deg/(2*data_60deg))
b1 = np.nan_to_num(b1)
b1 = np.squeeze(b1)

b1 *= 180/np.pi
b1 = 60 - b1

b1_interp = ndimage.zoom(b1, zoom=2, order=1)
b1_interp *= mask

b1_interp_masked = np.where(mask, b1_interp, np.nan)

fig,ax=plt.subplots(1,1, figsize=(8,8))
ax.imshow(data_ref[y_min:y_max, x_min:x_max], cmap='gray')
im = ax.imshow(b1_interp_masked[y_min:y_max, x_min:x_max], cmap='plasma')
ax.set_title('$B_1$ Map', fontsize=22, fontname='Arial', weight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('$\Delta\\theta$ (°)', fontsize=18)
ax.axis('off')
plt.show()
plt.savefig('b1_map.png', bbox_inches='tight', dpi=300)
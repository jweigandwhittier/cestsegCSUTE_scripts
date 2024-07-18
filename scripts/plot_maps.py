#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:26:46 2024

@author: jonah
"""
from scripts import load_study, draw_rois, cest_fitting, misc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.signal import medfilt2d
from mpl_toolkits.axes_grid1 import make_axes_locatable

main_dir = '/Users/jonah/Documents/MRI_Data/Berkeley/Manuscript_Data/'
animal_id = '20240415_181720_C57BL6F3_1_5'
directory = main_dir + animal_id
exp = {'Cest': 12}

undersample = None

save_as = 'contrast_map'
savedir = directory + '/Data/' + save_as

x_start, x_end = 70, 150
y_start, y_end = 34, 114

data = load_study.load_study_bart(exp, directory, undersample, False)
proc_data = load_study.thermal_drift(data)
spectra, mask = draw_rois.Process_PerPixel(proc_data)
pixelwise = cest_fitting.per_pixel(proc_data['Cest'][1], spectra)

mt_list = []
amide_list = []
creatine_list = []
image = data['Cest'][0][:, :, 0]
for i in range(len(pixelwise)):
    mt_list.append(pixelwise[i][1]['Mt'])
    amide_list.append(pixelwise[i][1]['Amide'])
    creatine_list.append(pixelwise[i][1]['Creatine'])

# Initialize images with zeros
mt_image = np.zeros_like(mask, dtype=float)
amide_image = np.zeros_like(mask, dtype=float)
creatine_image = np.zeros_like(mask, dtype=float)

# Fill images with values from the lists
for i in range(len(mask)):
    for j in range(len(mask[0])):
        if mask[i][j]:
            mt_image[i][j] = mt_list.pop(0)
            amide_image[i][j] = amide_list.pop(0)
            creatine_image[i][j] = creatine_list.pop(0)

#Filter
mt_image = medfilt2d(mt_image, kernel_size = 3)
amide_image = medfilt2d(amide_image, kernel_size = 3)
creatine_image = medfilt2d(creatine_image, kernel_size = 3)

# Plotting with colorbars matching the image size
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

images = [mt_image, amide_image, creatine_image]
titles = ['MT', 'Amide', 'Creatine']

for i, (ax, img, title) in enumerate(zip(axs, images, titles)):
    ax.imshow(image[y_start:y_end,x_start:x_end], cmap='gray')
    im = ax.imshow(img[y_start:y_end,x_start:x_end], cmap='magma', alpha=0.7)
    ax.set_title(title, fontsize = 22)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=14)
    if i == 2:  # Only label the colorbar on the rightmost image (index 2)
        cbar.set_label('CEST Contrast (%)', fontsize = 16)
    ax.axis('off')

plt.tight_layout()
plt.show()
plt.savefig('cest_maps.tif', dpi=300)

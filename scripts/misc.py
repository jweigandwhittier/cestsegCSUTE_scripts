#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:14:14 2024

@author: jonah
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

def MakeDir(DirName):
    if os.path.isdir(DirName) == False:
        os.makedirs(DirName)
        
def SaveImg(Img, Dir, Save_As):
    fig, ax = plt.subplots(1,1)
    ax.imshow(Img, cmap='gray')
    ax.axis('off')
    plt.savefig(Dir + '/' + Save_As + '_Img.svg', bbox_inches = 'tight')
    plt.savefig(Dir + '/' + Save_As + '_Img.png', bbox_inches = 'tight')
    plt.close(fig)
    
def calc_ssim_matrix(images, mask):
    num_images = images.shape[2]
    ssim_matrix = np.zeros((num_images, num_images))
    # weights = np.ones_like(ssim_matrix)
    if mask is not None:
        mask = np.stack([mask] * num_images, axis=2)
        images *= mask
    for i in range(num_images):
        for j in range(i, num_images):
            img1 = images[:,:,i]
            img2 = images[:,:,j]
            similarity = ssim(img1, img2, data_range=img1.max() - img1.min())
            ssim_matrix[i, j] = similarity
            ssim_matrix[j, i] = similarity
            # if i == j:
                # weights[i,j] = 0
    # avg_ssim = np.average(ssim_matrix, weights=weights)
    avg_ssim = np.average(ssim_matrix)
    print('Mean SSIM is equal to %.2f' % avg_ssim)
    return ssim_matrix, avg_ssim

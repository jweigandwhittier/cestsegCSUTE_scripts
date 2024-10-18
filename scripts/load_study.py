#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:44:38 2024

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
import numpy as np
import scripts.BrukerMRI as bruker
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

def bulk_load(exp_list, directory):
    exps = {}
    for exp, num in exp_list.items():
        load = bruker.ReadExperiment(directory, num)
        exps[exp] = load
    return exps

def load_study_bruker(exp_list, directory, segmented):
    study = {}
    exps = bulk_load(exp_list, directory)
    for exp, data in exps.items():
        offsets = np.round(data.method["Cest_Offsets"]/data.method["PVM_FrqWork"][0],1)
        if exp == 'M0':
            offsets = [offsets]
        imgs = data.proc_data
        imgs = np.rot90(imgs, k=2)
        study[exp] = imgs, offsets
    if segmented == True:
        imgs = np.concatenate((study['Cest_Neg'][0], study['Cest_Mid'][0], study['Cest_Pos'][0]), axis=2)
        offsets = np.concatenate((study['Cest_Neg'][1], study['Cest_Mid'][1], study['Cest_Pos'][1]))
        m0 = study['M0']
        study = {}
        study['M0'] = m0
        study['Cest'] = imgs, offsets
        return study
    elif segmented == False:
        return study
    
def load_quesp(exp_list, directory):
    data = {}
    for exp, nums in exp_list.items():
        if exp == 'T1':
            load = bruker.ReadExperiment(directory, nums[0])
            img = load.proc_data
            inversion_times = load.method["MultiRepTime"]
            data[exp] = img, inversion_times
        if exp == 'Quesp':
            quesp = {}
            sat = {}
            for num in nums:
                load = bruker.ReadExperiment(directory, num)
                img = load.proc_data
                if 'PV-360' not in load.acqp['ACQ_sw_version']:
                    b1 = load.method['RF_Amplitude']
                    offsets = np.round(load.method["Cest_Offsets"]/load.method["PVM_FrqWork"][0],1)
                    quesp[b1] = img, offsets
                    sat['num_pulses'] = load.method["PVM_MagTransPulsNumb"]
                    sat['pulse_length'] = load.method["PVM_MagTransPulse1"][0]*10**-3
                    sat['pulse_delay'] = load.method["PVM_MagTransInterDelay"]*10**-3
                    sat['dc'] = sat['pulse_length']/(sat['pulse_length']+sat['pulse_delay'])
                    sat['tp'] = sat['pulse_length']*sat['num_pulses']/sat['dc']
                else:
                    b1 = load.method['PVM_SatTransPulseAmpl_uT']
                    offsets = load.method['PVM_SatTransFreqValues']
                    quesp[b1] = img, offsets
                    sat['num_pulses'] = load.method["PVM_SatTransNPulses"]
                    sat['pulse_length'] = load.method["PVM_StP0"]*10**-6
                    sat['pulse_delay'] = load.method["PVM_SatTransInterPulseDelay"]*10**-3
                    sat['dc'] = sat['pulse_length']/(sat['pulse_length']+sat['pulse_delay'])
                    sat['tp_cw'] = sat['pulse_length']*sat['num_pulses']/sat['dc']
                    
            data[exp] = quesp
    return sat, data
                
    
def load_study_bart(exps, directory, undersample, segmented):
    study = {}
    exps = bulk_load(exps, directory)
    for exp, data in exps.items():
        imgs = []
        offsets = np.round(data.method["Cest_Offsets"] / data.method["PVM_FrqWork"][0], 2)
        if exp == 'M0':
            offsets = [offsets]
        traj = data.traj
        ksp = data.GenerateKspace()
        if undersample is not None:
            samp = np.size(ksp, axis=1)
            samp = int(samp * undersample)
            ksp = ksp[:, :samp, :, :]
            traj = traj[:, :, :samp]
        for i in range(len(offsets)):
            offset_ksp = ksp[:, :, :, i]
            offset_ksp = np.expand_dims(offset_ksp, axis=0)
            img = bart(1, 'nufft -i', traj, offset_ksp)
            img = bart(1, 'rss 8', img)
            img = np.abs(img)
            imgs.append(img)
        imgs = np.stack(imgs, axis=2)
        satisfied = False
        while not satisfied:
            # Display the image and ensure it renders before the input prompt
            fig, ax = plt.subplots(1, 1)
            ax.imshow(imgs[:, :, 0], cmap='gray')
            ax.axis('off')
            fig.suptitle('Input number of rotations.')
            plt.show(block=False)
            plt.pause(0.1)  # Allow the plot to render
            # Input validation for number of rotations
            while True:
                try:
                    num_rot = int(input('Enter the number of 90-degree counterclockwise rotations to align ventral (top) to dorsal (bottom) (0-3): '))
                    if num_rot in range(4):
                        break
                    else:
                        print("Please enter an integer between 0 and 3.")
                except ValueError:
                    print("Invalid input. Please enter an integer between 0 and 3.")
            # Rotate the images
            plt.close(fig)
            rotated_imgs = np.rot90(imgs, k=num_rot, axes=(0, 1))
            # Display the rotated image for confirmation
            fig, ax = plt.subplots(1, 1)
            ax.imshow(rotated_imgs[:, :, 0], cmap='gray')
            ax.axis('off')
            fig.suptitle('Is this rotation correct? (yes/no)')
            plt.show(block=False)
            plt.pause(0.1)  # Allow the plot to render
            while True:
                user_input = input('Is this rotation correct? (yes/no): ').strip().lower()
                if user_input in ['yes', 'no']:
                    satisfied = (user_input == 'yes')
                    break
                else:
                    print("Please enter 'yes' or 'no'.")
            plt.close(fig)
        imgs = rotated_imgs
        study[exp] = imgs, offsets
    if segmented:
        imgs = np.concatenate((study['Cest_Neg'][0], study['Cest_Mid'][0], study['Cest_Pos'][0]), axis=2)
        offsets = np.concatenate((study['Cest_Neg'][1], study['Cest_Mid'][1], study['Cest_Pos'][1]))
        m0 = study['M0']
        study = {'M0': m0, 'Cest': (imgs, offsets)}
        return study
    else:
        return study
    
def thermal_drift_old(data):
    THRESHOLD_PPM = 15
    image = data['Cest'][0]
    offsets = data['Cest'][1]
    # Find reference index
    ref_index = np.where(offsets > THRESHOLD_PPM)[0]
    # Apply normalization
    for i in range(len(ref_index)):
        m0 = image[:, :, ref_index[i]]
        if i < len(ref_index) - 1:
            next_index = ref_index[i + 1]
        else:
            next_index = image.shape[2]
        for j in range(ref_index[i] + 1, next_index):
            image[:, :, j] /= m0
            image = np.nan_to_num(image)
    data = {}
    m0 = image[:,:,ref_index[0]]
    image = np.delete(image, ref_index, axis=2)
    offsets = np.delete(offsets, ref_index)
    data['M0'] = m0
    data['Cest'] = image, offsets
    return data  

def thermal_drift(data):
    THRESHOLD_PPM = 15
    images = data['Cest'][0]
    offsets = data['Cest'][1]
    # Find reference index
    ref_index = np.where(offsets > THRESHOLD_PPM)[0]
    # Apply normalization
    m0 = images[:,:,ref_index]
    offsets = np.delete(offsets, ref_index)
    images = np.delete(images, ref_index, axis=2)
    
    if np.size(ref_index) > 1:
        step = ref_index[1]-1
        ref_offsets = np.concatenate(([offsets[0]], offsets[step-1::step], [offsets[-1]]))

        matrix = np.size(images, 0)
        grid_index = np.arange(0,matrix)

        points = (grid_index,grid_index,ref_offsets)
        xi, yi, fi = np.meshgrid(grid_index, grid_index, offsets, indexing='ij')
        values = np.stack((xi, yi, fi), axis=-1)

        m0_interp = interpn(points, m0, values)
        images = np.nan_to_num(images/m0_interp)
    
        proc_data = {
            'Cest': (images, offsets),
            'M0': m0[:, :, 0],
            'M0_Interp': m0_interp
            }
    else:
        images = np.nan_to_num(images/m0)
        proc_data = {
        'Cest': (images, offsets),
        'M0': m0[:, :, 0]
        }
    return proc_data  

def thermal_drift_quesp(data):
    THRESHOLD_PPM = 15
    for b1, acq in data['Quesp'].items():
        images = acq[0]
        offsets = acq[1]
        # Find reference index
        ref_index = np.where(offsets > THRESHOLD_PPM)[0]
        # Apply normalization
        m0 = images[:,:,ref_index]
        step = ref_index[1]-1
        offsets = np.delete(offsets, ref_index)
        images = np.delete(images, ref_index, axis=2)
        ref_offsets = np.concatenate(([offsets[0]], offsets[step-1::step], [offsets[-1]]))
    
        matrix = np.size(images, 0)
        grid_index = np.arange(0,matrix)
    
        points = (grid_index,grid_index,ref_offsets)
        xi, yi, fi = np.meshgrid(grid_index, grid_index, offsets, indexing='ij')
        values = np.stack((xi, yi, fi), axis=-1)
    
        m0_interp = interpn(points, m0, values)
        images = np.nan_to_num(images/m0_interp)
        data['Quesp'][b1] = images, offsets
    return data  
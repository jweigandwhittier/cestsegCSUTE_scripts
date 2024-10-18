#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:56:50 2024

@author: jonah
"""
import roipoly
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def Process_Avg(Data):
    M0 = np.squeeze(Data['M0'][0])
    Imgs = Data['Cest'][0]
    Mask = Rois_Avg(M0)
    Spectra = []
    Pixels = []
    M0[~Mask] = 1
    for i in range(np.size(Imgs, axis=2)):
        Image = Imgs[:,:,i]
        Image[~Mask] = 0 
        Image /= M0
        Image = Image.flatten()[Image.flatten() != 0]
        Pixels.append(Image)
    Pixels = np.array(Pixels)
    Spectrum = np.mean(Pixels, axis=1) 
    Spectra.append(Spectrum)
    Spectra = np.array(Spectra)
    Spectra = np.swapaxes(Spectra, 0, 1)
    return Spectra

def Process_PerPixel(Data):
    M0 = np.squeeze(Data['M0'])
    Imgs = Data['Cest'][0]
    Mask = Rois_Avg(M0)
    Spectra = []
    Pixels = []
    M0[~Mask] = 1
    for i in range(np.size(Imgs, axis=2)):
        Image = Imgs[:,:,i]
        Image[~Mask] = 0 
        Image = Image.flatten()[Image.flatten() != 0]
        Pixels.append(Image)
    Pixels = np.array(Pixels)
    Pixels = np.swapaxes(Pixels, 0, 1)
    Spectra = Pixels.tolist()
    return Spectra, Mask

def process_aha(data):
    m0 = np.squeeze(data['M0'][0])
    imgs = data['Cest'][0]
    labeled_segments, mask = aha_segmentation(m0)
    show_segmentation(mask, m0, labeled_segments)
    spectra = {}
    for label, segment in labeled_segments.items():
        pixels = []
        segment_mask = np.zeros(np.shape(imgs[:,:,0])) 
        for coord in segment:
            segment_mask[coord[0], coord[1]] = 1  
        for i in range(np.size(imgs, axis=2)):
            img = imgs[:,:,i]      
            img_seg = img*segment_mask
            img_seg = np.nan_to_num(img_seg/m0)
            img_seg = img_seg.flatten()[img_seg.flatten() != 0]
            pixels.append(img_seg)
        pixels = np.array(pixels)
        spectrum = np.mean(pixels, axis=1)
        spectra[label] = spectrum
    return spectra

def process_aha_thermal_drift(data, savedir, save_as):
    m0 = np.squeeze(data['M0'])
    imgs = data['Cest'][0]
    labeled_segments, mask, rois = aha_segmentation(m0, savedir, save_as)
    spectra = {}
    for label, segment in labeled_segments.items():
        pixels = []
        segment_mask = np.zeros(np.shape(imgs[:,:,0])) 
        for coord in segment:
            segment_mask[coord[0], coord[1]] = 1  
        for i in range(np.size(imgs, axis=2)):
            img = imgs[:,:,i]      
            img_seg = img*segment_mask
            img_seg = img_seg.flatten()[img_seg.flatten() != 0]
            pixels.append(img_seg)
        pixels = np.array(pixels)
        spectrum = np.mean(pixels, axis=1)
        spectra[label] = spectrum
    return mask, labeled_segments, spectra
    
def Rois_Avg(M0):
    Roi_List = ['Epicardium', 'Endocardium']
    Fig, Ax = plt.subplots(1,1)
    Fig.suptitle('Draw ROIs')
    Ax.imshow(M0, cmap='gray')
    Ax.axis('off')
    MultiRoi_Named = roipoly.MultiRoi(roi_names = Roi_List)
    Mask_Exterior = MultiRoi_Named.rois['Epicardium'].get_mask(M0)
    Mask_Interior = MultiRoi_Named.rois['Endocardium'].get_mask(M0)
    Mask = np.logical_and(Mask_Exterior, np.logical_not(Mask_Interior))
    return Mask

def rois_avg_phantom(m0):
    fig, ax = plt.subplots(1,1)
    fig.suptitle('Draw ROIs')
    ax.imshow(m0, cmap='gray')
    ax.axis('off')
    roi = roipoly.RoiPoly(color = 'r')
    mask = roi.get_mask(m0)
    return mask
    
def process_thermal_drift(data):
    m0 = np.squeeze(data['M0'])
    imgs = data['Cest'][0]
    offsets = data['Cest'][1]
    mask = Rois_Avg(m0)
    pixels = []
    spectra = []
    for i in range(np.size(imgs, axis=2)):
        image = imgs[:,:,i]
        image[~mask] = 0 
        image = image.flatten()[image.flatten() != 0]
        pixels.append(image)
    pixels = np.array(pixels)
    spectrum = np.mean(pixels, axis=1)
    spectra.append(spectrum)
    spectra = np.array(spectra)
    spectra = np.swapaxes(spectra, 0, 1)
    return spectra

def process_thermal_drift_phantom(data, mask):
    m0 = np.squeeze(data['M0'])
    imgs = data['Cest'][0]
    offsets = data['Cest'][1]
    if mask is not None:
        pass
    else:
        mask = rois_avg_phantom(m0)
    spectra = []
    pixels = []
    for i in range(np.size(imgs, axis=2)):
        image = imgs[:,:,i]
        image[~mask] = 0 
        image = image.flatten()[image.flatten() != 0]
        pixels.append(image)
    pixels = np.array(pixels)
    spectrum = np.mean(pixels, axis=1)
    spectra.append(spectrum)
    spectra = np.array(spectra)
    spectra = np.swapaxes(spectra, 0, 1)
    return spectra

def process_thermal_drift_quesp(data, mask):
    imgs = data[0]
    offsets = data[1]
    spectra = []
    pixels = []
    for i in range(np.size(imgs, axis=2)):
        image = imgs[:,:,i]
        image[~mask] = 0 
        image = image.flatten()[image.flatten() != 0]
        pixels.append(image)
    pixels = np.array(pixels)
    pixels = np.swapaxes(pixels, 0, 1)
    spectra = pixels.tolist()
    return spectra

def show_segmentation(mask, rois, image, labeled_segments, savedir, save_as):
    segmented = np.zeros((np.size(mask, 0), np.size(mask, 1), 3))
    coords = {
        'Inferoseptal': (255, 0, 0),      # red       inferoseptal
        'Anteroseptal': (0, 255, 0),      # green     anteroseptal
        'Anterior': (0, 0, 255),          # blue      anterior
        'Anterolateral': (255, 165, 0),   # orange    anterolateral
        'Inferolateral': (255, 255, 100), # yellow    inferolateral
        'Inferior': (128, 0, 128)         # purple    inferior
    }
    for segment, color in coords.items():
        for coord in labeled_segments[segment]:
            segmented[coord[0], coord[1]] = np.array(color, dtype=np.uint8)
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].imshow(image, cmap='gray')  # Overlay grayscale image
    # Overlay segmented regions
    axs[0].imshow(segmented, alpha=0.5)
    # Create legend
    legend_elements = [Patch(facecolor=np.array(color)/255, edgecolor='black', label=label) for label, color in coords.items()]
    axs[0].legend(handles=legend_elements, loc=4)
    axs[1].imshow(image, cmap='gray')
    roi_names = []
    for name, roi in rois.rois.items():
        roi.display_roi()
        roi_names.append(name)
    axs[1].legend(roi_names, loc=4)
    for ax in axs:
        ax.axis('off')
    plt.show()
    plt.pause(0.1)
    plt.savefig(savedir + '/' + save_as + '_ROIs.svg', bbox_inches = 'tight')
    plt.savefig(savedir + '/' + save_as + '_ROIs.png', bbox_inches = 'tight')

def aha_segmentation(image, savedir, save_as):
    ## Distance calculation ##
    def distance(co1, co2):
        return abs(co1[0] - co2[0])**2 + abs(co1[1] - co2[1])**2
    ## Define centroid ##
    def centroid(array):
        x_c = 0
        y_c = 0
        area = array.sum()
        it = np.nditer(array, flags=['multi_index'])
        for i in it:
            x_c = i * it.multi_index[1] + x_c
            y_c = i * it.multi_index[0] + y_c
        return (int(x_c / area), int(y_c / area))
    while True:
        ## Make image ##
        roi_list = ['Insertion Points (anterior --> inferior)', 'Epicardium', 'Endocardium']
        ## Plot image ##
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Draw ROIs ("New ROI" to start)')
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ## Multi ROIs (epi, endo, insertion points) ##
        multiroi_named = roipoly.MultiRoi(roi_names=roi_list)
        ## Get myocardium mask ##
        mask_epi = multiroi_named.rois['Epicardium'].get_mask(image)
        mask_endo = multiroi_named.rois['Endocardium'].get_mask(image)
        mask = np.logical_and(mask_epi, np.logical_not(mask_endo))
        ## Get coordinates for myocardium and insertion points ##
        mask_coords = np.argwhere(mask == True)
        mask_coords = np.flip(mask_coords, axis=1)
        ip_coords = multiroi_named.rois['Insertion Points (anterior --> inferior)'].get_roi_coordinates()
        ## Get points in myocardium with closest proximity to defined insertion points ##
        insertion_points = []
        for coord in ip_coords:
            closest = mask_coords[0]
            for c in mask_coords:
                if distance(c, coord) < distance(closest, coord):
                    closest = c
            insertion_points.append(closest)
        arv = insertion_points[0]
        irv = insertion_points[1]
        [cx, cy] = centroid(mask)
        [y, x] = np.nonzero(mask)
        inds = np.nonzero(mask)
        inds = list(zip(inds[0], inds[1]))
        # Offset all points by centroid
        x = x - cx
        y = y - cy
        arvx = arv[0] - cx
        arvy = arv[1] - cy
        irvx = irv[0] - cx
        irvy = irv[1] - cy
        # Find angular segment cutoffs
        pi = math.pi
        angle = lambda a, b: (math.atan2(a, b)) % (2 * pi)
        arv_ang = angle(arvy, arvx)
        irv_ang = angle(irvy, irvx)
        ang = [angle(yc, xc) for yc, xc in zip(y, x)]
        sept_cutoffs = np.linspace(0, arv_ang - irv_ang, num=3)  # two septal segments
        wall_cutoffs = np.linspace(arv_ang - irv_ang, 2 * pi, num=5)  # four wall segments
        cutoffs = []
        cutoffs.extend(sept_cutoffs)
        cutoffs.extend(wall_cutoffs[1:])
        ang = [(a - irv_ang) % (2 * pi) for a in ang]
        # Create arrays of each pixel/index in each segment
        segment_image = lambda a, b: [j for (i, j) in enumerate(inds) if ang[i] >= a and ang[i] < b]
        get_pixels = lambda inds: [image[i] for i in inds]
        segmented_indices = [segment_image(a, b) for a, b in zip(cutoffs[:6], cutoffs[1:])]
        segmented_pixels = [get_pixels(inds) for inds in segmented_indices]
        # List of labeled segments
        labeled_segments = {}
        labeled_segments['Inferoseptal'] = segmented_indices[0]
        labeled_segments['Anteroseptal'] = segmented_indices[1]
        labeled_segments['Anterior'] = segmented_indices[2]
        labeled_segments['Anterolateral'] = segmented_indices[3]
        labeled_segments['Inferolateral'] = segmented_indices[4]
        labeled_segments['Inferior'] = segmented_indices[5]
        # Show segmentation and ask for confirmation
        show_segmentation(mask, multiroi_named, image, labeled_segments, savedir, save_as)
        while True:
            user_input = input('Are you satisfied with the segmentation? (yes/no): ').strip().lower()
            if user_input in ['yes', 'no']:
                break
            else:
                print("Please enter 'yes' or 'no'.")
        if user_input == 'yes':
            break
    return labeled_segments, mask, multiroi_named

def aha_per_pixel(data, savedir, save_as):
    m0 = np.squeeze(data['M0'])
    imgs = data['Cest'][0]
    labeled_segments, mask, rois = aha_segmentation(m0, savedir, save_as)
    spectra = []
    pixels = []
    for i in range(np.size(imgs, axis=2)):
        image = imgs[:,:,i]
        image[~mask] = 0 
        image = image.flatten()[image.flatten() != 0]
        pixels.append(image)
    pixels = np.array(pixels)
    pixels = np.swapaxes(pixels, 0, 1)
    spectra = pixels.tolist()
    return mask, labeled_segments, spectra
    
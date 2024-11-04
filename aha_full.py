#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:33:15 2024

@author: jonah
"""

from scripts import load_study, draw_rois, cest_fitting, wassr, misc
import pickle 
import matplotlib.pyplot as plt

#-------------Variables to set--------------#
#Primary data directory and experiment number(s) with labels 
main_dir = '/Users/jonah/Documents/MRI_Data/Berkeley/HCM_New/'
animal_id = '20241011_141940_M1913_1_1'
directory = main_dir + animal_id
exp_cest = {'Cest':12}
#Specify WASSR experiment (if there is one)
exp_wassr = {'Cest':13}
#Specify QUESP experiment (if there is one)
exp_quesp = {}
#Save as#
save_as = 'Cest_1p1uT'
#Undersampling ('None' or e.g., 0.5)#
undersample = None
#-------------Actually do the thing--------------#
savedir = directory + '/Data/' + save_as
#Make dir#
misc.MakeDir(savedir)
#Load data, use thermal drift normalization
data = load_study.load_study_bart(exp_cest, directory, undersample, False)
proc_data = load_study.thermal_drift(data)
#Save M0#
misc.SaveImg(proc_data['M0'], savedir, save_as)
#Process data, draw ROIs, save mask and segments
mask, labeled_segments, spectra = draw_rois.process_aha_thermal_drift(proc_data, savedir, save_as)
#Fit WASSR for B0 map if WASSR acquisition exists
if exp_wassr and len(exp_wassr) > 0:
    data_wassr = load_study.load_study_bart(exp_wassr, directory, None, False)
    proc_data_wassr = load_study.thermal_drift(data_wassr)
    mask, labeled_segments, spectra_wassr = draw_rois.aha_per_pixel(proc_data_wassr, mask, labeled_segments, savedir, save_as)
    pixelwise_wassr = cest_fitting.wassr(proc_data_wassr['Cest'][1], spectra_wassr)
    wassr.plot_wassr(pixelwise_wassr, proc_data_wassr, mask, labeled_segments, savedir, save_as)
#Fit QUESP if QUESP acquisition exists
if exp_quesp and len(exp_wassr) > 0:
    
#Fit z-spectra
fits = cest_fitting.two_step_aha(proc_data['Cest'][1], spectra)
##Pickle and save data##
with open(savedir + '/fits_%s.pkl' % save_as, 'wb') as handle:
    pickle.dump(fits, handle, protocol=pickle.HIGHEST_PROTOCOL)
##Plots fits##
cest_fitting.plot_zspec_aha(fits, savedir, save_as)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:48:42 2024

@author: jonah
"""

from scripts import load_study, draw_rois, cest_fitting, wassr, misc
import pickle 
import matplotlib.pyplot as plt

#-------------Variables to set--------------#
#Primary data directory and experiment number
main_dir = '/Users/jonah/Documents/MRI_Data/Berkeley/Liver/'
animal_id = '20241103_165423_Liver_Test_1_3'
directory = main_dir + animal_id
exp = {'Cest':14}
#Specify WASSR experiment (if there is one)
exp_wassr = {'Cest':5}
#Save as#
save_as = '3uT_Liver_Radial'
#Undersampling ('None' or e.g., 0.5)#
undersample = None
#-------------Actually do the thing, modified for LIVER--------------#
savedir = directory + '/Data/' + save_as
#Make dir#
misc.MakeDir(savedir)
# #Load data, use thermal drift normalization
data = load_study.load_study_bart(exp, directory, undersample, False)
proc_data = load_study.thermal_drift(data)
#Save M0#
misc.SaveImg(proc_data['M0'], savedir, save_as)
#Try for single ROI
mask, spectra = draw_rois.process_thermal_drift_liver(proc_data, savedir, None)
#Fit WASSR for B0 map if WASSR acquisition exists
if exp_wassr and len(exp_wassr) > 0:
    data_wassr = load_study.load_study_bart(exp_wassr, directory, None, False)
    proc_data_wassr = load_study.thermal_drift(data_wassr)
    mask, spectra_wassr = draw_rois.per_pixel(proc_data_wassr, mask, savedir, save_as)
    pixelwise_wassr = cest_fitting.wassr(proc_data_wassr['Cest'][1], spectra_wassr)
    wassr.plot_wassr(pixelwise_wassr, proc_data_wassr, mask, savedir, save_as)
#Fit z-spectra
fits = cest_fitting.two_step(proc_data['Cest'][1], spectra)
# ##Pickle and save data##
with open(savedir + '/fits_%s.pkl' % save_as, 'wb') as handle:
    pickle.dump(fits, handle, protocol=pickle.HIGHEST_PROTOCOL)
##Plots fits##
cest_fitting.plot_zspec(fits[0], savedir, save_as)
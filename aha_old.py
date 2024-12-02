#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:33:15 2024

@author: jonah
"""

from scripts import load_study, draw_rois, cest_fitting, misc
import pickle 
import matplotlib.pyplot as plt

#-------------Variables to set--------------#
#Primary data directory and experiment number
main_dir = '/Users/jonah/Documents/MRI_Data/Berkeley/Manuscript_Data/'
animal_id = '20240531_133343_214100_1R1L_1_5'
directory = main_dir + animal_id
exp = {'Cest':11}
#Save as#
save_as = 'Demo'
#Undersampling ('None' or e.g., 0.5)#
undersample = None
#-------------Actually do the thing--------------#
savedir = directory + '/Data/' + save_as
#Make dir#
misc.MakeDir(savedir)
#Load data, use thermal drift normalization
data = load_study.load_study_bart(exp, directory, undersample, False)
proc_data = load_study.thermal_drift(data)
#Save M0#
misc.SaveImg(proc_data['M0'], savedir, save_as)
#Process data, draw ROIs
mask, spectra = draw_rois.process_aha_thermal_drift(proc_data, savedir, save_as)
#Calc SSIM matrix
ssim_matrix, avg_ssim = misc.calc_ssim_matrix(data['Cest'][0], mask)
#Fit z-spectra
fits = cest_fitting.two_step_aha(proc_data['Cest'][1], spectra)
##Pickle and save data##
with open(savedir + '/fits_%s.pkl' % save_as, 'wb') as handle:
    pickle.dump(fits, handle, protocol=pickle.HIGHEST_PROTOCOL)
##Plots fits##
cest_fitting.plot_zspec_aha(fits, savedir, save_as)
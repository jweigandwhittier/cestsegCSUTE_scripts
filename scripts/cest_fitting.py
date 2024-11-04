#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:08:05 2024

@author: jonah
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.io import savemat
from scipy.interpolate import CubicSpline


###Pre-correction###
##Starting points for curve fitting: amplitude, FWHM, peak center##
p0_water = [0.8, 1.8, 0]
p0_mt = [0.15, 40, -1]
##Lower bounds for curve fitting##
lb_water = [0.02, 0.3, -10]
lb_mt = [0.0, 30, -2.5]
##Upper bounds for curve fitting##
ub_water = [1, 10, 10]
ub_mt = [0.5, 60, 0]

##Try different starting points for water and creating FWHM in phantom##
# p0_water = [0.8, 0.1, 0]

##Combine for curve fitting##
#B0 correction (tissue)
p0_corr = p0_water + p0_mt
lb_corr = lb_water + lb_mt
ub_corr = ub_water + ub_mt 

#B0 correction (phantom)
p0_corr_ph = p0_water
lb_corr_ph = lb_water
ub_corr_ph = ub_water

###Post-correction###
##Starting points for curve fitting: amplitude, FWHM, peak center##
p0_water = [0.8, 0.2, 0]
p0_mt = [0.15, 40, -1]
p0_noe = [0.05, 1, -2.75]
p0_creatine = [0.05, 0.5, 2.0]
p0_amide = [0.05, 1.5, 3.5]
##Lower bounds for curve fitting##
lb_water = [0.02, 0.01, -1e-6]
lb_mt = [0.0, 30, -2.5]
lb_noe = [0.0, 0.5, -4.5]
lb_creatine = [0.0, 0.5, 1.6]
lb_amide = [0.0, 0.5, 3.2]
##Upper bounds for curve fitting##
ub_water = [1, 10, 1e-6]
ub_mt = [0.5, 60, 0]
ub_noe = [0.25, 5, -1.5]
ub_creatine = [0.5, 5, 2.6]
ub_amide = [0.3, 5, 4.0]

##Try different starting points for water and creatine FWHM in phantom##
# p0_water = [0.8, 0.1, 0]
# p0_creatine = [0.05, 0.25, 2.6]

##Combine for curve fitting##
#Step 1
p0_1 = p0_water + p0_mt
lb_1 = lb_water + lb_mt
ub_1 = ub_water + ub_mt 
#Step 2 (cardiac)
p0_2 = p0_noe + p0_creatine + p0_amide
lb_2 = lb_noe + lb_creatine + lb_amide
ub_2 = ub_noe + ub_creatine + ub_amide
#Single step (Cr phantom)
p0_ph = p0_water + p0_creatine
lb_ph = lb_water + lb_creatine
ub_ph = ub_water + ub_creatine


#Cutoffs and options for fitting
Cutoffs = [-4, -1.4, 1.4, 4]
options = {'xtol': 1e-10, 'ftol': 1e-4, 'maxfev': 50}

def Lorentzian(x, Amp, Fwhm, Offset):
    Num = Amp * 0.25 * Fwhm ** 2
    Den = 0.25 * Fwhm ** 2 + (x - Offset) ** 2
    return Num/Den

def Step_1_Fit(x, *fit_parameters):
    Water_Fit = Lorentzian(x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
    Mt_Fit = Lorentzian(x, fit_parameters[3], fit_parameters[4], fit_parameters[5])
    Fit = 1 - Water_Fit - Mt_Fit
    return Fit

def Step_2_Fit(x, *fit_parameters):
    Noe_Fit = Lorentzian(x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
    Creatine_Fit = Lorentzian(x, fit_parameters[3], fit_parameters[4], fit_parameters[5])
    Amide_Fit = Lorentzian(x, fit_parameters[6], fit_parameters[7], fit_parameters[8])
    Fit = Noe_Fit + Creatine_Fit + Amide_Fit
    return Fit

def Step_2_Fit_Apt(x, *fit_parameters):
    Noe_Fit = Lorentzian(x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
    Amide_Fit = Lorentzian(x, fit_parameters[3], fit_parameters[4], fit_parameters[5])
    Fit = Noe_Fit + Amide_Fit
    return Fit

def Cr_Phantom_Fit(x, *fit_parameters):
    Water_Fit = Lorentzian(x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
    Creatine_Fit = Lorentzian(x, fit_parameters[3], fit_parameters[4], fit_parameters[5])
    Fit = 1 - Water_Fit - Creatine_Fit
    return Fit

def Water_Fit_Correction(x, *fit_parameters):
    Water_Fit = Lorentzian(x, fit_parameters[0], fit_parameters[1], fit_parameters[2])
    Fit = 1 - Water_Fit
    return Fit

def two_step_aha(Offsets, Spectra):
    start = time.time()
    n_interp = 4000
    Offsets_Interp = np.linspace(Offsets[0], Offsets[-1], n_interp)
    Offsets_Corrected = np.zeros_like(Offsets)
    Fits = {}
    for Segment, Spectrum in Spectra.items():
        if Offsets[0] > 0:
            Offsets = np.flip(Offsets)
            Spectrum = np.flip(Spectrum)
        ##Fit for corrections##
        Fit_1, _ = curve_fit(Step_1_Fit, Offsets, Spectrum, p0=p0_corr, bounds=(lb_corr, ub_corr), **options)
        ##Calculate water and MT fits from parameters##
        Water_Fit = Lorentzian(Offsets, Fit_1[0], Fit_1[1], Fit_1[2])
        Mt_Fit = Lorentzian(Offsets, Fit_1[3], Fit_1[4], Fit_1[5])
        ##B0 correction##
        Correction = Fit_1[2]
        # Correction = Offsets[np.argmax(Water_Fit+Mt_Fit)]
        Offsets_Corrected = Offsets - Correction
        # Spectrum = riccian_noise_correction(Spectrum, Offsets_Corrected)
        Condition = (Offsets_Corrected <= Cutoffs[0]) | (Offsets_Corrected >= Cutoffs[3]) | \
                    ((Offsets_Corrected >= Cutoffs[1]) & (Offsets_Corrected <= Cutoffs[2]))
        Condition_RMSE = ((Offsets_Corrected <= -1.4) & (Offsets_Corrected >= -4)) | \
                    ((Offsets_Corrected >= 1.4) & (Offsets_Corrected <= 4))
        Offsets_Cropped = Offsets_Corrected[Condition]
        Spectrum_Cropped = Spectrum[Condition]
        ##Set up interpolated frequency axis
        Offsets_Interp = np.linspace(Offsets_Corrected[0], Offsets_Corrected[-1], n_interp)
        Fit_1, _ = curve_fit(Step_1_Fit, Offsets_Cropped, Spectrum_Cropped, p0=p0_1, bounds=(lb_1, ub_1), **options)
        ##Calculate water and MT fits from parameters##
        Water_Fit = Lorentzian(Offsets_Interp, Fit_1[0], Fit_1[1], Fit_1[2])
        Mt_Fit = Lorentzian(Offsets_Interp, Fit_1[3], Fit_1[4], Fit_1[5])
        ##Calculate background and Lorentzian difference##
        Background = Lorentzian(Offsets_Corrected, Fit_1[0], Fit_1[1], Fit_1[2]) + Lorentzian(Offsets_Corrected, Fit_1[3], Fit_1[4], Fit_1[5])
        Lorentzian_Difference = 1 - (Spectrum + Background)
        Step_1_Fit_Values = Step_1_Fit(Offsets_Corrected, *Fit_1)
        ##Get RMSE and residuals
        Step_1_Fit_Values = Step_1_Fit(Offsets_Corrected, *Fit_1)
        Step_1_Residuals = Spectrum - Step_1_Fit_Values
        Step_1_RMSE = np.sqrt(mean_squared_error(Spectrum, Step_1_Fit_Values))
        ##Step 2##
        Fit_2, _ = curve_fit(Step_2_Fit, Offsets_Corrected, Lorentzian_Difference, p0=p0_2, bounds=(lb_2, ub_2), **options)
        ##Calulate NOE, creatine, and amide fits from parameters##
        Noe_Fit = Lorentzian(Offsets_Interp, Fit_2[0], Fit_2[1], Fit_2[2])
        Creatine_Fit = Lorentzian(Offsets_Interp, Fit_2[3], Fit_2[4], Fit_2[5])
        Amide_Fit = Lorentzian(Offsets_Interp, Fit_2[6], Fit_2[7], Fit_2[8])
        ##Get RMSE and residuals
        Step_2_Fit_Values = Step_2_Fit(Offsets_Corrected, *Fit_2)
        Step_2_Residuals = Lorentzian_Difference - Step_2_Fit_Values
        Step_2_RMSE = np.sqrt(mean_squared_error(Lorentzian_Difference, Step_2_Fit_Values))
        ##Calculate total fit and RMSE in specified regions
        Total_Fit = Step_1_Fit_Values - Step_2_Fit_Values
        Spectrum_Region = Spectrum[Condition_RMSE]
        Total_Fit_Region = Total_Fit[Condition_RMSE]
        Residuals = Spectrum_Region - Total_Fit_Region
        RMSE = np.sqrt(mean_squared_error(Spectrum_Region, Total_Fit_Region))
        ##Flip to match NMR convention##
        Offsets_Interp = np.flip(Offsets_Interp)
        Offsets_Corrected = np.flip(Offsets_Corrected)
        Offsets = np.flip(Offsets)
        Spectrum = np.flip(Spectrum)
        Water_Fit = np.flip(Water_Fit)
        Mt_Fit = np.flip(Mt_Fit)
        Noe_Fit = np.flip(Noe_Fit)
        Creatine_Fit = np.flip(Creatine_Fit)
        Amide_Fit = np.flip(Amide_Fit)
        Lorentzian_Difference = np.flip(Lorentzian_Difference)
        ##Residuals and RMSE dictionary
        # All_Residuals = {'Step_1':Step_1_Residuals, 'Step_2':Step_2_Residuals, 'Total': Residuals}
        # All_RMSE = {'Step_1':Step_1_RMSE, 'Step_2':Step_2_RMSE, 'Total': RMSE}
        ##Slap it in a dictionary##
        Fit_Parameters = [Fit_1, Fit_2]
        Contrasts = {'Water': 100*Fit_1[0], 'Mt': 100*Fit_1[3], 'Noe': 100*Fit_2[0], 
                     'Creatine': 100*Fit_2[3], 'Amide': 100*Fit_2[6]}
        DataDict = {'Zspec':Spectrum, 'Offsets':Offsets, 'Offsets_Corrected':Offsets_Corrected,
                    'Offsets_Interp':Offsets_Interp,'Water_Fit':Water_Fit, 'Mt_Fit':Mt_Fit, 'Noe_Fit':Noe_Fit, 
                    'Creatine_Fit':Creatine_Fit, 'Amide_Fit':Amide_Fit, 'Lorentzian_Difference':Lorentzian_Difference} 
        Fits[Segment] = {'Fit_Params':Fit_Parameters, 'Data_Dict':DataDict, 'Contrasts':Contrasts, 'Residuals':Residuals, 'RMSE':RMSE}
    end = time.time()
    total_time = end - start
    print(f"Time taken for fitting: {total_time:.2f} seconds")
    return Fits

def two_step(Offsets, Spectra):
    # n_seg = Spectra.shape[1]
    Spectrum = np.squeeze(Spectra)
    n_interp = 1000
    Offsets_Corrected = np.zeros_like(Offsets)
    Fits = {}
    try:
        # If Offsets[0] > 0, flip the Offsets and Spectrum
        if Offsets[0] > 0:
            Offsets = np.flip(Offsets)
            Spectrum = np.flip(Spectrum)
        # Fit for corrections
        Fit_1, _ = curve_fit(Step_1_Fit, Offsets, Spectrum, p0=p0_corr, bounds=(lb_corr, ub_corr))
        # Calculate water and MT fits from parameters
        Water_Fit = Lorentzian(Offsets, Fit_1[0], Fit_1[1], Fit_1[2])
        Mt_Fit = Lorentzian(Offsets, Fit_1[3], Fit_1[4], Fit_1[5])
        # B0 correction
        Correction = Offsets[np.argmax(Water_Fit + Mt_Fit)]
        Offsets_Corrected = Offsets - Correction
        # Interpolated frequency axis
        Offsets_Interp = np.linspace(Offsets_Corrected[0], Offsets_Corrected[-1], n_interp)
        # Fit again with corrected offsets
        Fit_1, _ = curve_fit(Step_1_Fit, Offsets_Corrected, Spectrum, p0=p0_1, bounds=(lb_1, ub_1))
        # Calculate water and MT fits from parameters again
        Water_Fit = Lorentzian(Offsets_Interp, Fit_1[0], Fit_1[1], Fit_1[2])
        Mt_Fit = Lorentzian(Offsets_Interp, Fit_1[3], Fit_1[4], Fit_1[5])
        # Calculate background and Lorentzian difference
        Background = Lorentzian(Offsets_Corrected, Fit_1[0], Fit_1[1], Fit_1[2]) + Lorentzian(Offsets_Corrected, Fit_1[3], Fit_1[4], Fit_1[5])
        Lorentzian_Difference = 1 - (Spectrum + Background)
        # Step 2 fit
        Fit_2, _ = curve_fit(Step_2_Fit, Offsets_Corrected, Lorentzian_Difference, p0=p0_2, bounds=(lb_2, ub_2))
        # Calculate NOE, creatine, and amide fits from parameters
        Noe_Fit = Lorentzian(Offsets_Interp, Fit_2[0], Fit_2[1], Fit_2[2])
        Creatine_Fit = Lorentzian(Offsets_Interp, Fit_2[3], Fit_2[4], Fit_2[5])
        Amide_Fit = Lorentzian(Offsets_Interp, Fit_2[6], Fit_2[7], Fit_2[8])
        # Flip to match NMR convention
        Offsets_Interp = np.flip(Offsets_Interp)
        Offsets_Corrected = np.flip(Offsets_Corrected)
        Offsets = np.flip(Offsets)
        Spectrum = np.flip(Spectrum)
        Water_Fit = np.flip(Water_Fit)
        Mt_Fit = np.flip(Mt_Fit)
        Noe_Fit = np.flip(Noe_Fit)
        Creatine_Fit = np.flip(Creatine_Fit)
        Amide_Fit = np.flip(Amide_Fit)
        Lorentzian_Difference = np.flip(Lorentzian_Difference)
        # Slap it in a dictionary
        Fit_Parameters = [Fit_1, Fit_2]
        Contrasts = {'Water': 100 * Fit_1[0], 'Mt': 100 * Fit_1[3], 'Noe': 100 * Fit_2[0], 
                     'Creatine': 100 * Fit_2[3], 'Amide': 100 * Fit_2[6]}
        DataDict = {'Zspec': Spectrum, 'Offsets': Offsets, 'Offsets_Corrected': Offsets_Corrected,
                    'Offsets_Interp': Offsets_Interp, 'Water_Fit': Water_Fit, 'Mt_Fit': Mt_Fit, 'Noe_Fit': Noe_Fit, 
                    'Creatine_Fit': Creatine_Fit, 'Amide_Fit': Amide_Fit, 'Lorentzian_Difference': Lorentzian_Difference}
    except RuntimeError:
        # Fill outputs with zeros if curve fitting fails
        Fit_Parameters = [np.zeros(6), np.zeros(9)]
        Contrasts = {'Water': 0, 'Mt': 0, 'Noe': 0, 'Creatine': 0, 'Amide': 0}
        DataDict = {'Zspec': Spectrum, 'Offsets': Offsets, 'Offsets_Corrected': Offsets_Corrected,
                    'Offsets_Interp': Offsets_Interp, 'Water_Fit': np.zeros(n_interp), 'Mt_Fit': np.zeros(n_interp), 
                    'Noe_Fit': np.zeros(n_interp), 'Creatine_Fit': np.zeros(n_interp), 
                    'Amide_Fit': np.zeros(n_interp), 'Lorentzian_Difference': np.zeros(n_interp)}
    return DataDict, Contrasts, Fit_Parameters

def One_Step_Phantom(Offsets, Spectra):
    n_seg = Spectra.shape[1]
    n_interp = 1000
    for i in range(n_seg):
        Spectrum = Spectra[:,i]
        if Offsets[0] > 0:
            Offsets = np.flip(Offsets)
            Spectrum = np.flip(Spectrum)
        ##Fit for corrections##
        Fit_Corr, _ = curve_fit(Water_Fit_Correction, Offsets, Spectrum, p0=p0_corr_ph, bounds=(lb_corr_ph, ub_corr_ph))
        ##Calculate water and MT fits from parameters##
        Water_Fit = Lorentzian(Offsets, Fit_Corr[0], Fit_Corr[1], Fit_Corr[2])
        ##B0 correction##
        Correction = Offsets[np.argmax(Water_Fit)]
        Offsets_Corrected = Offsets - Correction
        ##Set up interpolated frequency axis and fit##
        Offsets_Interp = np.linspace(Offsets_Corrected[0], Offsets_Corrected[-1], n_interp)
        Fit, _ = curve_fit(Cr_Phantom_Fit, Offsets_Corrected, Spectrum, p0=p0_ph, bounds=(lb_ph, ub_ph))
        ##Calculate fits from parameters##
        Water_Fit = Lorentzian(Offsets_Interp, Fit[0], Fit[1], Fit[2])
        Creatine_Fit = Lorentzian(Offsets_Interp, Fit[3], Fit[4], Fit[5])
        ##Flip to match NMR convention##
        Offsets_Interp = np.flip(Offsets_Interp)
        Offsets_Corrected = np.flip(Offsets_Corrected)
        Offsets = np.flip(Offsets)
        Spectrum = np.flip(Spectrum)
        Water_Fit = np.flip(Water_Fit)
        Creatine_Fit = np.flip(Creatine_Fit)
        ##Slap it in a dictionary##
        Fit_Parameters = Fit
        Contrasts = {'Water': 100*Fit[0], 'Creatine': 100*Fit[3]}
        DataDict = {'Zspec':Spectrum, 'Offsets':Offsets, 'Offsets_Corrected':Offsets_Corrected,
                    'Offsets_Interp':Offsets_Interp,'Water_Fit':Water_Fit,
                    'Creatine_Fit':Creatine_Fit}
    return DataDict, Contrasts, Fit_Parameters
        

def per_pixel(Offsets, Spectra):
    Pixelwise = []
    for Spectrum in Spectra:
        Data = two_step(Offsets, Spectrum)
        Pixelwise.append(Data)
    return Pixelwise

def plot_zspec(DataDict, Dir, Name):
    OffsetsInterp = DataDict['Offsets_Interp']
    Offsets = DataDict['Offsets_Corrected']
    Spectrum = DataDict['Zspec']
    Water_Fit = DataDict['Water_Fit']
    Mt_Fit = DataDict['Mt_Fit']
    Noe_Fit = DataDict['Noe_Fit']
    Creatine_Fit = DataDict['Creatine_Fit']
    Amide_Fit = DataDict['Amide_Fit']
    Lorentzian_Difference = DataDict['Lorentzian_Difference']
    # Plots fits
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.plot(Offsets, Spectrum, '.', markersize=15, fillstyle='none', color='black', label="Raw")  # Increased marker size
    ax.plot(OffsetsInterp, 1-Water_Fit, linewidth=4, color='#0072BD', label="Water")  # Increased linewidth
    ax.plot(OffsetsInterp, 1-Mt_Fit, linewidth=4, color='#EDB120', label="MT")  # Increased linewidth
    ax.plot(OffsetsInterp, 1-Noe_Fit, linewidth=4, color='#77AC30', label="NOE")  # Increased linewidth
    ax.plot(OffsetsInterp, 1-Amide_Fit, linewidth=4, color='#7E2F8E', label="Amide")  # Increased linewidth
    ax.plot(OffsetsInterp, 1-Creatine_Fit, linewidth=4, color='#A2142F', label="Creatine")  # Increased linewidth
    ax.plot(OffsetsInterp, 1-(Water_Fit+Mt_Fit+Noe_Fit+Creatine_Fit+Amide_Fit), linewidth=4, color='#D95319', label="Fit")  # Increased linewidth
    
    ax.legend(fontsize=24)
    ax.invert_xaxis()
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_ylim([0, 1])
    ax.set_xlabel("Offset frequency (ppm)", fontsize=32, fontname='Arial')
    ax.set_ylabel("$S/S_0$", fontsize=32, fontname='Arial')
    
    plt.grid(False)
    fig.savefig(Dir + "/" + Name + ".svg")
    fig.savefig(Dir + "/" + Name + ".tiff", dpi=300)
    plt.close(fig)
    
    # Plot Lorentzian difference
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.fill_between(Offsets, Lorentzian_Difference*100, 0, color='gray', alpha=0.5)
    ax.plot(OffsetsInterp, Noe_Fit*100, linewidth=4, color='#77AC30', label="NOE")  # Increased linewidth
    ax.plot(OffsetsInterp, Amide_Fit*100, linewidth=4, color='#7E2F8E', label="Amide")  # Increased linewidth
    ax.plot(OffsetsInterp, Creatine_Fit*100, linewidth=4, color='#A2142F', label="Creatine")  # Increased linewidth
    
    ax.legend(fontsize=24)
    ax.invert_xaxis()
    ax.set_xlabel("Offset frequency (ppm)", fontsize=32, fontname='Arial')
    ax.set_ylabel("CEST Contrast (%)", fontsize=32, fontname='Arial')
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    plt.grid(False)
    fig.savefig(Dir + "/" + Name + "_Lorentzian_Dif.svg")
    fig.savefig(Dir + "/" + Name + "_Lorentzian_Dif.tiff", dpi = 300)
    plt.close(fig)
        
    
def plot_zspec_aha(Fits, Dir, Name):
    for Segment, Fit in Fits.items():
        DataDict = Fit['Data_Dict']
        OffsetsInterp = DataDict['Offsets_Interp']
        Offsets = DataDict['Offsets_Corrected']
        Spectrum = DataDict['Zspec']
        Water_Fit = DataDict['Water_Fit']
        Mt_Fit = DataDict['Mt_Fit']
        Noe_Fit = DataDict['Noe_Fit']
        Creatine_Fit = DataDict['Creatine_Fit']
        Amide_Fit = DataDict['Amide_Fit']
        Lorentzian_Difference = DataDict['Lorentzian_Difference']
        
        # Plots fits
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.plot(Offsets, Spectrum, '.', markersize=15, fillstyle='none', color='black', label="Raw")  # Increased marker size
        ax.plot(OffsetsInterp, 1-Water_Fit, linewidth=4, color='#0072BD', label="Water")  # Increased linewidth
        ax.plot(OffsetsInterp, 1-Mt_Fit, linewidth=4, color='#EDB120', label="MT")  # Increased linewidth
        ax.plot(OffsetsInterp, 1-Noe_Fit, linewidth=4, color='#77AC30', label="NOE")  # Increased linewidth
        ax.plot(OffsetsInterp, 1-Amide_Fit, linewidth=4, color='#7E2F8E', label="Amide")  # Increased linewidth
        ax.plot(OffsetsInterp, 1-Creatine_Fit, linewidth=4, color='#A2142F', label="Creatine")  # Increased linewidth
        ax.plot(OffsetsInterp, 1-(Water_Fit+Mt_Fit+Noe_Fit+Creatine_Fit+Amide_Fit), linewidth=4, color='#D95319', label="Fit")  # Increased linewidth
        
        ax.legend(fontsize=24)
        ax.invert_xaxis()
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.set_ylim([0, 1])
        ax.set_xlabel("Offset frequency (ppm)", fontsize=32, fontname='Arial')
        ax.set_ylabel("$S/S_0$", fontsize=32, fontname='Arial')
        fig.suptitle(Segment, fontsize=32, weight='bold', fontname='Arial')
        
        plt.grid(False)
        fig.savefig(Dir + "/" + Name + "_" + Segment + ".svg")
        fig.savefig(Dir + "/" + Name + "_" + Segment + ".tiff", dpi=300)
        plt.close(fig)
        
        # Plot Lorentzian difference
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.fill_between(Offsets, Lorentzian_Difference*100, 0, color='gray', alpha=0.5)
        ax.plot(OffsetsInterp, Noe_Fit*100, linewidth=4, color='#77AC30', label="NOE")  # Increased linewidth
        ax.plot(OffsetsInterp, Amide_Fit*100, linewidth=4, color='#7E2F8E', label="Amide")  # Increased linewidth
        ax.plot(OffsetsInterp, Creatine_Fit*100, linewidth=4, color='#A2142F', label="Creatine")  # Increased linewidth
        
        ax.legend(fontsize=24)
        ax.invert_xaxis()
        ax.set_xlabel("Offset frequency (ppm)", fontsize=32, fontname='Arial')
        ax.set_ylabel("CEST Contrast (%)", fontsize=32, fontname='Arial')
        ax.tick_params(axis='both', which='major', labelsize=24)
        fig.suptitle(Segment, fontsize=32, weight='bold', fontname='Arial')
        
        plt.grid(False)
        fig.savefig(Dir + "/" + Name + "_" + Segment + "_Lorentzian_Dif.svg")
        fig.savefig(Dir + "/" + Name + "_" + Segment + "_Lorentzian_Dif.tiff", dpi = 300)
        plt.close(fig)
        
def Plot_Zspec_Phantom(DataDict, Dir, Name):
    OffsetsInterp = DataDict['Offsets_Interp']
    Offsets = DataDict['Offsets_Corrected']
    Spectrum = DataDict['Zspec']
    Water_Fit = DataDict['Water_Fit']
    Creatine_Fit = DataDict['Creatine_Fit']
    #Plots fits#
    fig, ax = plt.subplots(1,1)
    ax.plot(Offsets, Spectrum, '.', fillstyle='none', color='black', label = "Raw")
    ax.plot(OffsetsInterp, 1-Water_Fit, linewidth = 1.5, color = '#0072BD', label = "Water")
    ax.plot(OffsetsInterp, 1-Creatine_Fit, linewidth = 1.5, color = '#A2142F', label = "Creatine")
    ax.plot(OffsetsInterp, 1-(Water_Fit+Creatine_Fit), linewidth = 1.5, color = '#D95319', label = "Fit")
    ax.legend()
    ax.invert_xaxis()
    ax.set_ylim([0, 1])
    ax.set_xlabel("Offset frequency (ppm)")
    ax.set_ylabel("$S/S_0$")
    fig.savefig(Dir + "/" + Name + ".svg")
    
def riccian_noise_correction(spectrum, offsets):
    index = closest_to_zero_index = np.argmin(np.abs(offsets))
    z0 = spectrum[index]
    noise = z0
    for i in range(np.size(spectrum)):
        spectrum[i] = (spectrum[i] - noise)/(1 - noise)
    return spectrum

def wassr(offsets, spectra):
    pixelwise = []
    n_interp = 1000  # Number of points for interpolation
    for spectrum in spectra:
        if offsets[0] > 0:
            offsets = np.flip(offsets)
            spectrum = np.flip(spectrum)
        # Interpolate offsets and spectrum using cubic spline
        cubic_spline = CubicSpline(offsets, spectrum)
        offsets_interp = np.linspace(offsets[0], offsets[-1], n_interp)
        spectrum_interp = cubic_spline(offsets_interp)
        # Fit for corrections
        Fit_1, _ = curve_fit(Step_1_Fit, offsets_interp, spectrum_interp, p0=p0_corr, bounds=(lb_corr, ub_corr))
        # Calculate water and MT fits from parameters
        Water_Fit = Lorentzian(offsets_interp, Fit_1[0], Fit_1[1], Fit_1[2])
        Mt_Fit = Lorentzian(offsets_interp, Fit_1[3], Fit_1[4], Fit_1[5])
        # B0 correction: find offset corresponding to the peak
        b0_shift = offsets_interp[np.argmax(Water_Fit + Mt_Fit)]
        pixelwise.append(b0_shift)
    return pixelwise
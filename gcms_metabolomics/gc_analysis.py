#########################################################################################################
## 0. Import modules ####################################################################################
#########################################################################################################

import glob
import pandas as pd
import math
import numpy as np
import re
import os
#import pickle5 as pickle ## original: import pickle
import pickle
import time
import random
from scipy import spatial

from pyms.BillerBiemann import BillerBiemann, num_ions_threshold, rel_threshold
from pyms.Experiment import Experiment
from pyms.GCMS.IO.ANDI import ANDI_reader
from pyms.IntensityMatrix import build_intensity_matrix_i
from pyms.Noise.SavitzkyGolay import savitzky_golay
from pyms.Noise.Analysis import window_analyzer
from pyms.Peak.Function import peak_sum_area, peak_top_ion_areas
from pyms.TopHat import tophat
import matplotlib.pyplot as plt
from pyms.Display import plot_ic, plot_peaks
from pyms.DPA.PairwiseAlignment import PairwiseAlignment, align_with_tree
from pyms.DPA.Alignment import exprl2alignment
from pyms.Experiment import load_expr
from pyms.Spectrum import normalize_mass_spec
from pyms.Peak.List.IO import load_peaks, store_peaks

import pyms_nist_search
import pubchempy as pcp

import rdkit as Chem
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import BaseDocTemplate, PageTemplate, Flowable
from reportlab.platypus import Table, TableStyle, Spacer, Paragraph, Image, Frame, FrameBreak
from reportlab.lib.units import inch

from pdfrw import PdfReader, PdfDict
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl

from io import BytesIO

## convert svg to pdf, new
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

from rdkit import Chem
from rdkit.Chem import Draw
Draw.DrawingOptions.bondLineWidth = 2
Draw.DrawingOptions.atomLabelFontSize = 20
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator


def export_ic(ic, filename, outpath, mass = None):
    '''
    export ion chromatogram to csv file
    
    ic: ion chromatogram object
    name: string, output file name
    folder: string, output folder
    mass: only for EIC
    '''
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    if mass == None:
        print(f'\t export TIC of {filename} to {outpath}TIC_{filename}.csv')
    else:
        print(f'\t export EIC of {filename} at m/z {mass} to {outpath}EIC_{mass}_{filename}.csv')
        
    chromatogram = {'time': [float(i)/60 for i in ic.time_list],
                    'counts': ic.intensity_array_list}
    pd.DataFrame(chromatogram).to_csv(f'{outpath}/{filename}.csv', index = False)    
    
def process_cdf(cdf_file,name):
    '''
    input: raw cdf file directly output from chemstation
    output TIC and leco csv of the input data     
    '''

    # name = re.search(r'.*/(.*)/.CDF', cdf_file)[1]
    print(f'Processing experiment "{name}"')
    
    # read cdf file and build intensity matrix
    # time by mass matrix containing signal intensity at each time point (in seconds) and at each m/z
    data = ANDI_reader(cdf_file)
    im = build_intensity_matrix_i(data)  
    tic = im.tic
    export_ic(tic, name, './tic/')
    
    return im
    
def smooth_im(im, name, ref_rt):
    '''smooth chromatogram and baseline correction
       ref_rt: string of retention time for the structural element
    '''
    
    print(f'\t smoothing chromatogram of {name}')
    
    n_scan, n_mz = im.size
    
    # Preprocess the data (Savitzky-Golay smoothing and Tophat baseline detection)
    for ii in range(n_mz):
        ic = im.get_ic_at_index(ii)  # (extracted) ion chromatogram
        ic_smooth = savitzky_golay(ic)  # smooth (extracted) ion chromatogram
        ic_bc = tophat(ic_smooth, struct = ref_rt)  # baseline correction, providing a reference for correction
        im.set_ic_at_index(ii, ic_bc)  # update the intensity matrix
        
    return im

def peak_picking(im, msrange, ignorems = (0), minnoise = 600,
                 BBpoints = 9, BBscans = 2, threshold = 2, numsignal = 7):
    ''' peak detection, return a list of peak objects
    
        im: intensitry matrix input
        msrange: list of int, use to crop the ms of peak objects
        ignorems: tuple of int, m/z to be ignored, removed in peak objects
        BBpoints: int, number of scans to be considered as local maxima
        BBscans: int, number of scans, for which ions are combined  
        threshold: int, ion intensity threshold in % to be kept in each peak object
        numsignal: int, # of ions above noise level in a peak object to determine whether it is kept
    '''
    tic = im.tic
    
    # peak (species) detection
    # return a list of peak objects
    pl = BillerBiemann(im, points = BBpoints, scans = BBscans)

    # remove low intensity siganls (m/z, 2%) in each peak (species)
    apl = rel_threshold(pl, percent = threshold)

    # Trim the peak list by noise threshold (automatically determined by window_analyzer or provide a fix number)
    # the peak (species) must have at leat n signals (m/z) larger than the threshold
    noise_level = window_analyzer(tic)
    peak_list = num_ions_threshold(apl, n = numsignal, cutoff = max(noise_level, minnoise))
    
    print(f'\t Number of Peaks found: {len(peak_list)}')
    
    for peak in peak_list:
        peak.crop_mass(msrange[0], msrange[1])
        
        # siloxanes (common background in GC-MS)
        for mass in ignorems:
            peak.null_mass(mass)
        
        # peak area integrated over all signals (all m/z)
        peak.area = peak_sum_area(im, peak) + 1
        
        # peak area using only the higheset signal (m/z)
        peak.ion_areas = peak_top_ion_areas(im, peak)
        
    return(peak_list)

def create_exp(name, peak_list, rtrange = ['4.5m', '13m']):
    '''create and save and return experiment objects
       name: str, identifier
       peak_list: list of peak objects
       rtrange: list of str in minutes, only keep peaks within this range
    '''
    if not os.path.exists('experiment'):
        os.makedirs('experiment')
        
    expr = Experiment(name, peak_list)

    # Use the same retention time range for all experiments
    # (this can be done earlier when loading the data)
    expr.sele_rt_range(rtrange)

    # Save the experiment to disk for future use
    output_file = name + '.expr'
    expr.dump('./experiment/' + output_file)
    print(f'\t Number of Peaks saved: {len(expr.peak_list)}')
    print(f"\t Saving the result as ./experiment/{output_file}")
    
    return expr


def raw_data_process(core_id, cdf_path, ref_rt,
                     msrange = [53, 350], ignorems = (207, 281), minnoise = 500,
                     rtrange = ['5m', '12.5m'], test = False):

    file_ids_temp = []     # which LRD-CYP combination
    num_peaks = []         # total peaks detected
    num_peaks_saved = []   # num of peaks within the pre-defined retention time range
    
    
    for file in glob.glob(f'{cdf_path}*.CDF'):
        file_ids_temp.append(re.search(r'\\(.*)\.CDF', file)[1])


    # pick five files to test how many peaks detected -> if parameters need to be adjusted
    # if test == False, all files will be processed
    if test:
        file_ids = []
        for i in range(5):
            file_ids.append(random.choice(file_ids_temp))
    if not test:
        file_ids = file_ids_temp
        
    for file_id in file_ids:
        # create intensity matrix and save TIC to csv file
        file = f'{cdf_path}' + file_id + '.CDF'
        im = process_cdf(file, file_id)

        # smooth chromatogram
        sim = smooth_im(im, file_id, ref_rt)

        # peak picking and save to expr
        peak_list = peak_picking(sim, msrange, ignorems, minnoise)
        expr = create_exp(file_id, peak_list, rtrange)

        # save info to lists
        num_peaks.append(len(peak_list))
        num_peaks_saved.append(len(expr.peak_list))
    
    
    # create peak_info.csv if it's not a test run    
    if not test:
        peak_info = pd.DataFrame({'#peaks': num_peaks,
                                  '#peaks_final': num_peaks_saved},
                                  index = file_ids)

        peak_info.to_csv(core_id + '_peak_info.csv')


#########################################################################################################
## 1. Functions for initial data processing #############################################################
#########################################################################################################

def export_ic(ic, filename, outpath, mass = None):
    '''
    export ion chromatogram to csv file
    
    ic: ion chromatogram object
    name: string, output file name
    folder: string, output folder
    mass: only for EIC
    '''
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    if mass == None:
        print(f'\t export TIC of {filename} to {outpath}TIC_{filename}.csv')
    else:
        print(f'\t export EIC of {filename} at m/z {mass} to {outpath}EIC_{mass}_{filename}.csv')
        
    chromatogram = {'time': [float(i)/60 for i in ic.time_list],
                    'counts': ic.intensity_array_list}
    pd.DataFrame(chromatogram).to_csv(f'{outpath}/{filename}.csv', index = False)    
    
def process_cdf(cdf_file,name):
    '''
    input: raw cdf file directly output from chemstation
    output TIC and leco csv of the input data     
    '''

    # name = re.search(r'.*/(.*)/.CDF', cdf_file)[1]
    print(f'Processing experiment "{name}"')
    
    # read cdf file and build intensity matrix
    # time by mass matrix containing signal intensity at each time point (in seconds) and at each m/z
    data = ANDI_reader(cdf_file)
    im = build_intensity_matrix_i(data)  
    tic = im.tic
    export_ic(tic, name, './tic/')
    
    return im
    
def smooth_im(im, name, ref_rt):
    '''smooth chromatogram and baseline correction
       ref_rt: string of retention time for the structural element
    '''
    
    print(f'\t smoothing chromatogram of {name}')
    
    n_scan, n_mz = im.size
    
    # Preprocess the data (Savitzky-Golay smoothing and Tophat baseline detection)
    for ii in range(n_mz):
        ic = im.get_ic_at_index(ii)  # (extracted) ion chromatogram
        ic_smooth = savitzky_golay(ic)  # smooth (extracted) ion chromatogram
        ic_bc = tophat(ic_smooth, struct = ref_rt)  # baseline correction, providing a reference for correction
        im.set_ic_at_index(ii, ic_bc)  # update the intensity matrix
        
    return im

def peak_picking(im, msrange, ignorems = (0), minnoise = 600,
                 BBpoints = 9, BBscans = 2, threshold = 2, numsignal = 7):
    ''' peak detection, return a list of peak objects
    
        im: intensitry matrix input
        msrange: list of int, use to crop the ms of peak objects
        ignorems: tuple of int, m/z to be ignored, removed in peak objects
        BBpoints: int, number of scans to be considered as local maxima
        BBscans: int, number of scans, for which ions are combined  
        threshold: int, ion intensity threshold in % to be kept in each peak object
        numsignal: int, # of ions above noise level in a peak object to determine whether it is kept
    '''
    tic = im.tic
    
    # peak (species) detection
    # return a list of peak objects
    pl = BillerBiemann(im, points = BBpoints, scans = BBscans)

    # remove low intensity siganls (m/z, 2%) in each peak (species)
    apl = rel_threshold(pl, percent = threshold)

    # Trim the peak list by noise threshold (automatically determined by window_analyzer or provide a fix number)
    # the peak (species) must have at leat n signals (m/z) larger than the threshold
    noise_level = window_analyzer(tic)
    peak_list = num_ions_threshold(apl, n = numsignal, cutoff = max(noise_level, minnoise))
    
    print(f'\t Number of Peaks found: {len(peak_list)}')
    
    for peak in peak_list:
        peak.crop_mass(msrange[0], msrange[1])
        
        # siloxanes (common background in GC-MS)
        for mass in ignorems:
            peak.null_mass(mass)
        
        # peak area integrated over all signals (all m/z)
        peak.area = peak_sum_area(im, peak) + 1
        
        # peak area using only the higheset signal (m/z)
        peak.ion_areas = peak_top_ion_areas(im, peak)
        
    return(peak_list)

def create_exp(name, peak_list, rtrange = ['4.5m', '13m']):
    '''create and save and return experiment objects
       name: str, identifier
       peak_list: list of peak objects
       rtrange: list of str in minutes, only keep peaks within this range
    '''
    if not os.path.exists('experiment'):
        os.makedirs('experiment')
        
    expr = Experiment(name, peak_list)

    # Use the same retention time range for all experiments
    # (this can be done earlier when loading the data)
    expr.sele_rt_range(rtrange)

    # Save the experiment to disk for future use
    output_file = name + '.expr'
    expr.dump('./experiment/' + output_file)
    print(f'\t Number of Peaks saved: {len(expr.peak_list)}')
    print(f"\t Saving the result as ./experiment/{output_file}")
    
    return expr


def raw_data_process(core_id, cdf_path, ref_rt,
                     msrange = [53, 350], ignorems = (207, 281), minnoise = 500,
                     rtrange = ['5m', '12.5m'], test = False):

    file_ids_temp = []     # which LRD-CYP combination
    num_peaks = []         # total peaks detected
    num_peaks_saved = []   # num of peaks within the pre-defined retention time range
    
    
    for file in glob.glob(f'{cdf_path}*.CDF'):
        file_ids_temp.append(re.search(r'\\(.*)\.CDF', file)[1])


    # pick five files to test how many peaks detected -> if parameters need to be adjusted
    # if test == False, all files will be processed
    if test:
        file_ids = []
        for i in range(5):
            file_ids.append(random.choice(file_ids_temp))
    if not test:
        file_ids = file_ids_temp
        
    for file_id in file_ids:
        # create intensity matrix and save TIC to csv file
        file = f'{cdf_path}' + file_id + '.CDF'
        im = process_cdf(file, file_id)

        # smooth chromatogram
        sim = smooth_im(im, file_id, ref_rt)

        # peak picking and save to expr
        peak_list = peak_picking(sim, msrange, ignorems, minnoise)
        expr = create_exp(file_id, peak_list, rtrange)

        # save info to lists
        num_peaks.append(len(peak_list))
        num_peaks_saved.append(len(expr.peak_list))
    
    
    # create peak_info.csv if it's not a test run    
    if not test:
        peak_info = pd.DataFrame({'#peaks': num_peaks,
                                  '#peaks_final': num_peaks_saved},
                                  index = file_ids)

        peak_info.to_csv(core_id + '_peak_info.csv')


#########################################################################################################
## 2. Functions for peak alignment ######################################################################
#########################################################################################################

def alignment(expr_path, type = 'exp', Dw = 0.35, Gw = 0.3):
    '''
    align chromatograms to generate alignment objects
    expr_path: str, the folder containing expr files (dumped peak lists)
    type: str, exp or std, generating alignment objects for experiments of alkane standards
    Dw: float, rt modulation in seconds (allowed drift in retention time)
    Gw, float, gap penalty
    
    dump an alignment object to ./align/
    output an alignment object
    '''
    expr_list = []
               
    for expr_file in glob.glob(expr_path + '/*.expr'):
        if type == 'std' and 'std' in expr_file:
            expr = load_expr(expr_file)
            expr_list.append(expr)
        elif type == 'exp' and 'std' not in expr_file:
            expr = load_expr(expr_file)
            expr_list.append(expr)
        
    # self alignment, generating a list of alignment object
    # each experiment has its own self alignment
    F1 = exprl2alignment(expr_list)
    
    # all pairwise alignment of F1
    T1 = PairwiseAlignment(F1, Dw, Gw)
    
    # use pairwise alignment (T1) as the guide to build the overall alignment
    A1 = align_with_tree(T1)
    
    if not os.path.exists('./align'):
        os.makedirs('./align') 
    
    with open('./align/' + type + '_align.aln', 'wb') as file_out:
        pickle.dump(A1, file_out)
    
    return A1


def std_process(std_align, retention_index = 1900):
    '''input:
       std_align: alignment object for alkane std
       retention_index: int, first alkane index in the standard
       
       output: DataFrame with retention index and retention time for each alkane
    '''
    std_peak = std_align.aligned_peaks()
    retention_index = 1900
    I = []
    std_rt = []
    for peak in std_peak:
        if peak.top_ions(1) != [178]: # sometimes I forgot to use pure EtOAc to prep std, remove 
            std_rt.append(round(peak.rt/60,3))
            I.append(retention_index)
            retention_index += 100

    return pd.DataFrame({'rt': std_rt, 'I': I})

def calc_I(rt, std):
    '''calculate retention index
    
       input:
       rt: int, retention time of the peak
       std: DataFrame containing std information (rt, I)
    '''
    
    for ind, val in std.iterrows():
        if rt >= val['rt']:
            index = int(100*(rt - val['rt']) / (std.loc[ind + 1]['rt'] - val['rt']) + val['I'])
            return index
        
def weighted_cos(core_ms, peak_ms, signal_w = 0.51, mz_w = 1.1):
    '''
    compute weighted cosine similarity between core_ms and peak_ms
    
    core_ms: mass spectrum object of the LRD core
    peak_ms: mass spectrum object of the peak of interest
    signal_w: float, the weight used for ion intensities
    mz_w: float, the weight used for m/z
    '''

    ms_start = core_ms.mass_list[0]
    ms_end = core_ms.mass_list[-1]
    weights = [(i + ms_start) ** mz_w for i in range(ms_end - ms_start + 1)]
    
    core_weighted = [i**signal_w for i in core_ms.intensity_list]
    peak_weighted = [i**signal_w for i in peak_ms.intensity_list]    
    
    return round(1 - spatial.distance.cosine(core_weighted, peak_weighted, weights),3)

def peak_analysis(align, std_align, core,
                  retention_index = 1900,
                  signal_w = 0.51, mz_w = 1.1,
                  ion_threshold = 50000, similar_threshold = 0.5, occur_threshold = 15):
    '''analyze peaks identified across experiment
    
       align, std_align: alignment objects for experiments and stds
       core: str, name of the core
       retention_index: int, retention index for the first alkane in the standard
       signal_w, mz/w: float, weights used to compute similarity
       ion_threshold: float, the minimum ion counts to be considered as products
       similar_threshold: float, the minimum similarity to be consideread as products
       occur_threshold: int, the minimum occurence to be considered as common metabolites
       
       ouput csv files of rt, area, area ratio with analyzed peak info
       return DataFrame with peak objects and analyzed peak info
    '''
    if not os.path.exists('anlys'):
        os.makedirs('anlys')
    
    std = std_process(std_align, retention_index)
    std.to_csv('anlys/std.csv', index = False)   
    ###### Retrieve data from alignment object ################################################
    # DataFrame of retention time for each peak (rows) in different experiments (columns)
    df_rt = round(align.get_peak_alignment(require_all_expr = False),3)
    # DataFrame of peak area for each peak (rows) in different experiments (columns)
    df_area = round(align.get_area_alignment(require_all_expr = False))
    # DataFrame of peak object for each peak (rows) in different experiments (columns)
    df_peak = align.get_peaks_alignment(require_all_expr = False)
    
    # a serie of peak objects with average info (rt, spectrum, etc...) across experiments
    ser_peaks_ave = align.aligned_peaks()
    ###########################################################################################
    
    
    # Series: frequency of each peak across exp
    peak_count = df_rt.count(axis = 1)
    # assign names for each peak, padded with up to 3 zeros
    peak_ids = [f'{core}_{str(i + 1).zfill(3)}' for i in range(len(peak_count))]
    
    ###### Core Identification ################################################################
    # identify the index of the core from the exp without CYP
    # assuming the largest peak (greatest area) is the core
    core_index = df_area[core].idxmax()
    
    # mass spectrum of the core from the exp without CYP
    core_ms = normalize_mass_spec(df_peak.iloc[core_index][core].mass_spectrum)
    # print core information for manual confirmation
    print(f'{core}, retention time: {round(ser_peaks_ave[core_index].rt/60,3)}, id: {peak_ids[core_index]}')
    
    # a list of bool indicating whether a peak (row) is core LRD
    iscore = [False] * len(peak_count)
    iscore[core_index] = True

    
    ###########################################################################################
    
    ###### Area and Representative Spectrum ###################################################
    # DataFrame of peak area ratio for each peak (rows) in different experiments (columns)
    # peak area ratio = peak area/core peak area
    df_area_ratio = round(df_area/df_area.iloc[core_index], 4)
    
    # A series of exp_id that has the largest area for each peak
    ser_peak_maxid = df_area.idxmax(axis = 1)
    # A series of max area and max area ratio for each peak
    ser_peak_maxarea = df_area.max(axis = 1)
    ser_peak_maxratio = df_area_ratio.max(axis = 1)

    
    # Use the mass spectrum from the experiment with the greatest area as a representative
    msmax = []
    for i in range(len(peak_count)):
        ms = df_peak.iloc[i][ser_peak_maxid[i]].mass_spectrum
        msmax.append(normalize_mass_spec(ms))
    
    # A DataFrame of mass spectrum (msmax) for each peak (row)
    df_msmax = pd.DataFrame({'msmax': msmax})

    ############################################################################################
    
    ###### Average Peak Info ###################################################################
    # averaged peak list for each peak: rt, uid (unique ID), msavg, I
    rtavgs,uids,msavgs,I = [], [], [], []
    for peak in ser_peaks_ave:
        peak_rtavg = round(peak.rt/60,3)
        rtavgs.append(peak_rtavg)
        uids.append(peak.UID)
        msavgs.append(peak.mass_spectrum)
        I.append(calc_I(peak_rtavg, std))
    
    # A DataFrame of average mass spectrum (msmax) for each peak (row)
    df_msavg = pd.DataFrame({'msavg': msavgs})
    ###########################################################################################
    
    ###### Count occurence of a peak across experiments #######################################
    exp_has_peak = []
    for i in range(len(peak_count)):
        # iloc[index] returns a Series, which has no columns attribute
        # iloc[[index]] returns a DataFrame and the following code can work
        # notna() returns True or False (for NaN) for each cell
        # dot() operation with column names -> only True (not null)'s column names remain
        # return a Series object, the index remains the same (does not reset to 0)
        exp = df_rt.iloc[[i]].notna().dot(df_rt.iloc[[i]].columns + ',').str.rstrip(',')
        # take the value of the Series: exp[i] -> index does not reset to 0
        # remove LRD01_ prefix
        exp_has_peak.append(exp[i].replace(core + '_', ''))

    ###########################################################################################
    

    ###### Cosine Similarity ################################################################
    # a list of similiarity for each peak against the core
    similarity = []    
    for i in range(len(msmax)):
        ms = msmax[i]
        similarity.append(weighted_cos(core_ms, ms, signal_w = signal_w, mz_w = mz_w))

    
    # a list of confidence for each peak (the max observed is used) to be
    ## core-side: side products likely come with core
    ## common: common metabolites
    ## high: very likely to be modified LRD
    ## low: unlikely to be modified LRD
    ## very-low: singals to be considered
    ## core: core
    confidence = []
    for i in range(len(peak_count)):
        if peak_count[i] > occur_threshold:
            if similarity[i] > similar_threshold:
                confidence.append('core-side')
            else:
                confidence.append('common')
        elif similarity[i] > similar_threshold and ser_peak_maxarea[i] > ion_threshold:
            confidence.append('high')
        elif similarity[i] > similar_threshold/2 and ser_peak_maxarea[i] > ion_threshold:
            confidence.append('low')
        else:
            confidence.append('low signal')
    
    # replace the row of core with 'core' in the confidence column
    confidence[core_index] = 'core'
    
    ##########################################################################################

    ###### Concatenate all analysis ##########################################################
    anlys = pd.DataFrame({'id': peak_ids,
                         'UID': uids,
                         'RTavg': rtavgs,
                         'I': I,
                         'core': iscore,
                         'product':[False]*len(peak_count),
                         'similarity': similarity,
                         'max_area': ser_peak_maxarea,
                         'max_area_ratio': ser_peak_maxratio,
                         'confidence': confidence,
                         'counts': peak_count,
                         'exp_rep': ser_peak_maxid,
                         'exp_list': exp_has_peak,                      
                         })
    
    # if confidence is 'high', label the peak as 'True' in the 'product' column
    anlys['product'] = anlys['confidence'] == 'high'
    
    ### output anlys DataFrame 
    anlys.to_csv('anlys/anlys.csv', index = False)
    
    ### output anlys DataFrame with full rt, area, or area ratio to csv files
    pd.concat([anlys[['id']], df_rt], axis = 1).to_csv('anlys/rt.csv', index = False)
    pd.concat([anlys[['id']], df_area], axis = 1).to_csv('anlys/area.csv', index = False)
    pd.concat([anlys[['id']], df_area_ratio], axis = 1).to_csv('anlys/ratio.csv', index = False)
    
    ### save temp df_peak
    df_peak = pd.concat([anlys, df_msmax, df_msavg, df_peak], axis = 1)
    df_peak.to_pickle('anlys/peak.df')
    
    # nist_temp.df only has 'id' columns and other empty attributes as below:
    num_peaks = len(peak_ids)
    nist = pd.DataFrame({'nist_match': [{}]*num_peaks,
                        'match': ['']*num_peaks,
                        'formula': ['']*num_peaks,
                        'smiles': ['']*num_peaks,
                        'cid': ['']*num_peaks,
                        'prob': ['']*num_peaks,
                        'f_score': ['']*num_peaks,
                        'r_score': ['']*num_peaks}
                       )
    pd.concat([df_peak.iloc[:,0], nist, df_msmax], axis = 1).to_pickle('anlys/nist_temp.df')
    #########################################################################################
    
    print('done with analysis')
    return df_peak



#########################################################################################################
## 3. Functions for NIST search #########################################################################
#########################################################################################################

def nist_search(ms, search):
    '''
    ms: mass spectrum object
    search: Engine object to set up nist search

    ouput: a list of dictionary, top 3 hit
           dict: name, cid, smiles, formula, prob, f_score, r_score, compound object
    '''
    names = []
    cids = []
    smiles = []
    formulas = []
    probs = []
    f_scores = []
    r_scores = []
    compounds = []
    mss = []
    
    nist_hit = search.full_search_with_ref_data(ms)
    num_hit = len(nist_hit)
    
    ## each item in nist_hit is a tuple: (search_result, reference_data)
    ## search_result is related to the scoring
    ## reference_data is related to the molecular information
    
    # only keep at most three hits
    for i in range(min(3, num_hit)):
        hit = nist_hit[i]

        ## use the default name to search cid from pubchem first
        ## return a list of possible Compound objects
        ## sleep 0.5 seconds to avoid TimeoutError
        pubchem_hit = pcp.get_compounds(hit[0].name,'name', listkey_count=1)
        time.sleep(0.5)
        
        ## if first search fails:
        ## search on pubchem with at most five other synonyms
        if pubchem_hit == []:
            for j in range(min(5, len(hit[1].synonyms))):
                if pubchem_hit == []:
                    altname = hit[1].synonyms[j]
                    if '$:28' in altname:
                        ## inchikey sometimes is a synonum in the nist search result
                        ## remove the prefix "$:28" before search
                        pubchem_hit = pcp.get_compounds(altname.replace('$:28',''),'inchikey', listkey_count=1)
                        time.sleep(0.5)
                    else:
                        pubchem_hit = pcp.get_compounds(altname,'name', listkey_count=1)
                        time.sleep(0.5)
        
        
        formulas.append(hit[1].formula)
        probs.append(hit[0].hit_prob)
        f_scores.append(hit[0].match_factor)
        r_scores.append(hit[0].reverse_match_factor)
        mss.append(hit[1].mass_spec)
        
        ## no hit from pubchem, use the name from NIST database, leave other fields empty
        if pubchem_hit == []:           
            compounds.append(None)
            cids.append(None)
            smiles.append(None)
            names.append(hit[0].name)   
            
        ## with pubchem hit, use the first compound in the list
        else:
            compounds.append(pubchem_hit[0])             
            cids.append(int(pubchem_hit[0].cid))
            smiles.append(pubchem_hit[0].isomeric_smiles)
            if pubchem_hit[0].synonyms != []:
                names.append(pubchem_hit[0].synonyms[0])
            else:
                names.append(pubchem_hit[0].iupac_name)
            
    ## repeated search can crash the code
    
    
    return {'name': names,
            'cid': cids,
            'smiles': smiles,
            'formula': formulas,
            'prob': probs,
            'f_score': f_scores,
            'r_score': r_scores,
            'ms': mss
            }
    

def update_nist(path, search):
    '''
    Do the NIST search for each peak in the dataframe, using the mass spectrum object
    Save the temp df every 5 peaks (sometimes the code would crash)
    Can restart with the temp df
    
    input:
    path: str, where the file nist_temp.df locates
    search: Engine object to set up nist search
    
    output:
    df: dataframe (nist_temp.df) with updated search results
    
    '''
    
    df = pd.read_pickle(path)
    
    peak_ids = list(df['id'])
    count = 0
    
    for i, val in df.iterrows():
        
        ## peak not searched yet
        if val['match'] == '':
            
            # nist search using msmax, not msavg
            ms = val['msmax']            
            nist_match = nist_search(ms, search)
            
            # record the top match to the spreadsheet
            df.at[i,'match'] = nist_match['name'][0]
            df.at[i,'formula'] = nist_match['formula'][0]
            df.at[i,'smiles'] = nist_match['smiles'][0]
            df.at[i,'cid'] = nist_match['cid'][0]
            df.at[i,'prob'] = nist_match['prob'][0]
            df.at[i,'f_score'] = nist_match['f_score'][0]
            df.at[i,'r_score'] = nist_match['r_score'][0]
            # keep all the search result (a list of dictionaries)
            df.at[i,'nist_match'] = nist_match
            
            top_formula = nist_match['formula'][0]
            top_match = nist_match['name'][0]
            top_f = nist_match['f_score'][0]
            top_r = nist_match['r_score'][0]
            cid = nist_match['cid'][0]
            
            print(f'top hit of {peak_ids[i]}: {cid}, {top_formula}, {top_match}; score: {top_f}/{top_r}')
            count += 1
            
        ## peak has been searched
        else:
            print(f'{peak_ids[i]} has been analyzed')
        
        # process and pickle every 5 entries
        if count == 5:
            df.to_pickle('./anlys/nist_temp.df')
            count = 0
            
    df.to_pickle('./anlys/nist_temp.df')
    print('all peaks analyzed and saved')
           
    anlys = pd.read_csv('./anlys/anlys.csv')
    for key in ['match', 'formula', 'smiles', 'cid', 'prob', 'f_score', 'r_score']:
        anlys[key] = df[key]
        
    anlys.to_csv('./anlys/anlys_nist.csv', index = False)
    anlys.to_csv('anlys_rev.csv', index = False)
    
    # this file contains all the objects and full nist_match
    df_peak = pd.read_pickle('anlys/peak.df')
    for key in ['match', 'formula', 'smiles', 'cid', 'prob', 'f_score', 'r_score', 'nist_match']:
        df_peak[key] = df[key]
        df_peak.to_pickle('anlys/peak_nist.df')
    print('peak data updated (pickled)')


#########################################################################################################
## 4. Functions for generating a pdf report #############################################################
#########################################################################################################

def MS_compare(width, length, style, core_id, peak_id, peak_ms, compare_ms):
    '''
    width: float
    length: float
    style: int, 0: peak vs core, 1-3: NIST match
    core_id: str 
    peak_id: str
    peak_ms: DataFrame for the peak (mass, counts)
    compare_ms: DataFrame for the core or NIST match (mass, counts)
    
    '''
    plot_color = ['black','red','blue','blue','blue']
    
    fig, ax = plt.subplots(1,1, figsize = (width, length), constrained_layout=False)
    
    if style == 0:    
        ax.set_title(f'{peak_id} vs {core_id}', loc = 'left')
    else:
        ax.set_title(f'{peak_id}    NIST match {style}', loc = 'left')
    
    for i, raw_data in enumerate([peak_ms, compare_ms]):
        raw_data.counts = raw_data.counts/max(raw_data.counts)*100
        raw_data.loc[raw_data.counts < max(raw_data.counts)* 0.01, 'counts'] = 0
        
        ax.bar(raw_data.mass,
               raw_data.counts * (-2*i + 1),    # (-2*i + 1) flips the compare_ms
               color = plot_color[i*style + i], # i*style + i to select colors 
               linewidth = 0,
               width = 1
               )
        
        # only conisder ions > 10 for labeling
        top_ions = raw_data.loc[raw_data.counts > 10].sort_values('counts', ascending = False)
        
        # use prevent label overlapping with borders
        blocks = [(355, 360),
                  (0, 45)]
        # from the highest ions,
        # if the ion resides outside the blocks, label the ions
        # 'block' a range of mass that has no space for text labels
        # repeat until go through all the top_ions

        counts = 0
        add = True
        for ind, val in top_ions.iterrows():
            if counts <= 5:         # label at most 5 ions
                if any(lower <= val.mass - 10 <= upper for (lower, upper) in blocks):
                    add = False
                elif any(lower <= val.mass + 10 <= upper for (lower, upper) in blocks):
                    add = False                    
                else:
                    ax.text(val.mass,
                            (val.counts + 0.04)*(-2*i + 1) - 35*i,
                            # 0.04 to avoid ion label overlaps with the signals
                            # (-2*i + 1) to flip compare_ms
                            # - 35*i to ship ion labels in compare_ms down
                            str(int(val.mass)),
                            ha='center',
                            va='bottom',
                            color = plot_color[i*style + i]
                            )
                    counts += 1

                blocks.append((int(val.mass - 10),
                                int(val.mass + 10)))
        
    ax.axhline(color = 'black')
    ax.set_ylim(-140,140)
    ax.set_xlim(40,360)
    ax.set_xticks([50,100,150,200,250,300,350])
    ax.set_yticks([100,50,0,-50,-100])
    ax.set_yticklabels([str(abs(x)) for x in ax.get_yticks()])
    
    ax.set_xlabel('$m/z$')
    ax.set_ylabel('Ion counts (%)')
    
    fig.tight_layout()
    
    imgdata = BytesIO()
    fig.savefig(imgdata, format='pdf')
    imgdata.seek(0)
    plt.close()
    return imgdata

def GC_trace(width, length, tic, exp, RTavg, zoom = False):
    '''
    width: float
    length: float
    tic: DataFrame (time, counts)
    exp: str, the experiment to generate this TIC
    RTavg: float, retention time of the peak
    zoom: Bool, zoom in around the peak or not

    '''
    fig, ax = plt.subplots(1,1, figsize = (width, length), constrained_layout=False)
    # differece between RTavg (avg from multiple exps) and recored RT
    tic['dif'] = abs(tic.time - RTavg)
    
    x_min, x_max = 5, 14
    if zoom:
        x_min, x_max = max(5, RTavg - 1), min(14, RTavg + 1)
        exp = 'zoom in'       
        
    tic_trim = tic.loc[(tic.time >= x_min) & (tic.time <= x_max)]
    
    # the max ion (product) from the closest 6 RT to RTavg
    pdt_ion = tic_trim.sort_values(by = 'dif')[:6].counts.max()
    
    if zoom:  
        y_max = pdt_ion * 1.5        
    else:
        y_max = tic_trim.counts.max() * 1.2
        

    
    ax.plot(tic_trim.time,
            tic_trim.counts,
            color = 'black')
    
    # mark the peak with an arrow
    ax.annotate(text = '',
                xy = (RTavg, pdt_ion),
                xytext=(RTavg, pdt_ion + 0.25*y_max),
                arrowprops=dict(arrowstyle='-|>', color = 'red')
               )

    ax.set_xlim(x_min, x_max)    
    plotscale = 10**math.floor((math.log10(y_max)))
    ax.set_ylim(-y_max*0.05, y_max)
    
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(['{:,.1f}'.format(x) for x in ax.get_yticks()/plotscale])
    ax.set_ylim(-y_max*0.05, y_max)    
    
    a = math.floor((math.log10(y_max)))
    ax.set_ylabel(f'Ion counts (x $10^{a}$)')
    ax.set_xlabel('Retention time (min)')
    ax.set_title(exp, loc = 'left')
    
    fig.tight_layout()
    
    imgdata = BytesIO()
    fig.savefig(imgdata, format='pdf')
    imgdata.seek(0)
    plt.close()
    return imgdata


def CYP_heatmap(width, height, core_id, df_ratio, peak_id):
    '''
    Generate a heat map showing the ratio of peak/core for every experiment
    
    input:
    width: float
    height: float
    core_id: str, LRD core
    df_ratio: DataFrame of ratio
    peak_id: str, id of the peak
    
    input is a DataFrame, row name is ratio, and columns are experiment name
    '''
    # add nan to non-existene experiment
    
    exp_num = len(df_ratio.columns)   ### including LRD core + no CYP
         
    df_ratio = round(df_ratio.loc[peak_id]*100)
    
    # fill the array to 10x items
    # this is for plotting the heatmap
    a = np.append(df_ratio.to_numpy(), [np.nan]*(exp_num % 10))
    heatmap = pd.DataFrame(a.reshape(len(a)//10,10))
    
    # fill the array to 10x items, only the padded cells have values
    # this is for coloring the padded cells black
    a_str = np.append(df_ratio.astype(str).to_numpy(), [np.nan]*(exp_num % 10))
    heatmap_str = pd.DataFrame(a_str.reshape(len(a)//10,10))
    b = pd.DataFrame([[1]*10]*(len(a)//10))
    no_cyp = b.mask(heatmap_str.notna())
    
    fig, ax = plt.subplots(figsize=(width,height))
    
    # color the padded cells black
    sns.heatmap(no_cyp,
                cmap = 'gray',
                linewidth = 0,
                cbar = False,
                annot = False
               )
    
    # plot the heat map
    sns.heatmap(heatmap,
                cmap = 'YlGnBu',
                linecolor = 'black',
                linewidth = 0.8,
                cbar = False,
                vmin = 0,
                vmax = 20,
                xticklabels = [i for i in range(10)],
                yticklabels = [i for i in range(len(heatmap))],
                annot = True,
                fmt = 'g',
                annot_kws = {'fontsize':6, 'fontfamily':'arial', 'va':'center', 'ha':'center'},
                )
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 8, fontfamily = 'arial', rotation = 0)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 8, fontfamily = 'arial', rotation = 0)
    ax.set_title(f'{peak_id}/{core_id} by CYP (%)', fontsize = 8, fontfamily = 'arial')
    ax.set_xlabel('CYPs (00 = no CYP)')
     

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('black')
    
    fig.tight_layout()
    imgdata = BytesIO()
    plt.savefig(imgdata, format='pdf')
    imgdata.seek(0)
    plt.close()
    return imgdata
          
            
ps = ParagraphStyle('title_paragraph', fontName = 'Helvetica-Bold', fontSize = 12)
ps_red = ParagraphStyle('title_paragraph', fontName = 'Helvetica-Bold', fontSize = 12,
                        textColor= 'red')

ts_product = TableStyle([
                      ('LINEABOVE',(0,-1), (-1,-1), 0.5, colors.black),
                      ('LINEABOVE',(0,0), (-1,0), 0.75, colors.black),
                      ('LINEBELOW',(0,-1), (-1,-1), 0.75, colors.black),
                      ('SIZE', (0,0), (-1,-1), 8),
                      ('TOPPADDING', (0,0), (-1,-1), 1),
                      ('BOTTOMPADDING', (0,0), (-1,-1), 1),
                      ('ALIGN',(0,0),(-1,-1), 'CENTER'),
                      ])
ts_info = TableStyle([
                      ('LINEBEFORE',(1,0),(1,-1), 0.8, colors.black),
                      ('SIZE', (0,0), (-1,-1), 8),
                      ('TOPPADDING', (0,0), (-1,-1), 1.5),
                      ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
                      ('ALIGN',(0,0),(0,-1), 'RIGHT'),
                      ])
ts_info_color = TableStyle([
                      ('LINEBEFORE',(1,0),(1,-1), 0.5, colors.black),
                      ('SIZE', (0,0), (-1,-1), 8),
                      ('TOPPADDING', (0,0), (-1,-1), 1),
                      ('BOTTOMPADDING', (0,0), (-1,-1), 1),
                      ('ALIGN',(0,0),(0,-1), 'RIGHT'),
                       # color the score red if the score is above 800
                      ('TEXTCOLOR', (-1,-1), (-1,-1), colors.red)
                      ])

def form_xo_reader(imgdata):
    page, = PdfReader(imgdata).pages
    return pagexobj(page)


class PdfImage(Flowable):
    def __init__(self, img_data, width, height):
        self.img_width = width
        self.img_height = height
        self.img_data = img_data

    def wrap(self, width, height):
        return self.img_width, self.img_height

    def drawOn(self, canv, x, y, _sW=0):
        if _sW > 0 and hasattr(self, 'hAlign'):
            a = self.hAlign
            if a in ('CENTER', 'CENTRE', TA_CENTER):
                x += 0.5*_sW
            elif a in ('RIGHT', TA_RIGHT):
                x += _sW
            elif a not in ('LEFT', TA_LEFT):
                raise ValueError("Bad hAlign value " + str(a))
        canv.saveState()
        img = self.img_data
        if isinstance(img, PdfDict):
            xscale = self.img_width / img.BBox[2]
            yscale = self.img_height / img.BBox[3]
            canv.translate(x, y)
            canv.scale(xscale, yscale)
            canv.doForm(makerl(canv, img))
        else:
            canv.drawImage(img, x, y, self.img_width, self.img_height)
        canv.restoreState()
        
def initate_report(core, out_path):
    '''
    set the frames (the size for each block in a page) for the report 
    
    input:
    core: str, LRD core
    out_path: path for the output pdf
    
    output:
    doc: BaseDocTemplate object
    '''
    
    # set up the template for a page
    # a title frame
    # a 1:2 top frame, left for MS comparison with the core, right for RT, I, similariy, and experiments with this peak
    # a 1:1:1 bottom frame to display 3 nist search results
    # left: MS comparison with the database
    # middle: structure
    # right: name, cid, formula, mw, probability, and scores
    # inch is the unit defined in the package
    doc = BaseDocTemplate(f'{out_path}{core}_report.pdf',
                      pagesize=letter,
                      rightMargin=0.5*inch,
                      leftMargin=0.5*inch,
                      topMargin=0.5*inch,
                      bottomMargin=0.5*inch,
                      showBoundary = 0
                      )

    frameWidth = doc.width / 3
    frameHeight = doc.height * 0.96 / 5

    # title
    titleframe = Frame(x1=doc.leftMargin,
                       y1=doc.bottomMargin + doc.height * 0.96,
                       width= doc.width,
                       height= doc.height * 0.04
                      )
    # ms comparison with core
    topleftframe = Frame(x1=doc.leftMargin,
                         y1=doc.bottomMargin + frameHeight*4,
                         width=frameWidth*1.1,
                         height=frameHeight)
    # peak info
    topmidframe = Frame(x1=doc.leftMargin + frameWidth*1.1,
                        y1=doc.bottomMargin + frameHeight*4,
                        width=frameWidth*0.8,
                        height=frameHeight)
    
    # empty
    toprightframe = Frame(x1=doc.leftMargin + frameWidth*1.9,
                          y1=doc.bottomMargin + frameHeight*4,
                          width=frameWidth*1.1,
                          height=frameHeight)
    
    # gc trace
    midleftframe = Frame(x1=doc.leftMargin,
                         y1=doc.bottomMargin + frameHeight*3,
                         width=frameWidth*1.1,
                         height=frameHeight)
    
    # gc trace zoomin
    midmidframe = Frame(x1=doc.leftMargin + frameWidth*1.1,
                         y1=doc.bottomMargin + frameHeight*3,
                         width=frameWidth*0.8,
                         height=frameHeight)
    
    # CYP heatmap
    midrightframe = Frame(x1=doc.leftMargin + frameWidth * 1.9,
                          y1=doc.bottomMargin + frameHeight*3,
                          width=frameWidth*1.1,
                          height=frameHeight)
    
    # ms comparison with NIST match
    bottomleftframe = Frame(x1=doc.leftMargin,
                            y1=doc.bottomMargin,
                            width=frameWidth*1.1,
                            height=frameHeight * 3)
    # chemical structure of NIST match
    bottommidframe = Frame(x1=doc.leftMargin + frameWidth * 1.1,
                           y1=doc.bottomMargin,
                           width=frameWidth*0.8,
                           height=frameHeight * 3)
    # info of NIST match
    bottomrightframe = Frame(x1=doc.leftMargin + frameWidth * 1.9,
                             y1=doc.bottomMargin,
                             width=frameWidth*1.1,
                             height=frameHeight * 3)

    frame_list = [titleframe,
                  topleftframe,
                  topmidframe,
                  toprightframe,
                  midleftframe,
                  midmidframe,
                  midrightframe,
                  bottomleftframe,
                  bottommidframe,
                  bottomrightframe
                 ]
    
    doc.addPageTemplates([PageTemplate(id='frames',
                                       frames=frame_list)])

    return(doc)

def generate_report(peak, peak_id, core_id, mscore, int_path, tic_path):
    '''
    
    input:
    peak: dataframe, all information of a peak
    peak_id: str, id of the peak
    core_id: str, LRD core
    mscore: MSHtoL object, mass spectrum for the core (flipped)
    int_path: str, the folder with all intermediate files
    tic_path: str, the folder with all tic files
    '''
    
    print('generating report of ' + peak_id)
    
    df_ratio = pd.read_csv(f'{int_path}ratio.csv', index_col = 'id')
    
    ### title ###############################################################
    page_title = Paragraph(peak_id, style = ps)
    titleframe_story = [page_title,
                        FrameBreak()] # jump to the next frame
    #########################################################################
    
   
    ### topleft frame: Peak vs Core mass spectrua comparison ################
    # peak MS
    peak_ms = pd.DataFrame({'mass': peak.msmax.mass_list,
                            'counts': peak.msmax.intensity_list})
    imgdata = MS_compare(width = 2.5,
                         length = 1.75,
                         style = 0,
                         core_id = core_id,
                         peak_id = peak_id,
                         peak_ms = peak_ms,
                         compare_ms = mscore
                         )

    image = form_xo_reader(imgdata)   # cannot use Pdfreader directly
    topleftframe_story = [PdfImage(image, width = 2.5*inch, height = 1.75*inch),
                           FrameBreak()]
    #########################################################################
    
    ### topmid frame: UID, RTavg, I, similarity info ########################
    
    ### create a table (a list of list)
    pdt_table = []
    pdt_table.append(['UID', peak.UID])
    pdt_table.append(['RTavg', str(peak.RTavg) + ' min'])
    try:
        pdt_table.append(['I', int(peak.I)])
    except ValueError:
        pdt_table.append(['I', ''])
    pdt_table.append(['similarity', peak.similarity])

    
    tbl_product = Table(pdt_table, style = ts_info, hAlign = 'LEFT')
    
    topmidframe_story = [Spacer(1,0.5*inch), tbl_product, FrameBreak()]   
    ######################################################################### 
    
    ### topright frame: automatic assignment ################################
    
    assignment = Paragraph(f'auto assignment: {peak.confidence}', style = ps)

    if peak['product']:
        product = Paragraph('Product', style = ps_red)
    elif peak.core:
        product = Paragraph('Core', style = ps_red)
    else:
        product = Paragraph('', style = ps_red)
    
    toprightframe_story = [Spacer(1,0.5*inch),
                           assignment,
                           Spacer(1,0.3*inch),
                           product,
                           FrameBreak()]
    ######################################################################### 
   
    ### midleft, midmid frame: GC-MS traces, full and zoom ##################

    if peak.core:
        tic_file = pd.read_csv(f'{tic_path}{core_id}.csv')
        exp_rep = core_id
    else:
        tic_file = pd.read_csv(f'{tic_path}{peak.exp_rep}.csv')
        exp_rep = peak.exp_rep
            
    imgdata = GC_trace(width = 2.5,
                       length = 1.75,
                       tic = tic_file,
                       exp = exp_rep,
                       RTavg = peak.RTavg,
                       zoom = False)
    image = form_xo_reader(imgdata)   # cannot use Pdfreader directly
    img_gc_trace = PdfImage(image, width = 2.5*inch, height = 1.75*inch)
    
    midleftframe_story = [img_gc_trace,
                         FrameBreak()]

    imgdata = GC_trace(width = 1.8,
                       length = 1.75,
                       tic = tic_file,
                       exp = exp_rep,
                       RTavg = peak.RTavg,
                       zoom = True)
    image = form_xo_reader(imgdata)   # cannot use Pdfreader directly
    img_gc_trace = PdfImage(image, width = 1.8*inch, height = 1.75*inch)
    
    midmidframe_story = [img_gc_trace,
                         FrameBreak()]    
    
    #########################################################################
        
    ### midright frame: heatmap of peak detection ###########################
    
    imgdata = CYP_heatmap(width = 2.5, height = 1.75,
                          core_id = core_id,
                          df_ratio = df_ratio,
                          peak_id = peak_id)
    image = form_xo_reader(imgdata)
    img_heatmap = PdfImage(image, width=2.5*inch, height=1.75*inch)
    midrightframe_story = [img_heatmap, FrameBreak()]
    
    #########################################################################
    
   
    ### bottom frames #######################################################
    bottomleftframe_story = []
    bottommidframe_story = [Spacer(1,0.05*inch)]
    bottomrightframe_story = [Spacer(1,0.2*inch)]

    num_matches = len(peak.nist_match['name'])
    for j in range(num_matches):

        ### bottomleftframe #############################################
        match = normalize_mass_spec(peak.nist_match['ms'][j])
        match_ms = pd.DataFrame({'mass': match.mass_list, 'counts': match.intensity_list})
        # with a label matchx on the bottom right corner
        imgdata = MS_compare(width = 2.5,
                             length = 1.75,
                             style = j + 1,
                             core_id = core_id,
                             peak_id = peak_id,
                             peak_ms = peak_ms,
                             compare_ms = match_ms
                             )
        
        image = form_xo_reader(imgdata)   # cannot use Pdfreader directly
        img_core_match = PdfImage(image, width = 2.5*inch, height = 1.75*inch)
        bottomleftframe_story.append(img_core_match)
        bottomleftframe_story.append(Spacer(1,0.1*inch))

        ### bottommidframe ##############################################
        smiles = peak.nist_match['smiles'][j]
        try:
            m = Chem.MolFromSmiles(smiles)
        except TypeError:
            m = Chem.MolFromSmiles('')
        Draw.MolToFile(m, str(j) + '.svg', size = (360, 300))
        drawing = svg2rlg(str(j) + '.svg')
        renderPDF.drawToFile(drawing, str(j) + '.pdf')
        # this intermediate pdf needs to be removed later
        
        image = form_xo_reader(str(j) + '.pdf')   # cannot use Pdfreader directly
        img_nist_match = PdfImage(image, width = 1.8*inch, height = 1.5*inch)

        bottommidframe_story.append(img_nist_match)
        if j != 2:
            bottommidframe_story.append(Spacer(1,0.35*inch)) 

        ### bottomrightframe ############################################
        info_table = []
        info_table.append(['name', peak.nist_match['name'][j]])
        info_table.append(['cid', peak.nist_match['cid'][j]])
        info_table.append(['formula', peak.nist_match['formula'][j]])
        info_table.append(['mw', round(ExactMolWt(m),1)])
        info_table.append(['prob', str(peak.nist_match['prob'][j])])
        info_table.append(['score', str(peak.nist_match['f_score'][j]) + '/' + str(peak.nist_match['r_score'][j])])

        if peak.nist_match['f_score'][j] > 800 and peak.nist_match['r_score'][j] > 800:
            tbl_info = Table(info_table, style = ts_info_color, hAlign = 'LEFT')
        else:
            tbl_info = Table(info_table, style = ts_info, hAlign = 'LEFT')                

        bottomrightframe_story.append(tbl_info)
        if j != 2:
            bottomrightframe_story.append(Spacer(1,0.6*inch))     

    bottommidframe_story.append(FrameBreak())
    bottomrightframe_story.append(FrameBreak())
    bottomleftframe_story.append(FrameBreak())
    #########################################################################
    
    story = titleframe_story + topleftframe_story + topmidframe_story + toprightframe_story
    story = story + midleftframe_story + midmidframe_story + midrightframe_story
    story = story + bottomleftframe_story + bottommidframe_story + bottomrightframe_story
    
    print('\t...done')
    
    return(story)



def build_pdf_report(core_id, int_path, tic_path, out_path):
    '''
    generate human-readable pdf report for the screening results
    
    input:
    core_id: str, LRD core
    int_path: str, path for the peak_nist.df
    tic_path: str, path for all tic files
    out_path: str, path for the output
    
    output:
    a pdf file for the screening results
    '''
    
    df_peak = pd.read_pickle(f'{int_path}peak_nist.df')

    # core MS
    core = df_peak.loc[df_peak['core'] == True].iloc[0]['msmax']
    core_ms = pd.DataFrame({'mass': core.mass_list, 'counts': core.intensity_list})
    
    doc = initate_report(core_id, out_path = out_path)
    
    story = []
    
    for i, peak in df_peak.iterrows():
        if i == 0:
            print(f'start generating report: {out_path}{core_id}_report.pdf')
            print(f'total {len(df_peak)} peaks')

        story = story + generate_report(peak = peak,
                                        peak_id = peak.id,
                                        core_id = core_id,
                                        mscore = core_ms,
                                        int_path = int_path,
                                        tic_path = tic_path)

    doc.build(story)


    # remove the intermediary structure pdf file
    for j in range(3):
        if os.path.exists(str(j) + '.pdf'): os.remove(str(j) + '.pdf')
        if os.path.exists(str(j) + '.svg'): os.remove(str(j) + '.svg')
        
    print('\nfinishing generating the report')
            

#########################################################################################################
## 5. Functions for Finalizing product selection ########################################################
#########################################################################################################

def plot_heatmap(core_id, heatmap, info, out_path, annot = False):
    '''
    core_id: str
    heatmap: DataFrame
    info: DataFrame
    outpath: str
    
    output:
    a pdf file of product distribution heatmap
    '''

    rt = info.RTavg
    sim = info.similarity

    # determine the figure size based on the numbers of columns and rows
    w = 0.8658 + 0.17*len(heatmap.columns) + 1.9
    h = 0.5941 + 0.17*len(heatmap)
    fig, ax = plt.subplots(figsize=(w,h))

    sns.heatmap(round(heatmap*100),
                cmap = 'YlGnBu',
                linecolor = 'lightgray',
                cbar = False,
                fmt = 'g',
                vmin = 0,
                vmax = 20,
                xticklabels = heatmap.columns,
                yticklabels = info.index,
                annot = annot, # display % or not
                square = True, # force each cell to be square
                linewidths = 0.8,
                )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(ax.get_xmajorticklabels(),rotation = 270)
    ax.set_yticklabels(ax.get_ymajorticklabels(), va = 'center')
    
    for i in range(len(info.index)):
        x_offset = len(heatmap.columns)
        ax.text(x_offset + 0.25, i+0.5, info.index[i], va = 'center', ha = 'left')
        ax.text(x_offset + 8, i+0.5, "{:.3f}".format(rt[i]), va = 'center', ha = 'right')
        ax.text(x_offset + 11, i+0.5, "{:.3f}".format(sim[i]), va = 'center', ha = 'center')
        
    ax.text(x_offset + 2, -0.35, 'ID', va = 'center', ha = 'center')
    ax.text(x_offset + 7.25, -0.35, 'RT (min)', va = 'center', ha = 'center')
    ax.text(x_offset + 11, -0.35, 'similarity', va = 'center', ha = 'center')

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')

    fig.savefig(f'{out_path}{core_id}_heatmap.pdf', dpi = 300, bbox_inches = 'tight')
    print(f'...plotting finished, the heatmap saved to {out_path}{core_id}_heatmap.pdf')
    

def plot_pdt_ms(core_id, core_ms, df_peak, out_path):
    '''
    core_id: str
    core_ms: DataFrame, mass spec of the core
    df_peak: DataFrame
    out_path: str
    
    output:
    a pdf file with all product vs core mass spectra
    '''
    plot_color = ['black','red']
    fig, axs = plt.subplots(math.ceil(len(df_peak)/3), 3,
                            figsize = (6.7, 1.6*math.ceil(len(df_peak)/3)),
                            constrained_layout=False,
                            )
    j = 0
    for peak_id, val in df_peak.iterrows():
        
        ax = axs.flat[j]
        peak_ms = pd.DataFrame({'mass': val.msmax.mass_list, 'counts': val.msmax.intensity_list})
        
        print(f'plotting {peak_id}...')
        for i, raw_data in enumerate([peak_ms, core_ms]):
            raw_data.counts = raw_data.counts/max(raw_data.counts)*100
            raw_data.loc[raw_data.counts < max(raw_data.counts)* 0.01, 'counts'] = 0

            ax.bar(raw_data.mass,
                        raw_data.counts * (-2*i + 1),    # (-2*i + 1) flips the compare_ms
                        color = plot_color[i], # i*style + i to select colors 
                        linewidth = 0,
                        width = 1
                        )

            # only conisder ions > 10 for labeling
            top_ions = raw_data.loc[raw_data.counts > 10].sort_values('counts', ascending = False)

            # use prevent label overlapping with borders
            blocks = [(355, 360),
                      (0, 45)]
            # from the highest ions,
            # if the ion resides outside the blocks, label the ions
            # 'block' a range of mass that has no space for text labels
            # repeat until go through all the top_ions

            counts = 0
            add = True
            for ind,ion in top_ions.iterrows():
                if counts <= 5:         # label at most 5 ions
                    if any(lower <= ion.mass - 10 <= upper for (lower, upper) in blocks):
                        add = False
                    elif any(lower <= ion.mass + 10 <= upper for (lower, upper) in blocks):
                        add = False                    
                    else:
                        ax.text(ion.mass,
                                (ion.counts + 0.04)*(-2*i + 1) - 35*i,
                                # 0.04 to avoid ion label overlaps with the signals
                                # (-2*i + 1) to flip compare_ms
                                # - 35*i to ship ion labels in compare_ms down
                                str(int(ion.mass)),
                                ha='center',
                                va='bottom',
                                color = plot_color[i]
                                )
                        counts += 1

                    blocks.append((int(ion.mass - 10),
                                    int(ion.mass + 10)))

        ax.axhline(color = 'black')
        ax.set_ylim(-140,140)
        ax.set_xlim(40,360)
        ax.set_xticks([50,100,150,200,250,300,350])
        ax.set_yticks([100,50,0,-50,-100])
        ax.set_yticklabels([])
        if j % 3 == 0:
            ax.set_yticklabels([str(abs(x)) for x in ax.get_yticks()])
            
        ax.set_title(peak_id, loc = 'left')
        
        j += 1
        
    fig.supxlabel('$m/z$')
    fig.supylabel('Ion counts (%)')
    
    # hide no mass spec subplots
    if len(df_peak)%3 == 1:
        axs.flat[-1].axis('off')
        axs.flat[-2].axis('off')
    elif len(df_peak)%3 == 2:
        axs.flat[-1].axis('off')

    fig.savefig(f'{out_path}{core_id}_product_ms.pdf', dpi = 300, bbox_inches = 'tight')
    print(f'...plotting finished, mass spectra saved to {out_path}{core_id}_product_ms.pdf')
    

    

def finalize_report(core_id, int_path, rev_path, out_path, reorder = False):
    '''
    core_id: str
    int_path: str, the path with all analyzed files
    rev_path: str, the path has 'anlys_rev.csv'
    out_path: str, the path for output files
    reorder: bool, rename product id or not
    
    
    '''
    df_peak = pd.read_pickle(f'{int_path}peak_nist.df')
    final = pd.read_csv(f'{rev_path}anlys_rev.csv')
    final_ratio = pd.read_csv(f'{int_path}ratio.csv', index_col = 'id')
    
    
    core = df_peak.loc[df_peak['core'] == True].iloc[0]['msmax']
    core_ms = pd.DataFrame({'mass': core.mass_list, 'counts': core.intensity_list})
    
    # select product labeled TRUE (considered as product)
    final = final.loc[final['product']].sort_values(by = 'id')    
    
    final_ratio.columns = final_ratio.columns.str.replace(fr'{core_id}_','')
    final_ratio = final_ratio.loc[final.id]
    final_ratio = final_ratio.dropna(axis=1, how='all')
    
    final = final.set_index('id')
    df_peak = df_peak.set_index('id')
    df_peak = df_peak.loc[final.index]
    
    
    if reorder:
        new_index = [f'{core_id}_{str(i+1).zfill(2)}' for i in range(len(final))]
        final.index = new_index
        final_ratio.index = new_index
        df_peak.index = new_index

    final_ratio.to_csv(f'{out_path}{core_id}_heatmap.csv')
    print(f'write {out_path}{core_id}_heatmap.csv')
    
    final[['RTavg', 'similarity']].to_csv(f'{out_path}{core_id}_pdt_info.csv')
    print(f'write {out_path}{core_id}_pdt_info.csv')

    print('start plotting the heatmap')
    plot_heatmap(core_id = core_id,
                 heatmap = final_ratio,
                 info = final,
                 out_path = out_path,
                 annot = False)
    
    print('start plotting product mass spectra')
    plot_pdt_ms(core_id = core_id,
                core_ms = core_ms,
                df_peak = df_peak,
                out_path = out_path
               )
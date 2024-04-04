# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:57:21 2024

@author: Moayad
"""

#%% Load dependencies

# define the path to input file
gitDir = 'C:/Users/Moayad/Documents/GitHub/stickModel'
outDir = f'{gitDir}/test/nltha'

# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, f'{gitDir}')

import pandas as pd
import numpy as np
import os
import time
from stickModel import stickModel
import matplotlib.pyplot as plt

import openseespy.opensees as ops
from utils import *
import math 
import eqsig


# Initialise the time
start_time = time.time()

#%% User input

# read the input file
input_data = pd.read_excel(f'{gitDir}/test/nltha/input_file.xlsx', sheet_name='Basic')

# read the input parameters 
nst = int(input_data['Value'][0]) # number of storeys
flh = [float (i) for i in input_data['Value'][1].split()]
flm = [float (i) for i in input_data['Value'][2].split()]
fll = [float (i) for i in input_data['Value'][3].split()]
fla = [float (i) for i in input_data['Value'][4].split()]
flm = [1.0, 1.0, 1.0]

# test size
if len(flh)!=nst or len(flm)!=nst or len(fll)!=nst or len(fla)!=nst:
    raise ValueError('Number of entries exceed the number of storeys')

# define the structural typology
buildingClass = 'CR_LFINF+CDH+DUL_H1'


##############################################################################################################
# #%% Test NLTHA (single record)
##############################################################################################################
# # time-history parameters
# fnames = [f'{gitDir}/test/nltha/single_record/record.txt',f'{gitDir}/test/nltha/single_record/record.txt']
# dt_gm = 0.01         # acceleration time-history time step
# t_max = 300.00       # maximum duration
# dt_ansys = 0.05      # analysis time step
# sf = 1.00            # scaling factor
# collLimit = 1.5      # collapse limit

# # compile the model
# ops.wipe()
# model = stickModel(nst,flh,flm,fll,fla,structTypo,f'{gitDir}/raw/ip')
# model.mdof_initialise()
# model.mdof_nodes()
# model.mdof_fixity()
# model.mdof_loads()
# model.mdof_material()
# model.plot_model()
# model.do_gravity_analysis()
# model.do_modal_analysis()
# control_nodes, coll_index, peak_drift, peak_accel, max_peak_drift, max_peak_drift_dir, max_peak_drift_loc, max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp = model.do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, collLimit, outDir)


# model.do_fragility_analysis([0.5],max_peak_drift,[0.5, 1.0, 2.0])


# # plot acceleration profiles (in x)
# plt.plot(peak_accel[:,0], control_nodes, color = 'blue',linestyle='solid')
# plt.xlabel("Peak Floor Acceleration, PFA [g]")
# plt.ylabel("Floor No.")
# plt.grid(visible=True, which='major')
# plt.grid(visible=True, which='minor')
# plt.xlim([0,1])
# plt.ylim([0,np.max(control_nodes)])
# plt.title('Acceleration Profile in X-Direction')
# plt.show()

# # plot acceleration profiles (in y)
# plt.plot(peak_accel[:,0], control_nodes, color = 'blue',linestyle='solid')
# plt.xlabel("Peak Floor Acceleration, PFA [g]")
# plt.ylabel("Floor No.")
# plt.grid(visible=True, which='major')
# plt.grid(visible=True, which='minor')
# plt.xlim([0,1])
# plt.ylim([0,np.max(control_nodes)])
# plt.title('Acceleration Profile in Y-Direction')
# plt.show()


# # plot drift profiles (in x)
# bplot_driftX, bplot_nodes = duplicate_for_drift(peak_drift[:,0], control_nodes)
# plt.plot(bplot_driftX, bplot_nodes, color = 'blue',linestyle='solid')
# plt.xlabel("Peak Storey Drift, PSD [%]")
# plt.ylabel("Floor No.")
# plt.grid(visible=True, which='major')
# plt.grid(visible=True, which='minor')
# plt.title('Storey Drift Profile in X-Direction')
# plt.show()

# # plot drift profiles (in y)
# bplot_driftY, bplot_nodes = duplicate_for_drift(peak_drift[:,1], control_nodes)
# plt.plot(bplot_driftY, bplot_nodes, color = 'blue',linestyle='solid')
# plt.xlabel("Peak Storey Drift, PSD [%]")
# plt.ylabel("Floor No.")
# plt.grid(visible=True, which='major')
# plt.grid(visible=True, which='minor')
# plt.title('Storey Drift Profile in Y-Direction')
# plt.show()


##############################################################################################################
#%% Test NLTHA (multiple records)
##############################################################################################################
# time-history parameters (use os.listdir to get the names of the gmr files)

gmrs = os.listdir(f'{gitDir}/test/nltha/multiple_records/gmrs')

# Initialise storage lists
coll_index_lst = []
peak_drift_lst = []
peak_accel_lst = []
max_peak_drift_lst = []
max_peak_drift_dir_list = []
max_peak_drift_loc_list = []
max_peak_accel_lst = []
max_peak_accel_dir_list = []
max_peak_accel_loc_list = []
pga = []; sa = []

for i in range(len(gmrs)):
    
    # Compile the model
    ops.wipe()
    model = stickModel(nst,flh,flm,buildingClass,f'{gitDir}/raw/ip')
    model.mdof_initialise()
    model.mdof_nodes()
    model.mdof_fixity()
    model.mdof_loads()
    model.mdof_material()
    if i==0:
        model.plot_model()
    else:
        pass
    model.do_gravity_analysis()
    T = model.do_modal_analysis()
    
    # Define ground motion objects
    fnames = [f'{gitDir}/test/nltha/multiple_records/gmrs/{gmrs[i]}',f'{gitDir}/test/nltha/multiple_records/gmrs/{gmrs[i]}']
    fdts = f'{gitDir}/test/nltha/multiple_records/dts/dts_{i}.csv'
    
    dt_gm = pd.read_csv(fdts)[pd.read_csv(fdts).columns[0]].loc[0]
    t_max = pd.read_csv(fdts)[pd.read_csv(fdts).columns[0]].iloc[-1]

    # Get intensity measure level
    
    im = intensityMeasure(pd.read_csv(fnames[0]).to_numpy().flatten(),dt_gm)
    
    # get the response spectrum
    pga.append(im.get_sa(0.0))
    sa.append(im.get_sa(T[0]))
    
    # Define analysis params and do NLTHA
    dt_ansys = dt_gm
    sf = 1.0
    collLimit = 5.0
    control_nodes, coll_index, peak_drift, peak_accel, max_peak_drift, max_peak_drift_dir, max_peak_drift_loc, max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp = model.do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, collLimit, outDir)
    
    # Store the analysis arrays
    coll_index_list.append(coll_index)
    peak_drift_list.append(peak_drift)
    peak_accel_list.append(peak_accel)
    peak_disp_list.append(peak_disp)
    max_peak_drift_list.append(max_peak_drift)
    max_peak_drift_dir_list.append(max_peak_drift_dir)
    max_peak_drift_loc_list.append(max_peak_drift_loc)
    max_peak_accel_list.append(max_peak_accel)
    max_peak_accel_dir_list.append(max_peak_accel_dir)
    max_peak_accel_loc_list.append(max_peak_accel_loc)

    print('================================================================')
    print('============== ANALYSIS COMPLETED: {:d} out {:d} =================='.format(i, len(gmrs)))
    print('================================================================')


#%% Create output dict

ansys_dict = {}
labels = ['control_nodes', 'coll_index_lst','peak_drift_lst','peak_accel_lst','max_peak_drift_lst',
          'max_peak_drift_dir_list', 'max_peak_drift_loc_list','max_peak_accel_lst',
          'max_peak_accel_dir_list','max_peak_accel_loc_list','peak_disp_lst']

for i, label in enumerate(labels):
    ansys_dict[label] = vars()[f'{label}']

#%% Plot Cloud Analysis Results

import matplotlib.cbook as cbook
import matplotlib.image as image

# import the GEM watermark
with cbook.get_sample_data('C:/Users/Moayad/Documents/GitHub/stickModel/gem_logo2.png') as file:
    im = image.imread(file)
    
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

# Peak Ground Acceleration
imvector1, edpvector1, coeffs1, beta1 = model.cloud_analysis(pga, max_peak_drift_lst)

ax1.scatter(max_peak_drift_lst, pga)
ax1.plot(edpvector1, imvector1, linewidth=5.0, linestyle = '-', color = 'black')
ax1.set_xlabel(r'Peak Storey Drift, $\theta$ [%]')
ax1.set_ylabel(r'Peak Ground Acceleration, PGA [g]')
ax1.grid(visible=True, which='major')
ax1.grid(visible=True, which='minor')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim([5e-3, 1e1])
ax1.set_xlim([1e-1, 1e1])
plt.figimage(im, 60, 310, zorder=1, alpha=.7)

# Spectral Acceleration
imvector2, edpvector2, coeffs2, beta2 = model.cloud_analysis(sa, max_peak_drift_lst)

ax2.scatter(max_peak_drift_lst, sa)
ax2.plot(edpvector2, imvector2, linewidth=5.0, linestyle = '-', color = 'black')
ax2.set_xlabel(r'Peak Storey Drift, $\theta$ [%]')
ax2.set_ylabel(r'Spectral Acceleration, $Sa(T_{1})$ [g]')
ax2.grid(visible=True, which='major')
ax2.grid(visible=True, which='minor')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim([5e-3, 1e1])
ax2.set_xlim([1e-1, 1e1])

#%% Plot Fragilities

damageThresholds = [0.25, 0.5, 1.0, 5.0, 10.0]
im_range = np.linspace(0.0, 1.0, 500)

# initialise the figure
colors = ['blue','green','yellow','red','black']
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
# import the GEM watermark
with cbook.get_sample_data('C:/Users/Moayad/Documents/GitHub/stickModel/gem_logo2.png') as file:
    im = image.imread(file)


# Peak Ground Acceleration
thetaList = []
betaList = []
poeList = []
for i in range(len(damageThresholds)):
    theta, poe = model.get_probability_of_damage(coeffs1, beta1, damageThresholds[i])    
    thetaList.append(theta)
    betaList.append(beta1)
    poeList.append(poe)
    ax1.plot(im_range, poeList[i], linewidth=2.5, linestyle = '-', color = colors[i])
ax1.set_xlabel(r'Peak Ground Acceleration, PGA [g]')
ax1.set_ylabel(r'Probability of Damage')
ax1.grid(visible=True, which='major')
ax1.grid(visible=True, which='minor')
ax1.set_ylim([0, 1])
ax1.set_xlim([0, 1])
plt.figimage(im, 60, 310, zorder=1, alpha=.7)

# Spectral Acceleration
thetaList = []
betaList = []
poeList = []
for i in range(len(damageThresholds)):
    theta, poe = model.get_probability_of_damage(coeffs2, beta2, damageThresholds[i])    
    thetaList.append(theta)
    betaList.append(beta2)
    poeList.append(poe)
    ax2.plot(im_range, poeList[i], linewidth=2.5, linestyle = '-', color = colors[i])
ax2.set_xlabel(r'Spectral Acceleration, $Sa(T_{1})$ [g]')
ax2.set_ylabel(r'Probability of Damage')
ax2.grid(visible=True, which='major')
ax2.grid(visible=True, which='minor')
ax2.set_ylim([0, 1])
ax2.set_xlim([0, 1])
plt.show()


#%% Plot Demand Profiles

nansysShow = 500 # number of analysis to show

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

nst = peak_drift_lst[0].shape[0]
nfl = nst+1

for i in range(nansysShow):
    
    x,y = duplicate_for_drift(peak_drift_lst[i][:,0],control_nodes)
    
    ax1.plot(x, y, linewidth=2.5, linestyle = 'solid', color = 'gray', alpha = 0.7)
    ax1.set_xlabel(r'Peak Storey Drift, $\theta$ [%]')
    ax1.set_ylabel('Floor No.')
    ax1.grid(visible=True, which='major')
    ax1.grid(visible=True, which='minor')
    ax1.set_yticks(np.linspace(0,nst,nst+1))
    xticks = np.linspace(0,5,11)
    ax1.set_xticks(xticks, labels=xticks, minor=True)
    ax1.set_xlim([0, 5.0])
    plt.figimage(im, 60, 310, zorder=1, alpha=.7)

    ax2.plot([float(x)*9.81 for x in peak_accel_lst[i][:,0]], control_nodes, linewidth=2.5, linestyle = 'solid', color = 'gray', alpha=0.7)
    ax2.set_xlabel(r'Peak Floor Acceleration, $a_{max}$ [g]')
    ax2.set_ylabel('Floor No.')
    ax2.grid(visible=True, which='major')
    ax2.grid(visible=True, which='minor')
    ax2.set_yticks(np.linspace(0,nst,nst+1))
    xticks = np.linspace(0,5,11)
    ax2.set_xticks(xticks, labels=xticks, minor=True)
    ax2.set_xlim([0, 5.0])
    


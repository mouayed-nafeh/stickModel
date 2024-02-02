# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:57:21 2024

@author: Moayad
"""

# load dependencies
import pandas as pd
import numpy as np
from stickModel import stickModel
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from utils import *

# script to test modules 

#%% test the input data compiler

# define the path to input file
path_to_input_file = 'C:/Users/Moayad/Desktop/python/stick-model/input_file.xlsx'

# read the input file
input_data = pd.read_excel(path_to_input_file, sheet_name='Basic')

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


#structTypo = 'CR_LFINF+CDH+DUL_H1'
structTypo = 'S_LFM+CDM+DUM_H6'
#structTypo = 'S_LWAL+CDH+DUH_H11'

gitDir = 'C:/Users/Moayad/Documents/GitHub/stickModel/raw/ip'

D = list(pd.read_csv(f'{gitDir}/{structTypo}.csv').iloc[:,0])
F = list(pd.read_csv(f'{gitDir}/{structTypo}.csv').iloc[:,1])



#%% Test static pushover analysis

ref_disp = 0.5
disp_scale_factor = 10
push_dir = 1

model = stickModel(nst,flh,flm,fll,fla,structTypo,gitDir)
model.mdof_initialise()
model.mdof_nodes()
model.mdof_fixity()
model.mdof_loads()
model.mdof_material()
model.do_gravity_analysis()
ops.wipeAnalysis()
model.do_modal_analysis()
ops.wipeAnalysis()

spo_disps, spo_rxn = model.do_spo_analysis(ref_disp, disp_scale_factor, push_dir, pflag=False)
ops.wipe()
 

#%% Test cyclic pushover analysis

ref_disp = 0.005
numCycles = 1000
push_dir = 2
dispIncr = 200
mu_list = [10]

for i in range(len(mu_list)):

    model = stickModel(nst,flh,flm,fll,fla,structTypo,gitDir)
    model.mdof_initialise()
    model.mdof_nodes()
    model.mdof_fixity()
    model.mdof_loads()
    model.mdof_material()
    model.plot_model()
    model.do_gravity_analysis()
    ops.wipeAnalysis()
    model.do_modal_analysis()
    ops.wipeAnalysis()
    
    cpo_disps, cpo_rxn = model.do_cpo_analysis(ref_disp, mu_list[i], numCycles, push_dir, dispIncr, pflag = False)
    
    plt.plot(spo_disps[:,nst-1], spo_rxn, color = 'black',linestyle='solid')
    plt.plot(-spo_disps[:,nst-1], -spo_rxn, color = 'black',linestyle='solid')
    plt.plot(cpo_disps[:,nst-1], cpo_rxn, color = 'blue',linestyle='dashed')
    plt.xlabel("Top Displacement, $\delta$ [m]")
    plt.ylabel("Base Shear, V [kN]")
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor')
    plt.xlim([-1,1])


#%% Test NLTHA

dt_gm = 0.01; 
t_max = [300.00, 300.00] ; 
dt_ansys = 0.05; 
sf = 1.00
test_directory = 'C:/Users/Moayad/Desktop/python/stick-model/test'
fnames = [f'{test_directory}/record.txt', f'{test_directory}/record.txt']

model = stickModel(nst,flh,flm,fll,fla,structTypo,gitDir)
model.mdof_initialise()
model.mdof_nodes()
model.mdof_fixity()
model.mdof_loads()
model.mdof_material()
model.plot_model()
model.do_gravity_analysis()
model.do_modal_analysis()
ops.wipeAnalysis()


control_nodes, coll_index, peak_drift, peak_accel, max_peak_drift, max_peak_drift_dir, max_peak_drift_loc, max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp = model.do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, 5)
    

# plot acceleration profiles (in x)
plt.plot(peak_accel[:,0], control_nodes, color = 'blue',linestyle='solid')
plt.xlabel("Peak Floor Acceleration, PFA [g]")
plt.ylabel("Floor No.")
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.xlim([0,1])
plt.ylim([0,np.max(control_nodes)])
plt.show()

# plot drift profiles (in x)

bplot_drift, bplot_nodes = duplicate_for_drift(peak_drift[:,0], control_nodes)
plt.plot(bplot_drift, bplot_nodes, color = 'blue',linestyle='solid')
plt.xlabel("Peak Storey Drift, PSD [%]")
plt.ylabel("Floor No.")
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.show()

# plot drift profiles (in y)

bplot_drift, bplot_nodes = duplicate_for_drift(peak_drift[:,1], control_nodes)
plt.plot(bplot_drift, bplot_nodes, color = 'blue',linestyle='solid')
plt.xlabel("Peak Storey Drift, PSD [%]")
plt.ylabel("Floor No.")
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.show()

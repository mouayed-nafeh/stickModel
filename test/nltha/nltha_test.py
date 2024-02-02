# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:57:21 2024

@author: Moayad
"""

#%% Load dependencies

# define the path to input file
gitDir = 'C:/Users/Moayad/Documents/GitHub/stickModel/'

# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, f'{gitDir}')

import pandas as pd
import numpy as np
from stickModel import stickModel
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from utils import *

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
structTypo = 'CR_LFINF+CDH+DUL_H1'

#%% Test NLTHA

# time-history parameters
fnames = [f'{gitDir}/test/nltha/record.txt']
dt_gm = 0.01         # acceleration time-history time step
t_max = 300.00       # maximum duration
dt_ansys = 0.05      # analysis time step
sf = 1.00            # scaling factor

# compile the model
ops.wipe()
model = stickModel(nst,flh,flm,fll,fla,structTypo,f'{gitDir}/raw/ip')
model.mdof_initialise()
model.mdof_nodes()
model.mdof_fixity()
model.mdof_loads()
model.mdof_material()
model.plot_model()
model.do_gravity_analysis()
model.do_modal_analysis()
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
bplot_driftX, bplot_nodes = duplicate_for_drift(peak_drift[:,0], control_nodes)
plt.plot(bplot_driftX, bplot_nodes, color = 'blue',linestyle='solid')
plt.xlabel("Peak Storey Drift, PSD [%]")
plt.ylabel("Floor No.")
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.show()

# plot drift profiles (in y)
bplot_driftY, bplot_nodes = duplicate_for_drift(peak_drift[:,1], control_nodes)
plt.plot(bplot_driftY, bplot_nodes, color = 'blue',linestyle='solid')
plt.xlabel("Peak Storey Drift, PSD [%]")
plt.ylabel("Floor No.")
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.show()

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

#%% Test SPO

ref_disp = 0.01
disp_scale_factor = 10
push_dir = 1

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

spo_disps, spo_rxn = model.do_spo_analysis(ref_disp, disp_scale_factor, push_dir, pflag=True)


plt.plot(spo_disps[:,-1], spo_rxn, color = 'black',linestyle='solid')
plt.xlabel("Top Displacement, $\delta$ [m]")
plt.ylabel("Base Shear, V [kN]")
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.xlim([0,np.max(spo_disps[:,-1])])
plt.ylim([0,1.5])



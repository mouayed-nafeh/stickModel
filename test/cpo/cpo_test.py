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


structTypo = 'CR_LFINF+CDH+DUL_H1'
#structTypo = 'S_LFM+CDM+DUM_H6'
#structTypo = 'S_LWAL+CDH+DUH_H11'

gitDir = 'C:/Users/Moayad/Documents/GitHub/stickModel/raw/ip'

D = list(pd.read_csv(f'{gitDir}/{structTypo}.csv').iloc[:,0])
F = list(pd.read_csv(f'{gitDir}/{structTypo}.csv').iloc[:,1])



#%% Test static pushover analysis

# ref_disp = 0.1
# disp_scale_factor = 10
# push_dir = 1

# model = stickModel(nst,flh,flm,fll,fla,structTypo,gitDir)
# model.mdof_initialise()
# model.mdof_nodes()
# model.mdof_fixity()
# model.mdof_loads()
# model.mdof_material()
# model.do_gravity_analysis()
# ops.wipeAnalysis()
# model.do_modal_analysis()
# ops.wipeAnalysis()

# spo_disps, spo_rxn = model.do_spo_analysis(ref_disp, disp_scale_factor, push_dir, pflag=True)
# ops.wipe()
 
# #%% Test cyclic pushover analysis

# ref_disp = 0.005
# numCycles = 1000
# push_dir = 2
# dispIncr = 200
# mu_list = [10]

# for i in range(len(mu_list)):

#     model = stickModel(nst,flh,flm,fll,fla,structTypo,gitDir)
#     model.mdof_initialise()
#     model.mdof_nodes()
#     model.mdof_fixity()
#     model.mdof_loads()
#     model.mdof_material()
#     model.plot_model()
#     model.do_gravity_analysis()
#     ops.wipeAnalysis()
#     model.do_modal_analysis()
#     ops.wipeAnalysis()
    
#     cpo_disps, cpo_rxn = model.do_cpo_analysis(ref_disp, mu_list[i], numCycles, push_dir, dispIncr, pflag = False)
    
#     plt.plot(spo_disps[:,nst-1], spo_rxn, color = 'black',linestyle='solid')
#     plt.plot(-spo_disps[:,nst-1], -spo_rxn, color = 'black',linestyle='solid')
#     plt.plot(cpo_disps[:,nst-1], cpo_rxn, color = 'blue',linestyle='dashed')
#     plt.xlabel("Top Displacement, $\delta$ [m]")
#     plt.ylabel("Base Shear, V [kN]")
#     plt.grid(visible=True, which='major')
#     plt.grid(visible=True, which='minor')
#     plt.xlim([-0.2,0.2])


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

ref_disp = 0.002 # Reference displacement (approximately the yield displacement)
numCycles = 5 # Number of cycles 
push_dir = 1     # Direction of pushover
dispIncr = 500   # Displacement increment 
mu_list = [0.5, 1.0, 2.0, 4.0, 5.0]   # List of target ductility

for i in range(len(mu_list)):

    # compile the model for every analysis object
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
    ops.wipeAnalysis()

    # do the analysis
    cpo_disps, cpo_rxn = model.do_cpo_analysis(ref_disp, mu_list[i], numCycles, push_dir, dispIncr, pflag = False)
    # plot the results
    plt.plot(cpo_disps[:,-1], cpo_rxn, color = 'blue',linestyle='dashed')
    plt.xlabel("Top Displacement, $\delta$ [m]")
    plt.ylabel("Base Shear, V [kN]")
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor')
    plt.xlim([-0.01,0.01])



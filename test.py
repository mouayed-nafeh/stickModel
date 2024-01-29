# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:57:21 2024

@author: Moayad
"""

# load dependencies
import pandas as pd
from stickModel import stickModel
import matplotlib
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


structTypo = 'CR_LDUAL+CDH+DUH_H1'
gitDir = 'C:/Users/Moayad/Documents/GitHub/stickModel/raw/ip'

D = list(pd.read_csv(f'{gitDir}/{structTypo}.csv').iloc[:,0])
F = list(pd.read_csv(f'{gitDir}/{structTypo}.csv').iloc[:,1])

#%% Create a stick model

model = stickModel(nst,flh,flm,fll,fla,structTypo,gitDir)
model.mdof_initialise()
model.mdof_nodes()
model.mdof_fixity()
model.mdof_loads()
model.mdof_material()
model.do_gravity_analysis()
model.do_modal_analysis()




#model.compileModel()

#%% Perform pushover analysis


# pushover analysis parameters
ref_disp = 0.1
disp_scale_factor = 10
push_dir = 2

spo_disp, spo_rxn = model.do_spo_analysis(ref_disp, disp_scale_factor, push_dir)



    
    
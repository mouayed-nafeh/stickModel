# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:57:21 2024

@author: Moayad
"""

# load dependencies
import pandas as pd
from stickModel import stickModel
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

# test size
if len(flh)!=nst or len(flm)!=nst or len(fll)!=nst or len(fla)!=nst:
    raise ValueError('Number of entries exceed the number of storeys')
    
#%% Create a stick model

model = stickModel(path_to_input_file)
model.compileModel()

#%% Perform pushover analysis


# pushover analysis parameters
ref_disp = 0.1
disp_scale_factor = 10
push_dir = 2

spo_disp, spo_rxn = model.do_spo_analysis(ref_disp, disp_scale_factor, push_dir, pflag=True, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-8, init_iter=1000, algorithm_type='KrylovNewton')




    
    
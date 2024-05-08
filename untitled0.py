# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:34:43 2024

@author: Moayad
"""

import pandas as pd
import numpy as np
import os
import pip

### Check if storeyloss is installed, if not, pip install it.
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])
import_or_install('storeyloss')
import storeyloss

def create_csv_from_database(path_to_database, bldgClass, bldgOccupancy, region, outDir):
    
    ### Load the table
    table = pd.read_excel(f'{path_to_database}', sheet_name = bldgClass)

    ### Item ID
    itemID = [i+1 for i in range(len(table))]

    ### Component ID
    componentID = [i for i in table.Components]

    ### Engineering Demand Parameter
    edp = [i for i in table['Engineering Demand Parameter']]

    ### Component Type
    componentType = ['NS']*len(table)

    ### Component Group
    componentGroup = [int(i) for i in table['Performance Group']]

    ### Component Quantity
    if bldgOccupancy == 'Res':
        componentQuantity = [i for i in table['RESIDENTIAL - METRIC']]
    elif bldgOccupancy == 'Com':
        componentQuantity = [i for i in table['COMMERCIAL - METRIC']]
    elif bldgOccupancy == 'Ind':
        componentQuantity = [i for i in table['INDUSTRIAL - METRIC']]
        
    ### Component Units
    componentUnits = [i for i in table['Quantity unit']]

    ### Damage States
    damageStates = [int(i) for i in table['Damage States']]

    ### Fetch remaining columns (fragilities and costs)
    for i, labels in enumerate(table.columns[10:]):
        vars()[f'{labels}'] = table[table.columns[10+i]].tolist()
        
    ### Build the dataFrame
    dataFrame = {'ITEM'         : itemID,
                 'ID'           : componentID,
                 'EDP'          : edp,
                 'Component'    : componentType,
                 'Group'        : componentGroup,
                 'Quantity'     : componentQuantity,
                 'Units'        : componentUnits,
                 'Damage States': damageStates}
    for i, labels in enumerate(table.columns[10:]):
        dataFrame[labels] = vars()[f'{labels}']

    ### Convert the dataFrame to csv
    df = pd.DataFrame(dataFrame)
    if not os.path.exists(f'{outDir}'):
        os.makedirs(f'{outDir}')
    df.to_csv(f'{outDir}/{bldgClass}_{bldgOccupancy}.csv', index=False)


def processSLFs(path_to_csv, bldgClass, bldgOccupancy, region, outDir):
    
    ### Process drift-sensitive components
    path = f'{path_to_csv}'
    component_psd = pd.read_csv(path)
    regF = 'Weibull'
    model = storeyloss.SLF(component_psd,edp= 'PSD', do_grouping= True,conversion = 1.00,realizations = 500,replacement_cost = 1.00,regression = regF)
    psd_slf, psd_cache = model.generate_slfs()
    
    ### Process acceleration-sensitive components
    path = f'{path_to_csv}'
    component_pfa = pd.read_csv(path)
    regF = 'Weibull'
    model = storeyloss.SLF(component_pfa,edp= 'PFA', do_grouping= True,conversion = 1.00,realizations = 500,replacement_cost = 1.00,regression = regF)
    pfa_slf, pfa_cache = model.generate_slfs()
    
    ### Evaluate the total repair cost of components
    total_psd_repair_cost= np.max(psd_slf[list(psd_slf.keys())[0]]['slf'])
    total_pfa_repair_cost= np.max(pfa_slf[list(pfa_slf.keys())[0]]['slf'])
    total_repair_cost= total_psd_repair_cost + total_pfa_repair_cost

    ### normalise the repair costs
    norm_psd = [x/total_repair_cost for x in psd_slf[list(psd_slf.keys())[0]]['slf']]
    norm_pfa = [x/total_repair_cost for x in pfa_slf[list(pfa_slf.keys())[0]]['slf']]

    ### adjust the range of psd to be in percent
    psd_slf[list(psd_slf.keys())[0]]['edp_range'] = [x*100 for x in psd_slf[list(psd_slf.keys())[0]]['edp_range']]

    
    
    
    


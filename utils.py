# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 08:31:21 2023

@author: Moayad
"""

def import_from_pkl(path):
    # import necessary libraries
    import pickle

    # import file
    with open(path, 'rb') as file:
        return pickle.load(file)


def export_to_pkl(path, var):
    # import necessary libraries
    import pickle

    # store file
    with open(path, 'wb') as file:
        return pickle.dump(var, file)

def read_one_column_file(file_name):
    with open(file_name, 'r') as data:
        x = []
        for number in data:
            x.append(float(number))

    return x

def read_two_column_file(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x, y

def duplicate_for_drift(drift_values,control_nodes):
    
    newDrft = []; newNodes = []
    nst = len(control_nodes)-1
    
    for i in range(nst):
        
        newNodes.extend((float(control_nodes[i]),float(control_nodes[i+1])))
        newDrft.extend((drift_values[i],drift_values[i]))
        
    
    newNodes.append(float(control_nodes[i+1]))
    newDrft.append(0.0)
        
    return newDrft, newNodes

def createPinching4(matTag, F,D):
    # import necessary libraries
    import openseesy.opensees as ops
    
    ePf1   = F[0];   ePf2 = F[1];   ePf3 = F[2], ePf4 = F[3]
    ePd1   = D[0];   ePd2 = D[1];   ePd3 = D[2], ePd4 = D[3]
    rDispP = 0.8; rForceP = 0.8; uForceP = 0.8;
    gK1    = 1.0;     gK2 = 0.2;     gK3 = 0.3;   gK4 = 0.2; gKLim = 0.9;
    gD1    = 0.5;     gD2 = 0.5;     gD3 = 2.0;   gD4 = 2.0; gDLim = 0.5;
    gF1    = 1.0;     gF2 = 0.0;     gF3 = 1.0;   gF4 = 1.0; gFLim = 0.9;
    gE     = 10.0;    dmgType = 'cycle'
    
    
    ops.uniaxialMaterial('Pinching4', matTag, 
                         ePf1, ePd1, ePf2, ePd2, 
                         ePf3, ePd3, ePf4, ePd4, 
                         rDispP, rForceP, uForceP,
                         gK1, gK2, gK3, gK4, gKLim, 
                         gD1, gD2, gD3, gD4, gDLim, 
                         gF1, gF2, gF3, gF4, gFLim, 
                         gE, dmgType)
    
    
    

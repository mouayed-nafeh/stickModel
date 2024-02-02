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


def createHystereticMaterial(matTag, F, D):
    # import necessary libraries
    import openseespy.opensees as ops
    
    pinchX = 1.0
    pinchY = 1.0
    damageX = 1.0
    damageY = 1.0
    
    if len(F)==2 and len(D)==2:
        # assign bilinear material
        ops.uniaxialMaterial('HystereticSM', matTag, '-posEnv', F[0], D[0], F[1], D[1], '-negEnv', -F[0], -D[0], -F[1], -D[1], '-pinch', pinchX, pinchY,'-damage', damageX, damageY, '-beta', 0)
    elif len(F)==3 and len(D)==3:
        # assign bilinear material
        ops.uniaxialMaterial('HystereticSM', matTag, '-posEnv', F[0], D[0], F[1], D[1], F[2], D[2], '-negEnv', -F[0], -D[0], -F[1], -D[1], -F[2], -D[2], '-pinch', pinchX, pinchY,'-damage', damageX, damageY, '-beta', 0)
    elif len(F)==4 and len(D)==4:
        # assign bilinear material
        ops.uniaxialMaterial('HystereticSM', matTag, '-posEnv', F[0], D[0], F[1], D[1], F[2], D[2], F[3], D[3],'-negEnv', -F[0], -D[0], -F[1], -D[1], -F[2], -D[2], -F[3], -D[3], '-pinch', pinchX, pinchY,'-damage', damageX, damageY, '-beta', 0)
        

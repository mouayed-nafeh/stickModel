##########################################################################
#                    GENERIC UTILITY FUNCTIONS                           #
##########################################################################
import numpy as np
import openseespy.opensees as ops
import pickle
import math
import re

### Function to import data stored in a pickle object
def import_from_pkl(path):
    # import file
    with open(path, 'rb') as file:
        return pickle.load(file)

### Function to store data in a pickle object
def export_to_pkl(path, var):
    # store file
    with open(path, 'wb') as file:
        return pickle.dump(var, file)

### Function to sort items alphanumerically
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

### Function to create box plot inputs for storey drift demands
def duplicate_for_drift(drifts,control_nodes):
    x = []; y = []
    for i in range(len(control_nodes)-1):
        y.extend((float(control_nodes[i]),float(control_nodes[i+1])))
        x.extend((drifts[i],drifts[i]))
    y.append(float(control_nodes[i+1]))
    x.append(0.0)
    return x, y

### Function to create nonlinear material for Opensees
def createHystereticMaterial(matTag, F, D, pinchX=0.50, pinchY=0.50, damageX= 0.50, damageY= 0.50):
    # Bilinear
    if len(F)==2 and len(D)==2:
        # assign bilinear material
        ops.uniaxialMaterial('HystereticSM', matTag, '-posEnv', F[0], D[0], F[1], D[1], '-negEnv', -F[0], -D[0], -F[1], -D[1], '-pinch', pinchX, pinchY,'-damage', damageX, damageY, '-beta', 0)
    # Trilinear
    elif len(F)==3 and len(D)==3:
        # assign bilinear material
        ops.uniaxialMaterial('HystereticSM', matTag, '-posEnv', F[0], D[0], F[1], D[1], F[2], D[2], '-negEnv', -F[0], -D[0], -F[1], -D[1], -F[2], -D[2], '-pinch', pinchX, pinchY,'-damage', damageX, damageY, '-beta', 0)
    # Multilinear
    elif len(F)==4 and len(D)==4:
        # assign bilinear material
        ops.uniaxialMaterial('HystereticSM', matTag, '-posEnv', F[0], D[0], F[1], D[1], F[2], D[2], F[3], D[3],'-negEnv', -F[0], -D[0], -F[1], -D[1], -F[2], -D[2], -F[3], -D[3], '-pinch', pinchX, pinchY,'-damage', damageX, damageY, '-beta', 0)

### Function to perform relative squared error estimation for uncertainty quantification in regression
def RSE(y_true, y_predicted):
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))
    rse = math.sqrt(RSS / (len(y_true) - 2))
    return rse

### Function to remove items from list (useful for filtering based on condition)
def remove_elements_at_indices(test_list, idx_list):
    # Base case: if index list is empty, return original list
    if not idx_list:
        return test_list
    # Recursive case: extract first index and recursively process the rest of the list
    first_idx = idx_list[0]
    rest_of_indices = idx_list[1:]
    sub_list = remove_elements_at_indices(test_list, rest_of_indices)
    # Remove element at current index
    sub_list.pop(first_idx)
    return sub_list


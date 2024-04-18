##########################################################################
#                    GENERIC UTILITY FUNCTIONS                           #
##########################################################################
import numpy as np
import openseespy.opensees as ops
import pickle
import math
import re

### Weibull function
def fun_weibull(x, a, b, c):
    return a * (1 - np.exp(-(x / b) ** c))

### Logistic function
def fun_logit(x,a,b):
    return np.exp(-(a+b*np.log(x)))/(1+np.exp(-(a+b*np.log(x))))

### Function to look for substrings
def find_between( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

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

### Function to read one column file
def read_one_column_file(file_name):
    with open(file_name, 'r') as data:
        x = []
        for number in data:
            x.append(float(number))
    return x

### Function to read two-column file
def read_two_column_file(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))
    return x, y

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

### Function to create hystereticSM nonlinear material for Opensees
def createHystereticMaterial(matTag, F, D, pinchX=0.80, pinchY=0.20, damageX= 0.01, damageY= 0.01):
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

### Function to create pinching4 nonlinear material for Opensees
def createPinching4Material(matTag, F, D):   

    f_vec=np.zeros([5,1])
    d_vec=np.zeros([5,1])
    
    # Bilinear
    if len(F)==2:
          #bilinear curve
          f_vec[1]=F[0]
          f_vec[4]=F[-1]
          
          d_vec[1]=D[0]
          d_vec[4]=D[-1]
          
          d_vec[2]=d_vec[1]+(d_vec[4]-d_vec[1])/3
          d_vec[3]=d_vec[1]+2*((d_vec[4]-d_vec[1])/3)
          
          f_vec[2]=np.interp(d_vec[2],D,F)
          f_vec[3]=np.interp(d_vec[3],D,F)
    
    # Trilinear
    elif len(F)==3:
          
          f_vec[1]=F[0]
          f_vec[4]=F[-1]
          
          d_vec[1]=D[0]
          d_vec[4]=D[-1]
          
          f_vec[2]=F[1]
          d_vec[2]=D[1]
          
          d_vec[3]=np.mean([d_vec[2],d_vec[-1]])
          f_vec[3]=np.interp(d_vec[3],D,F)
    
    # Quadrilinear
    elif len(F)==4:
          f_vec[1]=F[0]
          f_vec[4]=F[-1]
          
          d_vec[1]=D[0]
          d_vec[4]=D[-1]
          
          f_vec[2]=F[1]
          d_vec[2]=D[1]
          
          f_vec[3]=F[2]
          d_vec[3]=D[2]
          
    matargs=[f_vec[1,0],d_vec[1,0],f_vec[2,0],d_vec[2,0],f_vec[3,0],d_vec[3,0],f_vec[4,0],d_vec[4,0],
                         -1*f_vec[1,0],-1*d_vec[1,0],-1*f_vec[2,0],-1*d_vec[2,0],-1*f_vec[3,0],-1*d_vec[3,0],-1*f_vec[4,0],-1*d_vec[4,0],
                         0.5,0.25,0.05,
                         0.5,0.25,0.05,
                         0,0,0,0,0,
                         0,0,0,0,0,
                         0,0,0,0,0,
                         10,'energy']
    ops.uniaxialMaterial('Pinching4', matTag,*matargs)

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

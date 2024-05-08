##########################################################################
#                     SDOF-MDOF CALIBRATION MODULE                       #
##########################################################################   
import sys
repoDir = 'C:/Users/Moayad/Documents/GitHub/stickModel'
sys.path.insert(1, f'{repoDir}')
from stickModel import stickModel
from utils import *
from units import *
from postprocessors import *
from plotters import *
from im_calculator import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import os
from math import sqrt, pi

def calibrateModel(nst,gamma,sdofCapArray, tol=1e-2, pflag=False):
        
    # Initialise the mode shape
    phi_mdof = np.linspace(1/nst, 1, nst)
    print('Assumed Mode-Shape:',phi_mdof)
    
    # Get the masses
    mass = 1/sum(phi_mdof)
    print('Floor Mass:',mass)
    
    # Assign the MDOF mass
    flm_mdof = [mass]*nst

    ### Get the MDOF Capacity Curves Storey-Deformation Relationship
    rows, columns = np.shape(sdofCapArray)
    stD_mdof = np.zeros([nst,rows])
    stF_mdof = np.zeros([nst,rows])
    for i in range(nst):
        # get the displacement or spectral displacement arrays at each storey
        stF_mdof[i,:] = sdofCapArray[:,1].transpose()*gamma*units.g*sum(flm_mdof)
        # get the force or spectral acceleration arrays at each storey
        stD_mdof[i,:] = sdofCapArray[:,0].transpose()*gamma*phi_mdof[i]
        
    ### Get the elastic stiffness of each floor
    elasticStiffness = []
    for i in range(nst):
        elasticStiffness.append(stF_mdof[i,0]/stD_mdof[i,0])
        print('Elastic Stiffness of',f'{i+1}-th Storey:', elasticStiffness[i])
        
    ### Estimate the elastic period
    elasticPeriod = 2*pi*np.sqrt(sum(flm_mdof)/sum(elasticStiffness))

    ### Prepare the SDOF model
    flh_sdof = [2.8]*1 # arbitray height 
    flm_sdof = [1.0]*1 # unit mass (1 tonne)
    rows, columns = np.shape(sdofCapArray)
    stD_sdof = np.zeros([1,rows])
    stF_sdof = np.zeros([1,rows])
    for i in range(1):
        # get the displacement or spectral displacement arrays at each storey
        stF_sdof[i,:] = sdofCapArray[:,1].transpose()*gamma*units.g
        # get the force or spectral acceleration arrays at each storey
        stD_sdof[i,:] = sdofCapArray[:,0].transpose()*gamma
           
    ### Compile the SDOF model
    modelSDOF = stickModel(1,flh_sdof,flm_sdof,stD_sdof,stF_sdof)    # Build the model
    modelSDOF.mdof_initialise()                                      # Initialise the domain
    modelSDOF.mdof_nodes()                                           # Construct the nodes
    modelSDOF.mdof_fixity()                                          # Set the boundary conditions 
    modelSDOF.mdof_loads()                                           # Assign the loads
    modelSDOF.mdof_material()                                        # Assign the nonlinear storey material
    modelSDOF.plot_model()                                           # Visualise the model                               
    modelSDOF.do_gravity_analysis()                                  # Do gravity analysis
    sdofT = modelSDOF.do_modal_analysis(num_modes = 1)               # Do modal analysis and get period of vibration

    ### Run static pushover analyses on SDOF model
    ref_disp = 0.01
    disp_scale_factor = 100
    push_dir = 1
    spoSDOF_disps, spoSDOF_rxn= modelSDOF.do_spo_analysis(ref_disp,disp_scale_factor,push_dir, pflag)

    ### Compile the MDOF
    flh_mdof = [2.8]*nst
    modelMDOF = stickModel(nst,flh_mdof,flm_mdof,stD_mdof,stF_mdof)    # Build the model
    modelMDOF.mdof_initialise()                                      # Initialise the domain
    modelMDOF.mdof_nodes()                                           # Construct the nodes
    modelMDOF.mdof_fixity()                                          # Set the boundary conditions 
    modelMDOF.mdof_loads()                                           # Assign the loads
    modelMDOF.mdof_material()                                        # Assign the nonlinear storey material
    modelMDOF.plot_model()                                           # Visualise the model                               
    modelMDOF.do_gravity_analysis()                                  # Do gravity analysis
    mdofT = modelMDOF.do_modal_analysis(num_modes = 1)               # Do modal analysis and get period of vibration

    ### Run Static Pushover Analyses on MDOF 
    ref_disp = 0.01
    disp_scale_factor = 20
    push_dir = 1
    spoMDOF_disps, spoMDOF_rxn= modelSDOF.do_spo_analysis(ref_disp,disp_scale_factor,push_dir, pflag)

    ### Test 1: Check the periods
    error1 = (sdofT[0]-mdofT[0])/mdofT[0]*100
    if error1 < tol:
        print('Period check satisfied!')
        print('Error 1:', error1)
    else:
        print('Error 1:', error1)
        #raise ValueError('MDOF and SDOF periods do not match!! Revise your input')
    
    ### Test 2: Equivalent Masses
    error2 = (np.dot(flm_mdof,phi_mdof)-1.00)/1.00*100
    if error2 <= tol:
        print('Equivalent mass equation verified')
        print('Error 2:', error2)
    else:
        print('Error 2:', error2)
        #raise ValueError('Equivalent mass equation not satisfied!! Revise your input')
        
    ### Test 3: Transformation Equation
    error3 = (np.dot(flm_mdof,phi_mdof**2)-1/gamma)/(1/gamma)*100
    if error3 <= tol:
        print('Transformation factor equation verified')
        print('Error 3:', error3)
    else:
        print('Error 3:', error3)
        #raise ValueError('Transformation factor equation not satisfied!! Revise your input')

    if pflag:
        
        ### Print the modal analysis results
        print('SDOF Elastic Period from Modal Analysis:', sdofT[0], 's')
        print('MDOF Elastic Period from Modal Analysis:', mdofT[0], 's')
    
        ### Plot the comparison
        fig=plt.figure()    
        # Plot the individual storeys
        for i in range(nst): 
            storeyD = np.concatenate(([0.0],stD_mdof[i,:]))
            storeyF = np.concatenate(([0.0],stF_mdof[i,:]))
            plt.plot(storeyD,storeyF,'--',label = f'storey {i}')
        
        # Plot the backbone of the SDOF system
        plt.plot(spoSDOF_disps[:,-1],spoSDOF_rxn,'r--',label='Model - SDOF')
        # Plot the calibrated backbone of the MDOF 
        plt.plot(spoMDOF_disps[:,-1],spoMDOF_rxn,label='Model - MDOF')
        plt.grid(visible=True, which='major')
        plt.grid(visible=True, which='minor')
        plt.xlabel('Spectral Displacement, Sd [m]')
        plt.ylabel('Spectral Acceleration, Sa [g]')
        plt.title('Comparison of Capacity Curves')
        plt.legend()

    ### Pack the errors
    errors = [error1, error2, error3]
        
    return mdofT, flm_mdof, stD_mdof, stF_mdof, errors


def calibrateModel2(nst, gamma, sdofCapArray, pflag=False):
        
    # Initialise the mode shape
    phi_mdof = np.linspace(1/nst, 1, nst)
    print('Assumed Mode-Shape:',phi_mdof)
    
    # Get the masses
    mass = 1/sum(phi_mdof)
    print('Floor Mass:',mass)
    
    # Assign the MDOF mass
    flm_mdof = [mass]*nst

    ### Get the MDOF Capacity Curves Storey-Deformation Relationship
    rows, columns = np.shape(sdofCapArray)
    stD_mdof = np.zeros([nst,rows])
    stF_mdof = np.zeros([nst,rows])
    for i in range(nst):
        # get the displacement or spectral displacement arrays at each storey
        stF_mdof[i,:] = sdofCapArray[:,1].transpose()*gamma
        # get the force or spectral acceleration arrays at each storey
        stD_mdof[i,:] = sdofCapArray[:,0].transpose()*gamma/nst
        
    ### Prepare the SDOF model
    flh_sdof = [2.8]*1 # arbitray height 
    flm_sdof = [1.0]*1 # unit mass (1 tonne)
    rows, columns = np.shape(sdofCapArray)
    stD_sdof = np.zeros([1,rows])
    stF_sdof = np.zeros([1,rows])
    for i in range(1):
        # get the displacement or spectral displacement arrays at each storey
        stF_sdof[i,:] = sdofCapArray[:,1].transpose()*gamma
        # get the force or spectral acceleration arrays at each storey
        stD_sdof[i,:] = sdofCapArray[:,0].transpose()*gamma
           
    ### Compile the SDOF model and run SPO analysis
    modelSDOF = stickModel(1,flh_sdof,flm_sdof,stD_sdof,stF_sdof*units.g)    # Build the model
    modelSDOF.mdof_initialise()                                      # Initialise the domain
    modelSDOF.mdof_nodes()                                           # Construct the nodes
    modelSDOF.mdof_fixity()                                          # Set the boundary conditions 
    modelSDOF.mdof_material()                                        # Assign the nonlinear storey material
    modelSDOF.plot_model()                                           # Visualise the model                               
    modelSDOF.do_gravity_analysis()                                  # Do gravity analysis
    sdofT = modelSDOF.do_modal_analysis(num_modes = 1)               # Do modal analysis and get period of vibration
    ref_disp = 0.01
    disp_scale_factor = 100
    push_dir = 1
    spoSDOF_disps, spoSDOF_rxn= modelSDOF.do_spo_analysis(ref_disp,disp_scale_factor,push_dir, pflag)

    ### Compile the MDOF and run SPO analysis
    flh_mdof = [2.8]*nst
    modelMDOF = stickModel(nst,flh_mdof,flm_mdof,stD_mdof,stF_mdof*units.g)    # Build the model
    modelMDOF.mdof_initialise()                                      # Initialise the domain
    modelMDOF.mdof_nodes()                                           # Construct the nodes
    modelMDOF.mdof_fixity()                                          # Set the boundary conditions 
    modelMDOF.mdof_material()                                        # Assign the nonlinear storey material
    modelMDOF.plot_model()                                           # Visualise the model                               
    modelMDOF.do_gravity_analysis()                                  # Do gravity analysis
    mdofT = modelMDOF.do_modal_analysis(num_modes = 1)               # Do modal analysis and get period of vibration
    ref_disp = 0.01
    disp_scale_factor = 20
    push_dir = 1
    spoMDOF_disps, spoMDOF_rxn= modelMDOF.do_spo_analysis(ref_disp,disp_scale_factor,push_dir, pflag)


    if pflag:
            
        ### Plot the comparison
        fig=plt.figure()    
        # Plot the individual storeys
        for i in range(nst): 
            storeyD = np.concatenate(([0.0],stD_mdof[i,:]))
            storeyF = np.concatenate(([0.0],stF_mdof[i,:]))
            plt.plot(storeyD,storeyF*units.g,'--',label = f'storey {i}')
        
        # Plot the backbone of the SDOF system
        plt.plot(spoSDOF_disps[:,-1],spoSDOF_rxn,'r--',label='Model - SDOF')
        # Plot the calibrated backbone of the MDOF 
        plt.plot(spoMDOF_disps[:,-1],spoMDOF_rxn,label='Model - MDOF')
        plt.grid(visible=True, which='major')
        plt.grid(visible=True, which='minor')
        plt.xlabel('Spectral Displacement, Sd [m]')
        plt.ylabel('Spectral Acceleration, Sa [g]')
        plt.title('Comparison of Capacity Curves')
        plt.legend()
        
    return mdofT, flm_mdof, stD_mdof, stF_mdof
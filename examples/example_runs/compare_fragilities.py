#%% Load Dependencies
import sys
repoDir = 'C:/Users/Moayad/Documents/GitHub/stickModel'
sys.path.insert(1, f'{repoDir}')

from plotters import *
from vulnerability import *
from postprocessors import *
from utils import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

colors = ['blue','green','yellow','red','black']

#%% Load the IM-EDP relationships

im_labels = ['PGA','SA(T=0.3s)','SA(T=0.6s)','SA(T=1.0s)',
             'Saavg(T=0.3s)','Saavg(T=0.6s)','Saavg(T=1.0s)']

arbitraryThresholds = [0.1/100, 0.5/100, 1/100, 2/100]

files = sorted_alphanumeric([f for f in os.listdir(f'{repoDir}/examples/example_runs') if not f.endswith('.py')])


# Initialise some arrays for storage
fragParams = {}

thetaSDOF = np.zeros((len(files),len(arbitraryThresholds)))
thetaMDOF = np.zeros((len(files),len(arbitraryThresholds)))
betaSDOF = np.zeros((len(files),len(arbitraryThresholds)))
betaMDOF = np.zeros((len(files),len(arbitraryThresholds)))

for i, runs in enumerate(files):
    
    currentRun = import_from_pkl(f'{repoDir}/examples/example_runs/{runs}/{runs}.pkl')
    
    for j, currentIM in enumerate(im_labels):

        fragParams[currentIM] = {}    
        fragParams[currentIM]['SDOF_Medians'] = np.zeros((len(files),len(arbitraryThresholds)))    
        fragParams[currentIM]['MDOF_Medians'] = np.zeros((len(files),len(arbitraryThresholds))) 
        fragParams[currentIM]['SDOF_Betas'] = np.zeros((len(files),len(arbitraryThresholds)))    
        fragParams[currentIM]['MDOF_Betas'] = np.zeros((len(files),len(arbitraryThresholds))) 
        
        if j==0:
            # Peak Ground Acceleration
            im = currentRun['pga']
            edpMDOF = currentRun['mdof_max_peak_drift_list']
            edp_fitted_MDOF = currentRun['mdof_edpvector1']
            im_fitted_MDOF  = currentRun['mdof_imvector1']
                    
            edpSDOF = currentRun['sdof_max_peak_drift_list']
            edp_fitted_SDOF = currentRun['sdof_edpvector1']
            im_fitted_SDOF  = currentRun['sdof_imvector1']
                    
        if j==1:
            # Spectral Acceleration at 0.3s
            im = currentRun['sa03']
            edpMDOF = currentRun['mdof_max_peak_drift_list']
            edp_fitted_MDOF = currentRun['mdof_edpvector2']
            im_fitted_MDOF  = currentRun['mdof_imvector2']
                    
            edpSDOF = currentRun['sdof_max_peak_drift_list']
            edp_fitted_SDOF = currentRun['sdof_edpvector2']
            im_fitted_SDOF  = currentRun['sdof_imvector2']
        
        if j==2:
            # Spectral Acceleration at 0.6s
            im = currentRun['sa06']
            edpMDOF = currentRun['mdof_max_peak_drift_list']
            edp_fitted_MDOF = currentRun['mdof_edpvector3']
            im_fitted_MDOF  = currentRun['mdof_imvector3']
                    
            edpSDOF = currentRun['sdof_max_peak_drift_list']
            edp_fitted_SDOF = currentRun['sdof_edpvector3']
            im_fitted_SDOF  = currentRun['sdof_imvector3']
    
        if j==3:
            # Spectral Acceleration at 1.0s
            im = currentRun['sa10']
            edpMDOF = currentRun['mdof_max_peak_drift_list']
            edp_fitted_MDOF = currentRun['mdof_edpvector4']
            im_fitted_MDOF  = currentRun['mdof_imvector4']
                    
            edpSDOF = currentRun['sdof_max_peak_drift_list']
            edp_fitted_SDOF = currentRun['sdof_edpvector4']
            im_fitted_SDOF  = currentRun['sdof_imvector4']
    
        if j==4:
            # Average Spectral Acceleration at 0.3s
            im = currentRun['saavg03']
            edpMDOF = currentRun['mdof_max_peak_drift_list']
            edp_fitted_MDOF = currentRun['mdof_edpvector5']
            im_fitted_MDOF  = currentRun['mdof_imvector5']
                    
            edpSDOF = currentRun['sdof_max_peak_drift_list']
            edp_fitted_SDOF = currentRun['sdof_edpvector5']
            im_fitted_SDOF  = currentRun['sdof_imvector5']
            
        if j==5:
            # Average Spectral Acceleration at 0.6s
            im = currentRun['saavg06']
            edpMDOF = currentRun['mdof_max_peak_drift_list']
            edp_fitted_MDOF = currentRun['mdof_edpvector6']
            im_fitted_MDOF  = currentRun['mdof_imvector6']
                    
            edpSDOF = currentRun['sdof_max_peak_drift_list']
            edp_fitted_SDOF = currentRun['sdof_edpvector6']
            im_fitted_SDOF  = currentRun['sdof_imvector6']
        
        if j==6:
            # Average Spectral Acceleration at 1.0s
            im = currentRun['saavg10']
            edpMDOF = currentRun['mdof_max_peak_drift_list']
            edp_fitted_MDOF = currentRun['mdof_edpvector7']
            im_fitted_MDOF  = currentRun['mdof_imvector7']
                    
            edpSDOF = currentRun['sdof_max_peak_drift_list']
            edp_fitted_SDOF = currentRun['sdof_edpvector7']
            im_fitted_SDOF  = currentRun['sdof_imvector7']

        for k, damageThreshold in enumerate(arbitraryThresholds):
            
            # MDOF Fragility
            theta, beta = calculateFragParams(edpMDOF, im, edp_fitted_MDOF, im_fitted_MDOF, damageThreshold, beta_build2build =0.3)
            imlsMDOF, poeMDOF = getDamageProbability(theta, beta)
            fragParams[currentIM]['MDOF_Medians'][i,k] = theta
            fragParams[currentIM]['MDOF_Betas'][i,k] = beta
            plt.plot(imlsMDOF, poeMDOF, linewidth=2.5, linestyle = '-', color = colors[k], label = f'DS{k+1} - MDOF')
                        
            # SDOF Fragility
            theta, beta = calculateFragParams(edpSDOF, im, edp_fitted_SDOF, im_fitted_SDOF, damageThreshold, beta_build2build =0.3)
            imlsSDOF, poeSDOF = getDamageProbability(theta, beta)        
            fragParams[currentIM]['SDOF_Medians'][i,k] = theta
            fragParams[currentIM]['SDOF_Betas'][i,k] = beta
            plt.plot(imlsSDOF, poeSDOF, linewidth=2.5, linestyle = '--', color = colors[k], label = f'DS{k+1} - SDOF')
        
        plt.xlabel(currentIM)
        plt.ylabel(r'Probability of Damage')
        plt.grid(visible=True, which='major')
        plt.grid(visible=True, which='minor')
        plt.ylim([0, 1])
        plt.xlim([0, 2])
        plt.legend(loc='lower right')
        plt.savefig(f'{repoDir}/examples/example_runs/{runs}/{currentIM}_{runs}.png', dpi=600, format='png')
        plt.show()

    


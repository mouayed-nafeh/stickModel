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
from matplotlib.lines import Line2D
import os 

colors = ['blue','green','yellow','red','black']
markers = list(Line2D.markers)

#%% Load the IM-EDP relationships

### Define output directory
outDir = f'{repoDir}/examples/example_runs/output'

### Define intensity measure labels based on their hierarchy in the analysis dictionary
im_labels = ['PGA','SA(T=0.3s)','SA(T=0.6s)','SA(T=1.0s)',
             'Saavg(T=0.3s)','Saavg(T=0.6s)','Saavg(T=1.0s)']

### Define edp-based damage thresholds for fragility analysis (these values were assigned arbitrarily for this comparative assessment)
arbitraryThresholds = [0.1/100, 0.5/100, 1/100, 2/100]

### Fetch the files in the example runs folder
files = sorted_alphanumeric([f for f in os.listdir(f'{repoDir}/examples/example_runs/analysis') if not f.endswith('.py')])

### Initialise dictionary, sub-dictionaries and arrays for storage
fragParams = {}
for j, currentIM in enumerate(im_labels):
    fragParams[currentIM] = {}    
    fragParams[currentIM]['Building_Class'] = []
    fragParams[currentIM]['SDOF_Medians'] = np.zeros((len(files),len(arbitraryThresholds)))    
    fragParams[currentIM]['MDOF_Medians'] = np.zeros((len(files),len(arbitraryThresholds))) 
    fragParams[currentIM]['SDOF_Betas'] = np.zeros((len(files),len(arbitraryThresholds)))    
    fragParams[currentIM]['MDOF_Betas'] = np.zeros((len(files),len(arbitraryThresholds))) 

### Iterate over the analysis folder
for i, runs in enumerate(files):
    
    ### Import the current analysis files from pickle
    currentRun = import_from_pkl(f'{repoDir}/examples/example_runs/analysis/{runs}/{runs}.pkl')
    fragParams[currentIM]['Building_Class'].append(runs)
    
    ### Iterate over the intensity measures (this part is semi-hard-coded due to the nature of the input)
    for j, currentIM in enumerate(im_labels):
        
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
        
        ### Loop over the edp-based damage thresholds or damage states
        for k, damageThreshold in enumerate(arbitraryThresholds):
            
            ### Calculate MDOF fragility, store the parameters and plot the fragility curves
            theta, beta = calculateFragParams(edpMDOF, im, edp_fitted_MDOF, im_fitted_MDOF, damageThreshold, beta_build2build =0.3)
            imlsMDOF, poeMDOF = getDamageProbability(theta, beta)
            fragParams[currentIM]['MDOF_Medians'][i,k] = theta
            fragParams[currentIM]['MDOF_Betas'][i,k] = beta
            plt.plot(imlsMDOF, poeMDOF, linewidth=2.5, linestyle = '-', color = colors[k], label = f'DS{k+1} - MDOF')
                        
            ### Calculate SDOF fragility, store the parameters and plot the fragility curves
            theta, beta = calculateFragParams(edpSDOF, im, edp_fitted_SDOF, im_fitted_SDOF, damageThreshold, beta_build2build =0.3)
            imlsSDOF, poeSDOF = getDamageProbability(theta, beta)        
            fragParams[currentIM]['SDOF_Medians'][i,k] = theta
            fragParams[currentIM]['SDOF_Betas'][i,k] = beta
            plt.plot(imlsSDOF, poeSDOF, linewidth=2.5, linestyle = '--', color = colors[k], label = f'DS{k+1} - SDOF')
        
        ### Finish the rest of the plot elements and save it
        plt.xlabel(currentIM)
        plt.ylabel(r'Probability of Damage')
        plt.grid(visible=True, which='major')
        plt.grid(visible=True, which='minor')
        plt.ylim([0, 1])
        plt.xlim([0, 2])
        plt.legend(loc='lower right')
        plt.savefig(f'{outDir}/{currentIM}_{runs}.png', dpi=600, format='png')
        plt.show()

#%% Calculate and visualise deviations in estimation

### Calculate deviations between MDOF and SDOF fragility parameters
### Loop over the files
fragParams[currentIM]['Intensity_Errors']  = np.zeros((len(files),len(arbitraryThresholds)))
fragParams[currentIM]['Dispersion_Errors'] = np.zeros((len(files),len(arbitraryThresholds)))
for i, runs in enumerate(files):
    ### Loop over the intensity measures
    for j, currentIM in enumerate(im_labels):   
        ### Loop over the damage states
        for k, damageThreshold in enumerate(arbitraryThresholds):
            fragParams[currentIM]['Intensity_Errors'][i,k]  = (fragParams[currentIM]['SDOF_Medians'][j,k]-fragParams[currentIM]['MDOF_Medians'][j,k])/fragParams[currentIM]['SDOF_Medians'][j,k]*100
            fragParams[currentIM]['Dispersion_Errors'][i,k] = (fragParams[currentIM]['SDOF_Betas'][j,k]-fragParams[currentIM]['MDOF_Betas'][j,k])/fragParams[currentIM]['SDOF_Betas'][j,k]*100


#%% Plot one-to-one charts for the fragility parameters deviations
### Median DS seismic intensities
for i, runs in enumerate(files):
    for j, currentIM in enumerate(im_labels):
        for k, damageThreshold in enumerate(arbitraryThresholds):
            plt.scatter(fragParams[currentIM]['SDOF_Medians'][i,k],fragParams[currentIM]['MDOF_Medians'][i,k],color=colors[k], marker = markers[j], alpha=0.5)
### Plot the one-to-one line
plt.plot([0,2],[0,2],'--', color = 'black')
plt.xlabel('SDOF Median DS Intensities [g]')
plt.ylabel('MDOF Median DS Intensities [g]')
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.ylim([0, 2])
plt.xlim([0, 2])
plt.legend(['DS0','DS1','DS2','DS3'])
plt.savefig(f'{outDir}/median_sdof_vs_mdof.png', dpi=600, format='png')
plt.show()
    
### Total dispersions
for i, runs in enumerate(files):
    for j, currentIM in enumerate(im_labels):
        for k, damageThreshold in enumerate(arbitraryThresholds):
            plt.scatter(fragParams[currentIM]['SDOF_Betas'][i,k],fragParams[currentIM]['MDOF_Betas'][i,k],color=colors[k], marker = markers[j], alpha=0.5)
### Plot the one-to-one line
plt.plot([0,2],[0,2],'--', color = 'black')
plt.xlabel(r'SDOF - $\beta_{total}$ ')
plt.ylabel(r'MDOF - $\beta_{total}$ ')
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.savefig(f'{outDir}/beta_sdof_vs_mdof.png', dpi=600, format='png')
plt.show()

#%% Plot one-to-one charts for the fragility parameters deviations for only efficient IMs

searchIndex = []
for i, runs in enumerate(files):
    ### Initialise the collection list
    searchEfficiencyList = []
    for j, currentIM in enumerate(im_labels):
        ### Fetch the index of the intensity measure with lowest beta value (most efficient)
        searchEfficiencyList.append(fragParams[currentIM]['MDOF_Betas'][i,0])
    searchIndex.append(searchEfficiencyList.index(min(searchEfficiencyList)))


for i in range(len(files)):
    for k, damageThreshold in enumerate(arbitraryThresholds):
        plt.scatter(fragParams[list(fragParams.keys())[searchIndex[i]]]['SDOF_Medians'][i,k],fragParams[list(fragParams.keys())[searchIndex[0]]]['MDOF_Medians'][i,k],color=colors[k], marker = markers[i], alpha=0.5)   
### Plot the one-to-one line
plt.plot([0,2],[0,2],'--', color = 'black')
plt.xlabel('SDOF Median DS Intensities [g]')
plt.ylabel('MDOF Median DS Intensities [g]')
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.ylim([0, 2])
plt.xlim([0, 2])
plt.legend(['DS0','DS1','DS2','DS3'])
plt.savefig(f'{outDir}/efficient_median_sdof_vs_mdof.png', dpi=600, format='png')
plt.show()

for i in range(len(files)):
    for k, damageThreshold in enumerate(arbitraryThresholds):
        plt.scatter(fragParams[list(fragParams.keys())[searchIndex[i]]]['SDOF_Betas'][i,k],fragParams[list(fragParams.keys())[searchIndex[0]]]['MDOF_Betas'][i,k],color=colors[k], marker = markers[i], alpha=0.5)   
### Plot the one-to-one line
plt.plot([0,2],[0,2],'--', color = 'black')
plt.xlabel(r'SDOF - $\beta_{total}$ ')
plt.ylabel(r'MDOF - $\beta_{total}$ ')
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.savefig(f'{outDir}/efficient_beta_sdof_vs_mdof.png', dpi=600, format='png')
plt.show()

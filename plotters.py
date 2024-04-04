##########################################################################
#                        VISUALISATION MODULES                           #
##########################################################################
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as image
            
def plotDemandProfiles(ansys_dict, nansys, saveOutput):
    """
    Plots the demand profiles associated with each record of cloud analysis

    Parameters
    ----------
    ansys_dict:                    dict                Dictionary with analysis output
    nansys:                         int                Number of analysis to show (i.e., best to set this to number of ground motion records)
    saveOutput:                   tuple                Tuple containing "True/False" as first item and save directory as second item
    
    Returns
    -------
    None.

    """
           
    ### Initialise the figure
    with cbook.get_sample_data('C:/Users/Moayad/Documents/GitHub/stickModel/gem_logo.png') as file:
        img = image.imread(file)
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    
    ### get number of storeys
    nst = ansys_dict['peak_drift_list'][0].shape[0]
    
    ### plot the results
    for i in range(nansys):
        x,y = duplicate_for_drift(ansys_dict['peak_drift_list'][i][:,0],ansys_dict['control_nodes'])
        ax1.plot([float(i)*100 for i in x], y, linewidth=2.5, linestyle = 'solid', color = 'gray', alpha = 0.7)
        ax1.set_xlabel(r'Peak Storey Drift, $\theta_{max}$ [%]')
        ax1.set_ylabel('Floor No.')
        ax1.grid(visible=True, which='major')
        ax1.grid(visible=True, which='minor')
        ax1.set_yticks(np.linspace(0,nst,nst+1))
        xticks = np.linspace(0,5,11)
        ax1.set_xticks(xticks, labels=xticks, minor=True)
        ax1.set_xlim([0, 5.0])
        plt.figimage(img, 60, 310, zorder=1, alpha=.7)

        ax2.plot([float(x)/9.81 for x in ansys_dict['peak_accel_list'][i][:,0]], ansys_dict['control_nodes'], linewidth=2.5, linestyle = 'solid', color = 'gray', alpha=0.7)
        ax2.set_xlabel(r'Peak Floor Acceleration, $a_{max}$ [g]')
        ax2.set_ylabel('Floor No.')
        ax2.grid(visible=True, which='major')
        ax2.grid(visible=True, which='minor')
        ax2.set_yticks(np.linspace(0,nst,nst+1))
        xticks = np.linspace(0,5,11)
        ax2.set_xticks(xticks, labels=xticks, minor=True)
        ax2.set_xlim([0, 5.0])         
    plt.show()
    ### save the output
    if saveOutput[0]:
        plt.savefig(f'{saveOutput[1]}/model_demands.png', dpi=1200, format='png')
           
def plotCloudAnalysis(im, edp, im_fitted, edp_fitted, im_label, saveOutput):
    """
    Plots the cloud analysis results 

    Parameters
    ----------
    im:                            list                Intensity measure levels of all the ground motion records from cloud analysis (e.g. PGA or Sa(T) of all records).
    edp:                           list                Resulting engineering demand parameter of all the ground motion records (e.g., peak storey drift).
    im_fitted:                     list                List of predicted intensity measures based on the fitted regression.
    edp_fitted:                    list                Range of sampled edp levels based on the minimum and maximum observed edps.
    im_label:                    string                Label for x-axis intensity measure.
    saveOutput:                   tuple                Tuple containing "True/False" as first item and save directory as second item
    
    Returns
    -------
    None.

    """
    ### Initialise the figure
    with cbook.get_sample_data('C:/Users/Moayad/Documents/GitHub/stickModel/gem_logo.png') as file:
        img = image.imread(file)
    ### Plot the cloud
    plt.scatter(edp, im, alpha = 0.5)
    plt.plot(edp_fitted, im_fitted, linewidth=5.0, linestyle = '-', color = 'black')
    plt.xlabel(r'Maximum Peak Storey Drift, $\theta_{max}$ [%]')
    plt.ylabel(im_label)
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([5e-3, 1e3])
    plt.xlim([np.min(edp), np.max(edp)])
    plt.figimage(img, 60, 310, zorder=1, alpha=.7)
    plt.show()
    ### save the output
    if saveOutput[0]:
        plt.savefig(f'{saveOutput[1]}/model_cloud_{im_label}.png', dpi=1200, format='png')
        
def plotFragilities(imls, poes, im_label, saveOutput):
    """
    Plots the fragility functions

    Parameters
    ----------
    imls:                          list                Range of intensity measure levels, output of "getDamageProbability"
    poes:                          list                List of probabilities of exceedances
    im_label:                    string                Label for x-axis intensity measure.
    saveOutput:                   tuple                Tuple containing "True/False" as first item and save directory as second item
    
    Returns
    -------
    None.

    """
    
    ### Initialise the figure
    colors = ['blue','green','yellow','red','black']
    with cbook.get_sample_data('C:/Users/Moayad/Documents/GitHub/stickModel/gem_logo.png') as file:
        img = image.imread(file)
    for i in range(len(poes)):
        plt.plot(imls, poes[i], linewidth=2.5, linestyle = '-', color = colors[i])
    plt.xlabel(im_label)
    plt.ylabel(r'Probability of Damage')
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor')
    plt.ylim([0, 1])
    plt.xlim([0, 2])
    plt.figimage(img, 60, 310, zorder=1, alpha=.7)
    plt.show()
    ### save the output
    if saveOutput[0]:
        plt.savefig(f'{saveOutput[1]}/model_fragilities_{im_label}.png', dpi=1200, format='png')

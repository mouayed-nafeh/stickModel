##########################################################################
#                        VISUALISATION MODULES                           #
##########################################################################
from utilities import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

### Function to create box plot inputs for storey drift demands
def duplicate_for_drift(drifts,control_nodes):
    x = []; y = []
    for i in range(len(control_nodes)-1):
        y.extend((float(control_nodes[i]),float(control_nodes[i+1])))
        x.extend((drifts[i],drifts[i]))
    y.append(float(control_nodes[i+1]))
    x.append(0.0)
    return x, y


def anim_model(nrha_disps, control_nodes):
    """
    Animates the Opensees model

    Parameters
    ----------
    None.
    
    Returns
    -------
    None.

    """

    ## The function 
    fig, ax = plt.subplots()
    ax.set(xlim=(-0.2, 0.2), ylim=(0, 50))
    ax.set_autoscale_on(False)
    
    # whirl Initialisation
    x,y = [],[]
    line, = ax.plot(x,y, color = 'green', marker = 'o')
    
    def animate(i):
        
        x,y = [],[]
    
        x.append(nrha_disps[i,:].tolist())
        y.append(control_nodes)
            
        line.set_xdata(x)
        line.set_ydata(y)
        
        return line,
    
    ani = FuncAnimation(fig, animate, frames=nrha_disps.shape[0], interval=1, blit = True, repeat=False)
    plt.show()
    
def plotCloudAnalysis(im, edp, regression_array, im_label, saveOutput):
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
    
    ### Plot the cloud
    plt.scatter(im, edp, alpha = 0.5)
    plt.plot(regression_array[:,0], regression_array[:,1], linewidth=5.0, linestyle = '-', color = 'black')
    plt.ylabel(r'Maximum Peak Storey Drift, $\theta_{max}$ [%]')
    plt.xlabel(im_label)
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor')
    plt.xscale('log')
    plt.yscale('log')
    ### save the output
    if saveOutput[0]:
        plt.savefig(f'{saveOutput[1]}/model_cloud_{im_label}.png', dpi=1200, format='png')
    plt.show()

def plotCloudAnalysis_withDamagePoints(im, edp, regression_array, damage_points, damage_thresholds, im_label, saveOutput):
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
    
    ### Plot the cloud
    colors = ['green','yellow','orange','red']
    DSs = ['Slight', 'Moderate', 'Extensive', 'Complete']
    plt.rcParams['axes.axisbelow'] = True
    plt.plot(regression_array[:,0], regression_array[:,1], linewidth=5.0, linestyle = '-', color = 'black')
    for i in range(len(damage_points)):
        plt.scatter(damage_points[i], damage_thresholds[i], color = colors[i], s = 100, alpha=1.0, zorder=2)
        plt.plot([1e-5, damage_points[i], damage_points[i]],[damage_thresholds[i],damage_thresholds[i], 1e-5], color = colors[i], linestyle = 'dashed', label = f'{DSs[i]}')
    plt.scatter(im, edp, alpha = 0.5)
    plt.ylabel(r'Maximum Peak Storey Drift, $\theta_{max}$ [rad]')
    plt.xlabel(im_label)
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlim([1e-5, 1e1])
    plt.ylim([1e-5, 1e1])
    
    ### save the output
    if saveOutput[0]:
        plt.savefig(f'{saveOutput[1]}/model_cloud_{im_label}.png', dpi=1200, format='png')
    plt.show()


def plotCloudAnalysis_mpfa(im, edp, regression_array, damage_points, damage_thresholds, im_label, saveOutput):
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
    
    ### Plot the cloud
    colors = ['red']
    DSs = ['Complete']
    plt.rcParams['axes.axisbelow'] = True
    plt.plot(regression_array[:,0], regression_array[:,1], linewidth=5.0, linestyle = '-', color = 'black')
    for i in range(len(damage_points)):
        plt.scatter(damage_points[i], damage_thresholds[i], color = colors[i], s = 100, alpha=1.0, zorder=2)
        plt.plot([1e-5, damage_points[i], damage_points[i]],[damage_thresholds[i],damage_thresholds[i], 1e-5], color = colors[i], linestyle = 'dashed', label = f'{DSs[i]}')
    plt.scatter(im, edp, alpha = 0.5)
    plt.ylabel(r'Maximum Peak Floor Acceleration, $a_{max}$ [g]')
    plt.xlabel(im_label)
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlim([1e-3, 10e1])
    plt.ylim([1e-3, 10e1])
    
    ### save the output
    if saveOutput[0]:
        plt.savefig(f'{saveOutput[1]}/model_cloud_{im_label}.png', dpi=1200, format='png')
    plt.show()

                   
def plotDemandProfiles(peak_drift_list, peak_accel_list, control_nodes, saveOutput):
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
    plt.figure(figsize=(12, 6))
    plt.rcParams['axes.axisbelow'] = True
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    
    ### get number of storeys
    nst = len(control_nodes)-1
    
    ### plot the results
    for i in range(len(peak_drift_list)):
        x,y = duplicate_for_drift(peak_drift_list[i][:,0],control_nodes)
        ax1.plot([float(i)*100 for i in x], y, linewidth=2.5, linestyle = 'solid', color = 'gray', alpha = 0.7)
        ax1.set_xlabel(r'Peak Storey Drift, $\theta_{max}$ [%]')
        ax1.set_ylabel('Floor No.')
        ax1.grid(visible=True, which='major')
        ax1.grid(visible=True, which='minor')
        ax1.set_yticks(np.linspace(0,nst,nst+1))
        xticks = np.linspace(0,5,11)
        ax1.set_xticks(xticks, labels=xticks, minor=True)
        ax1.set_xlim([0, 5.0])

        ax2.plot([float(x)/9.81 for x in peak_accel_list[i][:,0]], control_nodes, linewidth=2.5, linestyle = 'solid', color = 'gray', alpha=0.7)
        ax2.set_xlabel(r'Peak Floor Acceleration, $a_{max}$ [g]')
        ax2.set_ylabel('Floor No.')
        ax2.grid(visible=True, which='major')
        ax2.grid(visible=True, which='minor')
        ax2.set_yticks(np.linspace(0,nst,nst+1))
        xticks = np.linspace(0,5,11)
        ax2.set_xticks(xticks, labels=xticks, minor=True)
        ax2.set_xlim([0, 5.0])         
    if saveOutput[0]:
        plt.savefig(f'{saveOutput[1]}/model_demands.png', dpi=1200, format='png')
    plt.show()
           

def plotFragilities(fragility_array, im_label, saveOutput):
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
    colors = ['green','yellow','orange','red']
    DSs = ['Slight', 'Moderate', 'Extensive', 'Complete']
    for i in range(np.shape(fragility_array)[1]-1):
        plt.plot(fragility_array[:,0], fragility_array[:,i+1], linewidth=2.5, linestyle = '-', color = colors[i], label = DSs[i])
    plt.xlabel(im_label)
    plt.ylabel(r'Probability of Damage')
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor')
    plt.ylim([0, 1])
    plt.xlim([0, 5])
    plt.legend(loc='upper right')
    ### save the output
    if saveOutput[0]:
        plt.savefig(f'{saveOutput[1]}/model_fragility_{im_label}.png', dpi=1200, format='png')
    plt.show()

def plotFragilities_mpfa(fragility_array, im_label, saveOutput):
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
    colors = ['red']
    DSs = ['Complete']
    for i in range(np.shape(fragility_array)[1]-1):
        plt.plot(fragility_array[:,0], fragility_array[:,i+1], linewidth=2.5, linestyle = '-', color = colors[i], label = DSs[i])
    plt.xlabel(im_label)
    plt.ylabel(r'Probability of Damage')
    plt.grid(visible=True, which='major')
    plt.grid(visible=True, which='minor')
    plt.ylim([0, 1])
    plt.xlim([0, 5])
    plt.legend(loc='upper right')
    ### save the output
    if saveOutput[0]:
        plt.savefig(f'{saveOutput[1]}/model_fragility_{im_label}.png', dpi=1200, format='png')
    plt.show()


def plot_slf_rlz(psd_slf, psd_cache, pfa_slf, pfa_cache, rlz_to_consider = 100):
    """
    Plots the storey loss functions and the individual realizations

    Parameters
    ----------
    psd_slf:                 dictionary                Dictionary with the slfs of drift-sensitive components (output of create_slfs function) 
    psd_cache:               dictionary                Dictionary with the intermediate outputs of slf-generator for drift-sensitive components (output of create_slfs function) 
    pfa_slf:                 dictionary                Dictionary with the slfs of acceleration-sensitive components (output of create_slfs function) 
    pfa_cache:               dictionary                Dictionary with the intermediate outputs of slf-generator for acceleration-sensitive components (output of create_slfs function) 
    rlz_to_consider:              float                Number of realizations to plot (default set to 100 but will fail if total number of realizations is below)
    
    Returns
    -------
    None.

    """
    
    ### Initialise the figure
    with cbook.get_sample_data('C:/Users/Moayad/Documents/GitHub/stickModel/imgs/gem_logo.png') as file:
        img = image.imread(file)
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)

    ### Calculate total repair cost of all components
    cumSum_psd = []
    cumSum_pfa = []
    for i in range(len(list(psd_slf.keys()))):        
        cumSum_psd.append(np.max(psd_slf[list(psd_slf.keys())[i]]['slf']))
        cumSum_pfa.append(np.max(pfa_slf[list(pfa_slf.keys())[i]]['slf']))
    totalRepairCost = np.sum(cumSum_psd)+np.sum(cumSum_pfa)
        
    ### Fetch some params
    rlz = len(psd_cache[list(psd_cache.keys())[0]]['total_loss_storey'])
    categories = len(psd_slf)
    
    ### Initialise some colors
    colors = ['blue','green','yellow','red']
    
    ### Loop over drift-sensitive components
    for i in range(categories): 
        # Plot individual slfs for each component category
        edp_range = [x*100 for x in psd_slf[list(psd_slf.keys())[i]]['edp_range']]
        slf = psd_slf[list(psd_slf.keys())[i]]['norm_slf'][0]
        ax1.plot(edp_range, slf, color = colors[i], linewidth=8, label = psd_slf[list(psd_slf.keys())[i]]['category'][0])
        
        for j in range(rlz_to_consider):               
            # Plot scatter of individual realizations
            edp_range = [x*100 for x in psd_slf[list(psd_slf.keys())[i]]['edp_range']]
            realization = psd_cache[list(psd_cache.keys())[i]]['total_loss_storey'][j]
            maxval = totalRepairCost
            normRealization = [i/maxval for i in realization]   
            ax1.scatter(edp_range, normRealization, color = colors[i], alpha = 0.7)
        
        ax1.set_xlabel(r'Peak Storey Drift, $\theta$ [%]')
        ax1.set_ylabel('Loss Ratio')
        ax1.grid(visible=True, which='major')
        ax1.grid(visible=True, which='minor')
        ax1.legend(loc='upper left')
        ax1.set_xlim([0, 5])
        ax1.set_ylim([0, 1])

    ### Loop over acceleration-sensitive components
    for i in range(categories):
        # Plot individual slfs for each component category
        edp_range = pfa_slf[list(pfa_slf.keys())[i]]['edp_range']
        slf = pfa_slf[list(pfa_slf.keys())[i]]['norm_slf'][0]
        ax2.plot(edp_range, slf, color = colors[i], linewidth=8, label = pfa_slf[list(pfa_slf.keys())[i]]['category'][0])
        
        for j in range(rlz_to_consider):
            # Plot scatter of individual realizations
            edp_range = pfa_slf[list(pfa_slf.keys())[i]]['edp_range']
            realization = pfa_cache[list(pfa_cache.keys())[i]]['total_loss_storey'][j]
            maxval = totalRepairCost
            normRealization = [i/maxval for i in realization]   
            ax2.scatter(edp_range, normRealization, color = colors[i], alpha = 0.7)
        
        ax2.set_xlabel(r'Peak Floor Acceleration, $a_{max}$ [g]')
        ax2.set_ylabel('Loss Ratio')
        ax2.grid(visible=True, which='major')
        ax2.grid(visible=True, which='minor')
        ax2.legend(loc='upper left')
        ax2.set_xlim([0, 5])
        ax2.set_ylim([0, 1])
        
    plt.show()

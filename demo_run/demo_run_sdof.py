
# Change this to the local directory of the repository (where the repository is cloned)
repoDir = 'C:/Users/Moayad/Documents/GitHub/stickModel'
capCurvesDir= f'{repoDir}/demo_run/capacities'          # equivalent sdof capacity curves directory (no need to change, unless files relocated to different directory)
gmDir  = f'{repoDir}/demo_run/records_v2'                  # ground-motion records directory (no need to change, unless files relocated to different directory)
outDir = f'{repoDir}/demo_run/sdof_old_records_without_load'                   # output files directory (no need to change, unless files need to be relocated to different directory) 

### Import toolkit libraries
import sys
sys.path.insert(1, f'{repoDir}')
from stickModel import stickModel
from utils import *
from units import *
from postprocessors import *
from plotters import *
from im_calculator import *
from calibration import *

### Import other libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from math import sqrt, pi

# Create the output directory
if not os.path.exists(f'{outDir}'):
    os.makedirs(f'{outDir}')

#%% User Input 

# Flag to import or run the next code section (set to False if first run)
import_ims = True

#%% Get intensity measure levels

### Define intensity measure labels based on their hierarchy in the analysis dictionary
im_labels = ['PGA','SA(T=0.3s)','SA(T=0.6s)','SA(T=1.0s)',
              'Saavg(T=0.3s)','Saavg(T=0.6s)','Saavg(T=1.0s)']

if import_ims is False:
    ### Create a dictionary for storage
    ims = {}
    for i, labels in enumerate(im_labels):
        ims[labels] = []
    
    ### Fetch the ground-motion records files
    gmrs = sorted_alphanumeric(os.listdir(f'{gmDir}/gmrs'))   # Sort the ground-motion records alphanumerically
    
    for i in range(len(gmrs)):    
        fnames = [f'{gmDir}/gmrs/{gmrs[i]}']                                      # Ground-motion record names
        fdts = f'{gmDir}/dts/dts_{i}.csv'                                         # Ground-motion time-step names 
        dt_gm = pd.read_csv(fdts)[pd.read_csv(fdts).columns[0]].loc[0]            # Ground-motion time-step
        im = intensityMeasure(pd.read_csv(fnames[0]).to_numpy().flatten(),dt_gm)  # Initialise the intensityMeasure object
        
        ### Get acceleration response spectrum
        prd, sas = im.get_spectrum()
        
        ### Calculate the intensity measures 
        for j in range(len(im_labels)):
            if j==0:
                ims[im_labels[j]].append(im.get_sa(prd,sas,0.0))
            elif j==1:
                ims[im_labels[j]].append(im.get_sa(prd,sas,0.3))
            elif j==2:
                ims[im_labels[j]].append(im.get_sa(prd,sas,0.6))
            elif j==3:
                ims[im_labels[j]].append(im.get_sa(prd,sas,1.0))
            elif j==4:
                ims[im_labels[j]].append(im.get_saavg(prd,sas,0.3))
            elif j==5:
                ims[im_labels[j]].append(im.get_saavg(prd,sas,0.6))
            elif j==6:
                ims[im_labels[j]].append(im.get_saavg(prd,sas,1.0))
                
    export_to_pkl(f'{outDir}/ims_v2.pkl',ims)

elif import_ims: 
    ### Load the intensity measure file
    ims = import_from_pkl(f'{outDir}/ims_v2.pkl')

#%% 

buildingClasses = [x.split('.')[0] for x in sorted_alphanumeric(os.listdir(f'{capCurvesDir}'))]  # Fetch the building classes list

for i in np.arange(0,len(buildingClasses)): # This will run the entire set of ten low-code low-ductility infilled RC models 
    
    ###########################################################################################################
    ###                                                                                                     ###
    ###                                       1.PREPROCESS THE INPUTS                                       ### 
    ###                                                                                                     ###
    ###########################################################################################################

    ### 1.1) Get the current building class and print the run
    currentBuildingClass = buildingClasses[i]
    
    print('Current Building Class:',currentBuildingClass)
    print('================================================================')
    print('============== ANALYSING: {:d} out of {:d} =================='.format(i+1, len(buildingClasses)))
    print('================================================================')

    ### 1.2) Create output directory
    currentAnsysOutDir = f'{outDir}/{currentBuildingClass}'
    if not os.path.exists(f'{currentAnsysOutDir}'):
        os.makedirs(f'{currentAnsysOutDir}')
    
    ### 1.3) Initialise MDOF storage lists
    sdof_coll_index_list = []               # List for collapse index
    sdof_peak_disp_list  = []               # List for peak floor displacement (returns all peak values along the building height)
    sdof_peak_drift_list = []               # List for peak storey drift (returns all peak values along the building height)
    sdof_peak_accel_list = []               # List for peak floor acceleration (returns all peak values along the building height)
    sdof_max_peak_drift_list = []           # List for maximum peak storey drift (returns the maximum value) 
    sdof_max_peak_drift_dir_list = []       # List for maximum peak storey drift directions
    sdof_max_peak_drift_loc_list = []       # List for maximum peak storey drift locations
    sdof_max_peak_accel_list = []           # List for maximum peak floor acceleration (returns the maximum value)
    sdof_max_peak_accel_dir_list = []       # List for maximum peak floor acceleration directions 
    sdof_max_peak_accel_loc_list = []       # List for maximum peak floor acceleration locations 

    ### 1.4) Load and extract the building class info
    classInfo = pd.read_csv(f'{capCurvesDir}/in_plane_capacity_parameters_table.csv')    
    nst_mdof     = classInfo['Number_storeys'].loc[classInfo['Building_class']==currentBuildingClass].item()
    gamma        = classInfo['Participation_factor'].loc[classInfo['Building_class']==currentBuildingClass].item()
    storeyHeight = classInfo['Storey_height'].loc[classInfo['Building_class']==currentBuildingClass].item()

    nst_sdof     = 1
    flh_sdof     = [nst_mdof*storeyHeight/gamma]*nst_sdof
    flm_sdof     = [1.0]*nst_sdof
    
    ### 1.5) Import the equivalent SDOF capacity array 
    sdofCapArray = np.array(pd.read_csv(f'{capCurvesDir}/{currentBuildingClass}.csv', header = None))[1:,:]
    rows, columns = np.shape(sdofCapArray)
    stD_sdof = np.zeros([nst_sdof,rows])
    stF_sdof = np.zeros([nst_sdof,rows])
    for j in range(nst_sdof):
        # get the displacement or spectral displacement arrays at each storey
        stF_sdof[j,:] = sdofCapArray[:,1].transpose()*units.g
        # get the force or spectral acceleration arrays at each storey
        stD_sdof[j,:] = sdofCapArray[:,0].transpose()   
    
    ###########################################################################################################
    ###                                                                                                     ###
    ###                                      2.COMPILE THE SDOF MODEL                                       ### 
    ###                                                                                                     ###
    ###########################################################################################################

    ### 2.1) Loop over ground-motion records, compile SDOF model and run NLTHA
    gmrs = sorted_alphanumeric(os.listdir(f'{gmDir}/gmrs'))                         # Sort the ground-motion records alphanumerically
    for k in range(len(gmrs)):
        model = stickModel(nst_sdof,flh_sdof,flm_sdof,stD_sdof,stF_sdof)            # Build the model
        model.mdof_initialise()                                                     # Initialise the domain
        model.mdof_nodes()                                                          # Construct the nodes
        model.mdof_fixity()                                                         # Set the boundary conditions 
       #model.mdof_loads()                                                          # Assign the loads
        model.mdof_material()                                                       # Assign the nonlinear storey material
        if k==0:
            model.plot_model()                                                      # Visualise the model
        else: 
            pass
        model.do_gravity_analysis()                                                 # Do gravity analysis
        T = model.do_modal_analysis(num_modes = 1)                                  # Do modal analysis and get period of vibration
        
        ### Define ground motion objects
        fnames = [f'{gmDir}/gmrs/{gmrs[k]}']                                        # Ground-motion record names
        fdts = f'{gmDir}/dts/dts_{k}.csv'                                           # Ground-motion time-step names 
        dt_gm = pd.read_csv(fdts)[pd.read_csv(fdts).columns[0]].loc[0]              # Ground-motion time-step
        t_max = pd.read_csv(fdts)[pd.read_csv(fdts).columns[0]].iloc[-1]            # Ground-motion duration
        
        ### Define analysis params and do NLTHA
        dt_ansys = dt_gm                                                            # Set the analysis time-step
        sf = units.g                                                                # Set the scaling factor (if records are in g, a scaling factor of 9.81 m/s2 must be used to be consistent with opensees) 
        control_nodes, coll_index, peak_drift, peak_accel, max_peak_drift, max_peak_drift_dir, max_peak_drift_loc, max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp = model.do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, pflag=True, 
                                                                                                                                                                                                              ansys_soe='BandGeneral', 
                                                                                                                                                                                                              constraints_handler='Plain', numberer='RCM', 
                                                                                                                                                                                                              test_type='NormDispIncr', init_tol=1.0e-6, init_iter=50, 
                                                                                                                                                                                                              algorithm_type='Newton')    
        ### Store the analysis
        sdof_coll_index_list.append(coll_index)
        sdof_peak_drift_list.append(peak_drift)
        sdof_peak_accel_list.append(peak_accel)
        sdof_peak_disp_list.append(peak_disp)
        sdof_max_peak_drift_list.append(max_peak_drift)
        sdof_max_peak_drift_dir_list.append(max_peak_drift_dir)
        sdof_max_peak_drift_loc_list.append(max_peak_drift_loc)
        sdof_max_peak_accel_list.append(max_peak_accel)
        sdof_max_peak_accel_dir_list.append(max_peak_accel_dir)
        sdof_max_peak_accel_loc_list.append(max_peak_accel_loc)
    
        print('================================================================')
        print('============== ANALYSIS COMPLETED: {:d} out {:d} =================='.format(k+1, len(gmrs)))
        print('================================================================')
        
    ###########################################################################################################
    ###                                                                                                     ###
    ###                                          3.EXPORT THE RESULTS                                       ### 
    ###                                                                                                     ###
    ###########################################################################################################
    
    ### 3.1) Store the analysis results
    ansys_dict = {}
    labels = ['T','control_nodes', 'sdof_coll_index_list',
              'sdof_peak_drift_list','sdof_peak_accel_list','sdof_max_peak_drift_list',
              'sdof_max_peak_drift_dir_list', 'sdof_max_peak_drift_loc_list','sdof_max_peak_accel_list',
              'sdof_max_peak_accel_dir_list','sdof_max_peak_accel_loc_list','sdof_peak_disp_list']
        
    for l, label in enumerate(labels):
        ansys_dict[label] = vars()[f'{label}']
    
    ### 3.2) Export to pickle
    export_to_pkl(f'{outDir}/analysis_{currentBuildingClass}.pkl', ansys_dict)


        
        

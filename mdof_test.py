# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:36:52 2023

@author: Moayad
"""
import pandas as pd
import matplotlib.pyplot as plt
from mdof_models import *
from mdof_plotting import *
from mdof_analyses import *
from mdof_units import *
from mdof_utils import *
import openseespy.opensees as ops

test_directory = 'C:/Users/Moayad/Desktop/python/stick-model/test'

# define arbitrary input information
ops.wipe()

nst = 2
floor_height = [2.8,3]
floor_mass = [400,1000]
sqA = 2000;  # floor area in m2

# modal analysis parameters
Nmodes = nst

# pushover analysis parameters
ref_disp = 0.1
disp_scale_factor = 10
push_dir = 2

# storey capacity parameters
F = [10000,10000]
D = [0.01, 0.03]

#### initialise the model
model = mdof_model()

#### initialise the nodes
mdof_nodes(nst,floor_height,floor_mass)

#### define boundary conditions (fixities)
mdof_fixity()

#### define elements (this is where the future integration of Luis' capacity curves would go)
mdof_storeyCap(F,D,'bilinear')

#### define loads
mdof_loads(sqA)

# do gravity analysis
do_gravity_analysis()

#### do modal analysis
T, omega = do_modal_analysis(Nmodes, pflag=False)
print(f"\n----------------- Processing: Period of Vibration -----------------")
print(f"\n---------------------------- X-Direction --------------------------")
print(f"Fundamental Period (T\u2081)= {T[0]:,.3f}s")
print(f"Natural Frequency (\N{GREEK SMALL LETTER OMEGA})= {omega[0]:,.3f}rad/s")
print(f"\n---------------------------- Y-Direction --------------------------")
print(f"Fundamental Period (T\u2081)= {T[1]:,.3f}s")
print(f"Natural Frequency (\N{GREEK SMALL LETTER OMEGA})= {omega[1]:,.3f}rad/s")
print(f"\n-------------------------------------------------------------------")


#### Wipe the existing modal analysis object
ops.wipeAnalysis()

print(f"\n--------------- Processing: Static Pushover Analysis --------------")
# do pushover analysis
spo_disp, spo_rxn = do_spo_analysis(ref_disp, disp_scale_factor, push_dir, pflag=True, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-8, init_iter=1000, algorithm_type='KrylovNewton')
print(f"\n-------------------------------------------------------------------")

# plot spo results
plt.plot(spo_disp, spo_rxn/(np.sum(floor_mass)*9.81), color = 'blue',linestyle='solid')
plt.xlabel("Top Displacement, $\delta$ [m]")
plt.ylabel("Base Shear, V [kN]")
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.xlim([0,0.5])
plt.ylim([0,1])
#plt.savefig(f'{test_directory}/spo_{push_dir}.png', dpi=300)
plt.show()

ops.wipeAnalysis()

# visualise the model
plot_model_elements(display_info=True)

# do time-history analysis
fnames = [f'{test_directory}/record.txt']

# plot the ground motion
accel_file= f'{test_directory}/record.txt'
acc = read_one_column_file(accel_file)
#acc = [x/9.81 for x in acc]
t = np.arange(0.0, 300.01, 0.01).tolist()

# plot time-history
plt.plot(t, acc, color = 'blue',linestyle='solid')
plt.xlabel("Time, t [s]")
plt.ylabel("Acceleration, a [g]")
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.xlim([0,100])
plt.ylim([-2,2])
plt.show()

# do time-history analysis
control_nodes = [0,1,2]; dt_gm = 0.01; t_max = 300.00 ; dt_ansys = 0.05; sf = 1.00
coll_index, peak_drift, peak_accel, max_peak_drift, max_peak_drift_dir, max_peak_drift_loc, max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp = do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, 5,control_nodes)
    

# plot acceleration profiles (in x)
plt.plot(peak_accel[:,0], control_nodes, color = 'blue',linestyle='solid')
plt.xlabel("Peak Floor Acceleration, PFA [g]")
plt.ylabel("Floor No.")
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.xlim([0,1])
plt.ylim([0,np.max(control_nodes)])
plt.show()

# plot drift profiles (in x)

bplot_drift, bplot_nodes = duplicate_for_drift(peak_drift[:,0], control_nodes)
plt.plot(bplot_drift, bplot_nodes, color = 'blue',linestyle='solid')
plt.xlabel("Peak Storey Drift, PSD [%]")
plt.ylabel("Floor No.")
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.show()

# plot drift profiles (in y)

bplot_drift, bplot_nodes = duplicate_for_drift(peak_drift[:,1], control_nodes)
plt.plot(bplot_drift, bplot_nodes, color = 'blue',linestyle='solid')
plt.xlabel("Peak Storey Drift, PSD [%]")
plt.ylabel("Floor No.")
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.show()




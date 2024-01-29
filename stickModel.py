# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:26:11 2024

@author: Moayad
"""

## load dependencies
import openseespy.opensees as ops
import pandas as pd
import numpy as np
import os
from mdof_units import g
from utils import *           

class stickModel():

    def __init__(self, nst, flh, flm, fll, fla, structTypo, gitDir):
        """
        Stick Modelling and Processing Tool
        :param nst:         int                Number of storeys
        :param flh:        list               List of floor heights in metres (e.g. [2.5, 3.0])
        :param flm:        list               List of floor masses in tonnes (e.g. [1000, 1200])
        :param fll:        list               List of floor loads in kN/m2 (e.g. [5.0, 5.0])
        :param fla:        list               List of floor areas in m2 (e.g. [350, 350])
        :param structTypo:  str               String from GEM taxonomy (e.g. 'CR_LDUAL+CDH+DUH_H1')
        :param gitDir:      str               String corresponding to the GitHub file directory

        """
                
        # run some tests
        if len(flh)!=nst or len(flm)!=nst or len(fll)!=nst or len(fla)!=nst:
            raise ValueError('Number of entries exceed the number of storeys')
        
        self.nst = nst; self.flh = flh; self.flm = flm; self.fll = fll; self.fla = fla
        self.structTypo = structTypo
        
    def mdof_initialise(self):
        """
        Initialising the definition of an MDOF system

        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """    
        # set model builder
        ops.wipe() # wipe existing model
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        
    def mdof_nodes(self):
        """
        Initialising the definition of MDOF nodes and their corresponding mass
    
        Parameters
        ----------
        None.
         
        Returns
        -------
        None.
    
        """    
        # wipe everything before initialising model
        ops.wipe()
        
        # define base node (tag = 0)
        ops.node(0, *[0.0, 0.0, 0.0])
        # define floor nodes (tag = 1+)
        i = 1
        current_height = 0.0
        while i <= self.nst:
            nodeTag = i
            current_height = current_height + self.flh[i-1]
            current_mass = self.flm[i-1]
            coords = [0.0, 0.0, current_height]
            masses = [current_mass, current_mass, current_mass, current_mass, current_mass, current_mass]
            ops.node(nodeTag,*coords)
            ops.mass(nodeTag,*masses)
            i+=+1

    def mdof_fixity(self):
        """
        Initialising the definition of MDOF nodes' boundary conditions
    
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
    
        """    
        # get list of model nodes
        nodeList = ops.getNodeTags()
        # impose boundary conditions
        for i in nodeList:            
            if i==0:
                ops.fix(i,1,1,1,1,1,1)
            else:
                ops.fix(i,0,0,1,0,0,0)
    
    def mdof_loads(self):
        """
        Initialising the definition of MDOF loads based on floor mass

        Parameters
        ----------
        None.
                    
        Returns
        -------
        None.

        """        
        # get corresponding floor mass
        massFl = []
        for i in range(self.nst):
            massFl.append(self.fla[i]*self.fll[i])
        # get list of model nodes
        nodeList = ops.getNodeTags()
        # create a plain load pattern with a linear timeseries
        ops.timeSeries('Linear', 101)
        ops.pattern('Plain',101,101)
        # load the nodes
        for i in nodeList:
            if i==0:
                pass
            else:
                ops.load(i,0.0,0.0,-massFl[i], 0.0, 0.0, 0.0)

    def mdof_material(self):
        """
        Initialising the definition of MDOF storey material model
    
        Parameters
        ----------
        eleTag: int
         element tag
        matDef: string
         definition of material backbone. Supported definitions are:
             'bilinear', 'trilinear', 'quadrilinear'
        F: list of floats
         array of strength points
        D: list of floats
         array of deformation points
        
        Returns
        -------
        None.
    
        """
        matDir = f'{self.gitDir}/raw/in/'   
        
        # get number of zerolength elements required
        nodeList = ops.getNodeTags()
        numEle = len(nodeList)-1
        
        for i in self.nst:
            
            # define the material tag associated with each storey
            mat1Tag = int(f'{i}00')
            mat2Tag = int(f'{i}01')
            
            # get the backbone curve definition
            D = list(pd.read_csv(f'{self.gitDir}/{self.structTypo}.csv').iloc[:,0])
            F = list(pd.read_csv(f'{self.gitDir}/{self.structTypo}.csv').iloc[:,1])
    
            # create rigid elastic materials for the restrained dofs
            rigM = int(f'1{i}02')
            ops.uniaxialMaterial('Elastic', rigM, 1e6)
    
            # create the material
            createPinching4(mat1Tag, F,D)
            
            # create a min-max material to define ultimate displacement and overlay it to the previous material
            ops.uniaxialMaterial('MinMax', mat2Tag, mat1Tag, '-min', -D[-1], '-max', D[-1])
            
            # aggregate all material tags in one
            matTags = [mat2Tag, mat2Tag, rigM, rigM, rigM, rigM]            


            # define the connectivity parameters
            eleTag = int(f'{i}02')
            eleNodes = [i, i+1]
                
            # create the element
            ops.element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, '-dir', [1,2,3,4,5,6])
            i=i+1

        
        
##########################################################################
#                             ANALYSIS MODULES                           #
##########################################################################
    def do_gravity_analysis(self, nG=100,system='UmfPack',constraints='Transformation',
                            numberer='RCM',test='NormDispIncr',tol = 1.0e-6, iters = 500, algorithm='Newton' ,
                            integrator='LoadControl',analysis='Static'):        
        
        ops.system(system) # creates the system of equations, a sparse solver with partial pivoting
        ops.constraints(constraints) # creates the constraint handler, the transformation method
        ops.numberer(numberer) # creates the DOF numberer, the reverse Cuthill-McKee algorithm
        ops.test(test, tol, iters, 3) # creates the convergence test
        ops.algorithm(algorithm) # creates the solution algorithm, a Newton-Raphson algorithm
        ops.integrator(integrator, (1/nG)) # creates the integration scheme
        ops.analysis(analysis) # creates the analysis object
        ops.analyze(nG) # perform the gravity load analysis
        ops.loadConst('-time', 0.0)
    
    def do_modal_analysis(self, num_modes=2, solver = '-genBandArpack', pflag=True):
    
        # get fundamental frequency and fundamental period        
        omega = np.power(ops.eigen(solver, num_modes), 0.5)
        T = 2.0*np.pi/omega
        # get list of model nodes
        nodeList = ops.getNodeTags()            
        # print report
        if pflag == True:
            ops.modalProperties('-print')
        else:
            pass
            
    def do_spo_analysis(ref_disp, disp_scale_factor, push_dir, pflag=False, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-8, init_iter=1000, algorithm_type='KrylovNewton'):
                
        # apply the load pattern
        ops.timeSeries("Linear", 1) # create timeSeries
        ops.pattern("Plain", 1, 1) # create a plain load pattern
        
        # define control nodes
        nodeList = ops.getNodeTags()
        control_node = nodeList[-1]
        pattern_nodes = nodeList[1:]
        rxn_nodes = [nodeList[0]]
        
        # we can integrate modal patterns, inverse triangular, etc.
        for i in np.arange(len(pattern_nodes)):
            if push_dir == 1:
                ops.load(pattern_nodes[i], nodeList[i]/len(pattern_nodes), 0.0, 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 2:
                ops.load(pattern_nodes[i], 0.0, nodeList[i]/len(pattern_nodes), 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 3:
                ops.load(pattern_nodes[i], 0.0, 0.0, nodeList[i]/len(pattern_nodes), 0.0, 0.0, 0.0)
                
            
        # Set up the initial objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)        
        ops.algorithm(algorithm_type)
        
        # Set the integrator
        target_disp = ref_disp*disp_scale_factor
        delta_disp = target_disp/(1.0*num_steps)
        ops.integrator('DisplacementControl', control_node, push_dir, delta_disp)
        ops.analysis('Static')
        
        # Give some feedback if requested
        if pflag is True:
            print('Pushover analysis of node ' + str(control_node) + ' to ' + str(target_disp))
    
        # Set up the analysis
        ok = 0
        step = 1
        loadf = 1.0
        
        spo_rxn = np.array([0.])
        spo_disp = np.array([0.])
        
        while step <= num_steps and ok == 0 and loadf > 0:
            
            # Push it by one step
            ok = ops.analyze(1)
    
            # If the analysis fails, try the following changes to achieve convergence
            if ok != 0:
                print('FAILED: Trying relaxing convergence...')
                ops.test(test_type, init_tol*0.01, init_iter)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED: Trying relaxing convergence with more iterations...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED: Trying relaxing convergence with more iteration and Newton with initial then current...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initialThenCurrent')
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED: Trying relaxing convergence with more iteration and Newton with initial...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initial')
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED: Attempting a Hail Mary...')
                ops.test('FixedNumIter', init_tol*0.01, init_iter*10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            
            # This feature of disabling the possibility of having a negative loading has been included.
            # This has been adapted from a similar script by Prof. Garbaggio
            loadf = ops.getTime()
                
            # Give some feedback if requested
            if pflag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print('Currently pushed node ' + str(control_node) + ' to ' + str(curr_disp) + ' with ' + str(loadf))
                       
            # Increment to the next step
            step += 1
            
            # Get the results
            spo_disp = np.append(spo_disp, ops.nodeDisp(control_node, push_dir))
            
            ops.reactions()
            temp = 0
            for n in rxn_nodes:
                temp += ops.nodeReaction(n, push_dir)
            spo_rxn = np.append(spo_rxn, -temp)
            
                
        # Give some feedback on what happened  
        if ok != 0:
            print('------ ANALYSIS FAILED --------')
        elif ok == 0:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')
        
        if loadf < 0:
            print('Stopped because of load factor below zero')
            
        return spo_disp, spo_rxn

    def do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, drift_limit, 
                         control_nodes, pflag=False, ansys_soe='BandGeneral', 
                         constraints_handler='Transformation', numberer='RCM', 
                         test_type='EnergyIncr', init_tol=1.0e-8, init_iter=1000, 
                         algorithm_type='KrylovNewton'):
        """
        Method to carry out a NRHA of a 3D model
    
        Parameters
        ----------
        *fnames : list (str)
            List of the filepaths to the ground motions to be applied in the X Y and Z. At least the X direction is required.
        dt_gm : float
            time-step of the ground motions.
        sf : float
            scale factor to be applied to the records. THIS IS USUALLY GRAVITY.
        t_max : float
            duration of the record.
        dt_ansys : float
            time-step at which to conduct the analysis. Typically smaller than the record's.
        drift_limit : float
            Drift limit at which to stop the analysis (IN RADIANS).
        control_nodes : list (int)
            List of the nodes in sequential order to monitor.
        pflag : bool, optional
            Whether to print information on screen or not. The default is False.
        ansys_soe : TYPE, optional
            DESCRIPTION. The default is 'BandGeneral'.
        constraints_handler : TYPE, optional
            DESCRIPTION. The default is 'Transformation'.
        numberer : TYPE, optional
            DESCRIPTION. The default is 'RCM'.
        test_type : TYPE, optional
            DESCRIPTION. The default is 'EnergyIncr'.
        init_tol : TYPE, optional
            DESCRIPTION. The default is 1.0e-8.
        init_iter : TYPE, optional
            DESCRIPTION. The default is 1000.
        algorithm_type : TYPE, optional
            DESCRIPTION. The default is 'KrylovNewton'.
    
        Raises
        ------
        ValueError
            DESCRIPTION.
    
        Returns
        -------
        coll_index : int
            Collapse index (-1 for non-converged, 0 for stable, 1 for collapsed).
        peak_drift : numpy array
            array of the peak drifts at each storey (determined as the storeys between the listed control_nodes) in the directions excited. IN RADIANS
        peak_accel : numpy array
            array of the peak accelerations at each floor (determined at the control_nodes) in the directions excited. IN TERMS OF G
        max_peak_drift : float
            max in either dirction of peak_drift.
        max_peak_drift_dir : str
            direction of peak_drift.
        max_peak_drift_loc : int
            storey location of peak_drift.(Storeys = 1, 2, 3,...)
        max_peak_accel : float
            max in either dirction of peak_accel.
        max_peak_accel_dir : str
            direction of peak_accel.
        max_peak_accel_loc : ind
            storey location of peak_accel.(Floors = 0(G), 1, 2, 3,...)
    
        """
        # define control nodes
        #nodeList = ops.getNodeTags()
        #control_nodes = nodeList[-1]
        
        
        # Define the timeseries and patterns first
        if len(fnames) > 0:
            nrha_tsTagX = 11
            nrha_pTagX = 11
            ops.timeSeries('Path', nrha_tsTagX, '-dt', dt_gm, '-filePath', fnames[0], '-factor', sf) 
            ops.pattern('UniformExcitation', nrha_pTagX, 1, '-accel', nrha_tsTagX)
            ops.recorder('Node', '-file', "floor_accel_X.txt", '-timeSeries', nrha_tsTagX, '-node', *control_nodes, '-dof', 1, 'accel')
        if len(fnames) > 1:
            nrha_tsTagY = 22
            nrha_pTagY = 22
            ops.timeSeries('Path', nrha_tsTagY, '-dt', dt_gm, '-filePath', fnames[1], '-factor', sf) 
            ops.pattern('UniformExcitation', nrha_pTagY, 2, '-accel', nrha_tsTagY)
            ops.recorder('Node', '-file', "floor_accel_Y.txt", '-timeSeries', nrha_tsTagY, '-node', *control_nodes, '-dof', 2, 'accel')
        if len(fnames) > 2:
            nrha_tsTagZ = 33
            nrha_pTagZ = 33
            ops.timeSeries('Path', nrha_tsTagZ, '-dt', dt_gm, '-filePath', fnames[2], '-factor', sf) 
            ops.pattern('UniformExcitation', nrha_pTagZ, 3, '-accel', nrha_tsTagZ)
        
        # Set up the initial objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)        
        ops.algorithm(algorithm_type)
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')
        
        # Set up analysis parameters
        coll_index = 0   # Initially define the collapse index (-1 for non-converged, 0 for stable, 1 for collapsed)
        control_time = 0.0 
        ok = 0 # Set the convergence to 0 (initially converged)
        
        # Parse the data about the building
        top_nodes = control_nodes[1:]
        bottom_nodes = control_nodes[0:-1]
        h = []
        for i in np.arange(len(top_nodes)):
            topZ = ops.nodeCoord(top_nodes[i], 3)
            bottomZ = ops.nodeCoord(bottom_nodes[i], 3)
            dist = topZ - bottomZ
            if dist == 0:
                print("WARNING: Zero length found in drift check, using very large distance 1e9 instead")
                h.append(1e9)
            else:
                h.append(dist)
        
        # Create some arrays to record to
        peak_disp = np.zeros((len(control_nodes), 2))
        peak_drift = np.zeros((len(top_nodes), 2))
        peak_accel = np.zeros((len(top_nodes)+1, 2))
        
        # Run the actual analysis
        while coll_index == 0 and control_time <= t_max and ok == 0:
            ok = ops.analyze(1, dt_ansys)
            control_time = ops.getTime()
            
            if pflag is True:
                print('Completed {:.3f}'.format(control_time) + ' of {:.3f} seconds'.format(t_max) )
        
            # If the analysis fails, try the following changes to achieve convergence
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying reducing time-step in half...')
                ok = ops.analyze(1, 0.5*dt_ansys)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying reducing time-step in quarter...')
                ok = ops.analyze(1, 0.25*dt_ansys)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iterations...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iteration and Newton with initial then current...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initialThenCurrent')
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iteration and Newton with initial...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initial')
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Attempting a Hail Mary...')
                ops.test('FixedNumIter', init_tol*0.01, init_iter*10)
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                
            # Game over......
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Exiting analysis...')
                coll_index = -1
        
            # For each of the nodes to monitor, get the current drift
            for i in np.arange(len(top_nodes)):
                
                # Get the current storey drifts - absolute difference in displacement over the height between them
                curr_drift_X = np.abs(ops.nodeDisp(top_nodes[i], 1) - ops.nodeDisp(bottom_nodes[i], 1))/h[i]
                curr_drift_Y = np.abs(ops.nodeDisp(top_nodes[i], 2) - ops.nodeDisp(bottom_nodes[i], 2))/h[i]
        
                # Check if the current drift is greater than the previous peaks at the same storey
                if curr_drift_X > peak_drift[i, 0]:
                    peak_drift[i, 0] = curr_drift_X
                
                if curr_drift_Y > peak_drift[i, 1]:
                    peak_drift[i, 1] = curr_drift_Y
                    
            # For each node to monitor, get is absolute displacement
            for i in np.arange(len(control_nodes)):
                curr_disp_X = np.abs(ops.nodeDisp(control_nodes[i], 1))
                curr_disp_Y = np.abs(ops.nodeDisp(control_nodes[i], 2))
                
                # Check if the current drift is greater than the previous peaks at the same storey
                if curr_disp_X > peak_disp[i, 0]:
                    peak_disp[i, 0] = curr_disp_X
                
                if curr_disp_Y > peak_disp[i, 1]:
                    peak_disp[i, 1] = curr_disp_Y
            
            # Check that the maximum in either direction hasn't exceeded the drift limit
            if np.max(peak_drift) > drift_limit:
                coll_index = 1
        
        # Now that the analysis is finished, get the maximum in either direction and report the location also
        max_peak_drift = np.max(peak_drift)
        ind = np.where(peak_drift == max_peak_drift)
        if ind[1][0] == 0:
            max_peak_drift_dir = 'X'
        elif ind[1][0] == 1:
            max_peak_drift_dir = 'Y'
        max_peak_drift_loc = ind[0][0]+1
        
        # Get the floor accelerations. Need to use a recorder file because a direct query would return relative values
        ops.wipe() # First wipe to finish writing to the file
        if len(fnames) > 0:
            temp1 = np.transpose(np.max(np.abs(np.loadtxt("floor_accel_X.txt")), 0))/g
            peak_accel[:,0] = temp1
            os.remove("floor_accel_X.txt")
        if len(fnames) > 1:
            
            temp1 = np.transpose(np.max(np.abs(np.loadtxt("floor_accel_X.txt")), 0))/g
            temp2 = np.transpose(np.max(np.abs(np.loadtxt("floor_accel_Y.txt")), 0))/g
            peak_accel = np.stack([temp1, temp2], axis=1)
            os.remove("floor_accel_X.txt")
            os.remove("floor_accel_Y.txt")
        
        # Get the maximum in either direction and report the location also
        max_peak_accel = np.max(peak_accel)
        ind = np.where(peak_accel == max_peak_accel)
        if ind[1][0] == 0:
            max_peak_accel_dir = 'X'
        elif ind[1][0] == 1:
            max_peak_accel_dir = 'Y'
        max_peak_accel_loc = ind[0][0]
        
        # Give some feedback on what happened  
        if coll_index == -1:
            print('------ ANALYSIS FAILED --------')
        elif coll_index == 0:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')
        elif coll_index == 1:
            print('======= ANALYSIS STOPPED FOR STRUCTURE COLLAPSE AT STOREY {:d} in {:s} ========'.format(max_peak_drift_loc, max_peak_drift_dir))
            
        if pflag is True:
            print('Final state = {:d} (-1 for non-converged, 0 for stable, 1 for collapsed)'.format(coll_index))
            print('Maximum peak storey drift {:.3f} radians at storey {:d} in the {:s} direction (Storeys = 1, 2, 3,...)'.format(max_peak_drift, max_peak_drift_loc, max_peak_drift_dir))
            print('Maximum peak floor acceleration {:.3f} g at floor {:d} in the {:s} direction (Floors = 0(G), 1, 2, 3,...)'.format(max_peak_accel, max_peak_accel_loc, max_peak_accel_dir))
                
        # Give the outputs
        return coll_index, peak_drift, peak_accel, max_peak_drift, max_peak_drift_dir, max_peak_drift_loc, max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp
        
##########################################################################
#                        PARALLEL PROCESSING MODULES                     #
##########################################################################

##########################################################################
#                             MODEL COMPILATION                          #
##########################################################################
    
    def compileModel(self):
        
        model = stickModel(self.path_to_input_file)
        model.mdof_initialise()
        model.mdof_nodes()
        model.mdof_fixity()
        model.mdof_loads()
        model.mdof_material()
        model.do_gravity_analysis()
        model.do_modal_analysis()
        
        return model

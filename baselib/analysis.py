import numpy as np
from basics import units
import openseespy.opensees as ops
import os

class analysis:
    """
    
    Class of analysis objects to be used in OpenSeesPy
    Author: Gerard J. O'Reilly
    
    """
    def __init__(self):
        pass
    
    @staticmethod
    def modal(num_modes=1, solver='-genBandArpack'):
        """
        Parameters
        ----------
        num_modes : int, optional
            Number of modes to request. The default is 1.
        solver : str, optional
            The type of solver to use. The default is '-genBandArpack'.

        Returns
        -------
        T, omega

        """
        omega = np.power(ops.eigen(solver, num_modes), 0.5)
        T = 2.0*np.pi/omega
        
        print(T)
        
        return T, omega
    
    # @staticmethod()
    # def damping(xi=[0.05, 0.05], modes=[1, 3], damp_model='Initial'):
        
        
    #     # Create the damping model
    #     T, omega = self.modal()
        
    #     # set damping based on first eigen mode
    #     alpha_m = 0.0
    #     beta_k = 2*xi/omega
        
    #     # apply the mode
    #     if damp_model == 'Initial':
    #         ops.rayleigh(alpha_m, 0.0, beta_k, 0.0)
    #     elif damp_model == 'Tangent':
    #         ops.rayleigh(alpha_m, 0.0, 0.0, beta_k)
    
    @staticmethod
    def spo(ref_disp, disp_scale_factor, control_node, push_dir, pattern_nodes, ref_loads, rxn_nodes, pflag=False, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-8, init_iter=1000, algorithm_type='KrylovNewton'):
        """
        Method to carry out a pushover of a 3D model

        Parameters
        ----------
        ref_disp : float
            Reference displacement to which cycles are run. Corresponds to yield or equivalent other, such as 1mm.
        disp_scale_factor : float
            Multiple of ref_disp to which the push is run. So pushover can be run to a specified ductility or displacement.
        control_node : int
            Node to control with the displacement integrator.
        push_dir : int
            Direction in which to push
        pattern_nodes : list
            List of nodes to which to apply the load pattern.
        ref_loads : list
            Reference values of the loads to apply.
        rxn_nodes : list
            List of nodes to record the reactions at
        pflag : bool, optional
            Whether to print supplemental information. The default is False.
        num_steps : int, optional
            Number of steps to reach the target. The default is 200.
        ansys_soe : str, optional
            System of equations. The default is 'BandGeneral'.
        constraints_handler : str, optional
            DESCRIPTION. The default is 'Transformation'.
        numberer : str, optional
            DESCRIPTION. The default is 'RCM'.
        test_type : str, optional
            DESCRIPTION. The default is 'EnergyIncr'.
            # test_type = 'NormUnbalance'				# Dont use with Penalty constraints
            # test_type = 'NormDispIncr'
            # test_type = 'EnergyIncr'					# Dont use with Penalty constraints
            # test_type = 'RelativeNormUnbalance'		# Dont use with Penalty constraints
            # test_type = 'RelativeNormDispIncr'		# Dont use with Lagrange constraints
            # test_type = 'RelativeTotalNormDispIncr'	# Dont use with Lagrange constraints
            # test_type = 'RelativeEnergyIncr'			# Dont use with Penalty constraints
        init_tol : float, optional
            DESCRIPTION. The default is 1.0e-8.
        init_iter : int, optional
            DESCRIPTION. The default is 1000.
        algorithm_type : str, optional
            DESCRIPTION. The default is 'KrylovNewton'.

        Returns
        -------
        spo_disp, spo_rxn

        """

        # Apply the load pattern
        spo_tsTag=1
        spo_pTag=1
        ops.timeSeries("Linear", spo_tsTag) # create timeSeries
        ops.pattern("Plain", spo_pTag, spo_tsTag) # create a plain load pattern

        for i in np.arange(len(pattern_nodes)):
            if push_dir == 1:
                ops.load(pattern_nodes[i], ref_loads[i], 0.0, 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 2:
                ops.load(pattern_nodes[i], 0.0, ref_loads[i], 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 3:
                ops.load(pattern_nodes[i], 0.0, 0.0, ref_loads[i], 0.0, 0.0, 0.0)
                
        
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
    
    @staticmethod
    def spo_sdof(ref_disp, disp_scale_factor, control_node, fix_node, pflag=False, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-8, init_iter=1000, algorithm_type='KrylovNewton'):
        """
        Method to carry out a pushover of an SDOF model

        Parameters
        ----------
        ref_disp : float
            Reference displacement to which cycles are run. Corresponds to yield or equivalent other, such as 1mm.
        disp_scale_factor : float
            Multiple of ref_disp to which the push is run. So pushover can be run to a specified ductility or displacement.
        control_node : int
            Node to control with the displacement integrator.
        fix_node : int
            Node to record the reactions at
        pflag : bool, optional
            Whether to print supplemental information. The default is False.
        num_steps : int, optional
            Number of steps to reach the target. The default is 200.
        ansys_soe : str, optional
            System of equations. The default is 'BandGeneral'.
        constraints_handler : str, optional
            DESCRIPTION. The default is 'Transformation'.
        numberer : str, optional
            DESCRIPTION. The default is 'RCM'.
        test_type : str, optional
            DESCRIPTION. The default is 'EnergyIncr'.
            # test_type = 'NormUnbalance'				# Dont use with Penalty constraints
            # test_type = 'NormDispIncr'
            # test_type = 'EnergyIncr'					# Dont use with Penalty constraints
            # test_type = 'RelativeNormUnbalance'		# Dont use with Penalty constraints
            # test_type = 'RelativeNormDispIncr'		# Dont use with Lagrange constraints
            # test_type = 'RelativeTotalNormDispIncr'	# Dont use with Lagrange constraints
            # test_type = 'RelativeEnergyIncr'			# Dont use with Penalty constraints
        init_tol : float, optional
            DESCRIPTION. The default is 1.0e-8.
        init_iter : int, optional
            DESCRIPTION. The default is 1000.
        algorithm_type : str, optional
            DESCRIPTION. The default is 'KrylovNewton'.

        Returns
        -------
        spo_disp, spo_rxn

        """

        # Apply the load pattern
        spo_tsTag=1
        spo_pTag=1
        ops.timeSeries("Linear", spo_tsTag) # create timeSeries
        ops.pattern("Plain", spo_pTag, spo_tsTag) # create a plain load pattern
        ops.load(control_node, 1.0)                
        
        # Set up the initial objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)        
        ops.algorithm(algorithm_type)
        
        # Set the integrator
        target_disp = ref_disp*disp_scale_factor
        delta_disp = target_disp/(1.0*num_steps)
        ops.integrator('DisplacementControl', control_node, 1, delta_disp)
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
                curr_disp = ops.nodeDisp(control_node, 1)
                print('Currently pushed node ' + str(control_node) + ' to ' + str(curr_disp) + ' with ' + str(loadf))
                       
            # Increment to the next step
            step += 1
            
            # Get the results
            spo_disp = np.append(spo_disp, ops.nodeDisp(control_node, 1))
            
            ops.reactions()
            spo_rxn = np.append(spo_rxn, -ops.nodeReaction(fix_node, 1))
            
                
        # Give some feedback on what happened  
        if ok != 0:
            print('------ ANALYSIS FAILED --------')
        elif ok == 0:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')
        
        if loadf < 0:
            print('Stopped because of load factor below zero')
                
        return spo_disp, spo_rxn
    
    @staticmethod
    def nrha(fnames, dt_gm, sf, t_max, dt_ansys, drift_limit, control_nodes, pflag=False, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-8, init_iter=1000, algorithm_type='KrylovNewton'):
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
            time-step at which to conduct the analysis. Typically small than the record's.
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

        # Define the timeseries and patterns first
        if len(fnames) > 0:
            nrha_tsTagX = 1
            nrha_pTagX = 1
            ops.timeSeries('Path', nrha_tsTagX, '-dt', dt_gm, '-filePath', fnames[0], '-factor', sf) 
            ops.pattern('UniformExcitation', nrha_pTagX, 1, '-accel', nrha_tsTagX)
            ops.recorder('Node', '-file', "floor_accel_X.txt", '-timeSeries', nrha_tsTagX, '-node', *control_nodes, '-dof', 1, 'accel')
        if len(fnames) > 1:
            nrha_tsTagY = 2
            nrha_pTagY = 2
            ops.timeSeries('Path', nrha_tsTagY, '-dt', dt_gm, '-filePath', fnames[1], '-factor', sf) 
            ops.pattern('UniformExcitation', nrha_pTagY, 2, '-accel', nrha_tsTagY)
            ops.recorder('Node', '-file', "floor_accel_Y.txt", '-timeSeries', nrha_tsTagY, '-node', *control_nodes, '-dof', 2, 'accel')
        if len(fnames) > 2:
            nrha_tsTagZ = 3
            nrha_pTagZ = 3
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
            temp1 = np.transpose(np.max(np.abs(np.loadtxt("floor_accel_X.txt")), 0))/units.g
            peak_accel[:,0] = temp1
            os.remove("floor_accel_X.txt")
        if len(fnames) > 1:
            
            temp1 = np.transpose(np.max(np.abs(np.loadtxt("floor_accel_X.txt")), 0))/units.g
            temp2 = np.transpose(np.max(np.abs(np.loadtxt("floor_accel_Y.txt")), 0))/units.g
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
    
    @staticmethod
    def nrha_sdof(fname, dt_gm, sf, t_max, dt_ansys, def_limit, control_node, pflag=False, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-8, init_iter=1000, algorithm_type='KrylovNewton'):
        """
        Method to carry out a NRHA of an SDOF model
        
        Parameters
        ----------
        fname : str
            Filepath to the ground motion to be applied.
        dt_gm : float
            time-step of the ground motions.
        sf : float
            scale factor to be applied to the records. THIS IS USUALLY GRAVITY.
        t_max : float
            duration of the record.
        dt_ansys : float
            time-step at which to conduct the analysis. Typically small than the record's.
        def_limit : float
            Deformation limit at which to stop the analysis.
        control_node : int
            Node to monitor.
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
        peak_def : float
            peak deformation
        peak_accel : numpy array
            array of the peak accelerations at each floor (determined at the control_nodes) in the directions excited. IN TERMS OF G

        """

        # Define the timeseries and patterns first
        nrha_tsTag = 1
        nrha_pTag = 1
        ops.timeSeries('Path', nrha_tsTag, '-dt', dt_gm, '-filePath', fname, '-factor', sf) 
        ops.pattern('UniformExcitation', nrha_pTag, 1, '-accel', nrha_tsTag)
        ops.recorder('Node', '-file', "floor_accel.txt", '-timeSeries', nrha_tsTag, '-node', control_node, '-dof', 1, 'accel')

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

         # Create some arrays to record to
        peak_def = 0.0
        
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
        
            # Get the current absolute deformation
            curr_def = np.abs(ops.nodeDisp(control_node, 1))
        
            # Check if the current deformation is greater than the previous
            if curr_def > peak_def:
                peak_def = curr_def
            
            # Check that the maximum in either direction hasn't exceeded the drift limit
            if peak_def > def_limit:
                coll_index = 1
        
        # Get the floor acceleration. Need to use a recorder file because a direct query would return relative values
        ops.wipe() # First wipe to finish writing to the file
        peak_accel = np.max(np.abs(np.loadtxt("floor_accel.txt")))/units.g
        os.remove("floor_accel.txt")
        
        # Give some feedback on what happened  
        if coll_index == -1:
            print('------ ANALYSIS FAILED --------')
        elif coll_index == 0:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')
        elif coll_index == 1:
            print('======= ANALYSIS STOPPED FOR COLLAPSE ========')
            
        if pflag is True:
            print('Final state = {:d} (-1 for non-converged, 0 for stable, 1 for collapsed)'.format(coll_index))
            print('Peak deformation {:.3f}'.format(peak_def))
            print('Peak floor acceleration {:.3f}'.format(peak_accel))
                
        # Give the outputs
        return coll_index, peak_def, peak_accel

   
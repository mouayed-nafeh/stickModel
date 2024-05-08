##########################################################################
#                          MDOF MODELLING MODULE                         #
##########################################################################   
from utils import *
import openseespy.opensees as ops
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as image
from scipy import stats, optimize
import piecewise_regression

class stickModel():

    def __init__(self, nst, flh, flm, stD, stF):
        """
        Stick Modelling and Processing Tool
        :param nst:           int                Number of storeys
        :param flh:           list               List of floor heights in metres (e.g. [2.5, 3.0])
        :param flm:           list               List of floor masses in tonnes (e.g. [1000, 1200])
        :param stD:           array              Array of storey displacements (size = nst, CapPoints)
        :param stF:           array              Array of storey forces (size = nst,CapPoints)
        """

        ### Run tests on input parameters
        if len(flh)!=nst or len(flm)!=nst:
            raise ValueError('Number of entries exceed the number of storeys!')
        
        self.nst = nst; 
        self.flh = flh; 
        self.flm = flm;
        self.stF = stF; 
        self.stD = stD
                
    def mdof_initialise(self):
        """
        Initialises the model builder

        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """    
        ### Set model builder
        ops.wipe() # wipe existing model
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        
    def mdof_nodes(self):
        """
        Initialises the MDOF nodes and masses
    
        Parameters
        ----------
        None.
         
        Returns
        -------
        None.
    
        """    
        ### Define base node (tag = 0)
        ops.node(0, *[0.0, 0.0, 0.0])
        ### Define floor nodes (tag = 1+)
        i = 1
        current_height = 0.0
        while i <= self.nst:
            nodeTag = i
            current_height = current_height + self.flh[i-1]
            current_mass = self.flm[i-1]
            coords = [0.0, 0.0, current_height]
            masses = [current_mass, current_mass, 1e-6, 1e-6, 1e-6, 1e-6]
            ops.node(nodeTag,*coords)
            ops.mass(nodeTag,*masses)
            i+=1

    def mdof_fixity(self):
        """
        Initialises the nodal fixities
    
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
    
        """    
        ### Get list of model nodes
        nodeList = ops.getNodeTags()
        ### Impose boundary conditions
        for i in nodeList:
            # fix the base node against all DOFs
            if i==0:
                ops.fix(i,1,1,1,1,1,1)
            # release the horizontal DOFs (1,2) and fix remaining
            else:
                ops.fix(i,0,0,1,1,1,1)
    
    def mdof_loads(self):
        """
        Initialises the nodal loads

        Parameters
        ----------
        None.
                    
        Returns
        -------
        None.

        """        
        ### Get list of model nodes
        nodeList = ops.getNodeTags()
        ### Create a plain load pattern with a linear timeseries
        ops.timeSeries('Linear', 101)
        ops.pattern('Plain',101,101)
        ### Assign the loads
        for i, node in enumerate(nodeList):
            if i==0:
                pass
            else:
                ops.load(node,0.0,0.0,-self.flm[i-1]*9.81, 0.0, 0.0, 0.0)

    def mdof_material(self):
        """
        Initialising the definition of MDOF storey material model
    
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
    
        """
        
        ### Get number of zerolength elements required
        nodeList = ops.getNodeTags()
        numEle = len(nodeList)-1
                        
        for i in range(self.nst):
            
            ### define the material tag associated with each storey
            mat1Tag = int(f'1{i}00') # hystereticSM material tag
            mat2Tag = int(f'1{i}01') # min-max material tag
            
            ### get the backbone curve definition
            D = self.stD[i,:].tolist() # deformation capacity (i.e., storey displacement in m)
            F = self.stF[i,:].tolist() # strength capacity (i.e., storey base shear in kN)
                        
            ### Create rigid elastic materials for the restrained dofs
            rigM = int(f'1{i}02')
            ops.uniaxialMaterial('Elastic', rigM, 1e6)
                        
            ### Create the nonlinear material for the unrestrained dofs
            createPinching4Material(mat1Tag, mat2Tag, F, D)
            
            ### Aggregate materials
            matTags = [mat2Tag, mat2Tag, rigM, rigM, rigM, rigM]            

            ### Define element connectivity
            eleTag = int(f'200{i}')
            eleNodes = [i, i+1]
            
            ### Create the element
            ops.element('zeroLength', eleTag, eleNodes[0], eleNodes[1], '-mat', mat2Tag, mat2Tag, rigM, rigM, rigM, rigM, '-dir', 1, 2, 3, 4, 5, 6, '-doRayleigh', 1)

    def plot_model(self, display_info=True):
        """
        Plots the Opensees model
    
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
    
        """
        with cbook.get_sample_data('C:/Users/Moayad/Documents/GitHub/stickModel/imgs/gem_logo.png') as file:
            img = image.imread(file)

        modelLineColor = 'blue'
        modellinewidth = 1
        Vert = 'Z'
               
        # get list of model nodes
        NodeCoordListX = []; NodeCoordListY = []; NodeCoordListZ = [];
        NodeMassList = []
        
        nodeList = ops.getNodeTags()
        for thisNodeTag in nodeList:
            NodeCoordListX.append(ops.nodeCoord(thisNodeTag,1))
            NodeCoordListY.append(ops.nodeCoord(thisNodeTag,2))
            NodeCoordListZ.append(ops.nodeCoord(thisNodeTag,3))
            NodeMassList.append(ops.nodeMass(thisNodeTag,1))
        
        # get list of model elements
        elementList = ops.getEleTags()
        for thisEleTag in elementList:
            eleNodesList = ops.eleNodes(thisEleTag)
            if len(eleNodesList)==2:
                [NodeItag,NodeJtag] = eleNodesList
                NodeCoordListI=ops.nodeCoord(NodeItag)
                NodeCoordListJ=ops.nodeCoord(NodeJtag)
                [NodeIxcoord,NodeIycoord,NodeIzcoord]=NodeCoordListI
                [NodeJxcoord,NodeJycoord,NodeJzcoord]=NodeCoordListJ
        
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')
        
        for i in range(len(nodeList)):
            ax.scatter(NodeCoordListX[i],NodeCoordListY[i],NodeCoordListZ[i],s=50,color='black')
            if display_info == True:
                ax.text(NodeCoordListX[i],NodeCoordListY[i],NodeCoordListZ[i],  'Node %s (%s,%s,%s)' % (str(i),str(NodeCoordListX[i]),str(NodeCoordListY[i]),str(NodeCoordListZ[i])), size=20, zorder=1, color='black') 
        
        i = 0
        while i < len(elementList):
            
            x = [NodeCoordListX[i], NodeCoordListX[i+1]]
            y = [NodeCoordListY[i], NodeCoordListY[i+1]]
            z = [NodeCoordListZ[i], NodeCoordListZ[i+1]]
            
            plt.plot(x,y,z,color='blue')
            i = i+1
        
        ax.set_xlabel('X-Direction [m]')
        ax.set_ylabel('Y-Direction [m]')
        ax.set_zlabel('Z-Direction [m]')
        plt.figimage(img, 60, 310, zorder=1, alpha=.7)
        
        plt.show()


##########################################################################
#                             ANALYSIS MODULES                           #
##########################################################################
    def do_gravity_analysis(self, nG=100,ansys_soe='UmfPack',constraints_handler='Transformation',
                            numberer='RCM',test_type='NormDispIncr',init_tol = 1.0e-6, init_iter = 500, 
                            algorithm_type='Newton' , integrator='LoadControl',analysis='Static'):        
        """
        Perform gravity analysis on MDOF
    
        Parameters
        ----------
        nG:                             int                Number of gravity analysis steps to perform.
        ansys_soe:                   string                System of equations type.
        constraints_handler:         string                The constraints handler object determines how the constraint equations are enforced in the analysis. Constraint equations enforce a specified value for a DOF, or a relationship between DOFs.
        numberer:                    string                The DOF numberer object determines the mapping between equation numbers and degrees-of-freedom – how degrees-of-freedom are numbered.
        test_type:                   string                This command is used to construct the LinearSOE and LinearSolver objects to store and solve the test of equations in the analysis.
        init_tol:                     float                Tolerance criteria used to check for convergence.
        init_iter:                    float                Max number of iterations to check.
        algorithm_type:              string                The integrator object determines the meaning of the terms in the system of equation object Ax=B.
        analysis:                    string                The analysis object, which defines what type of analysis is to be performed.
        
        Returns
        -------
        None.
    
        """
        
        ### Define the analysis objects and run gravity analysis
        ops.system(ansys_soe) # creates the system of equations, a sparse solver with partial pivoting
        ops.constraints(constraints_handler) # creates the constraint handler, the transformation method
        ops.numberer(numberer) # creates the DOF numberer, the reverse Cuthill-McKee algorithm
        ops.test(test_type, init_tol, init_iter, 3) # creates the convergence test
        ops.algorithm(algorithm_type) # creates the solution algorithm, a Newton-Raphson algorithm
        ops.integrator(integrator, (1/nG)) # creates the integration scheme
        ops.analysis(analysis) # creates the analysis object
        ops.analyze(nG) # perform the gravity load analysis
        ops.loadConst('-time', 0.0)

        ### Wipe the analysis objects
        ops.wipeAnalysis()
        
    def do_modal_analysis(self, num_modes=3, solver = '-genBandArpack', doRayleigh=False, pflag=False):
        """
        Perform modal analysis on MDOF
    
        Parameters
        ----------
        num_modes:                      int                Number of modes to consider (default is 2).
        solver:                      string                Type of solver (default is -genBandArpack).
        pflag:                       string                Flag to print (or not) the modal analysis report.
        
        Returns
        -------
        T:                            array                Periods of vibration.
        """        
    
        ### Get frequency and period        
        self.omega = np.power(ops.eigen(solver, num_modes), 0.5)
        T = 2.0*np.pi/self.omega
        ### Get list of model nodes
        nodeList = ops.getNodeTags()            
        ### Print optional report
        if pflag == True:
            ops.modalProperties('-print')
        else:
            pass
        ### Print output
        print(r'Fundamental Period:  T = {:.3f} s'.format(T[0]))
                
        ### Wipe the analysis objects
        ops.wipeAnalysis()      
        
        return T
            
    def do_spo_analysis(self, ref_disp, disp_scale_factor, push_dir, pflag=True, 
                        num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', 
                        numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-8, init_iter=1000, 
                        algorithm_type='KrylovNewton'):
        """
        Perform static pushover analysis on MDOF
    
        Parameters
        ----------
        ref_disp:                     float                Reference displacement to analyses are run. Corresponds to yield or equivalent other, such as 1mm.
        disp_scale_factor:            float                Multiple of ref_disp to which the push is run. So pushover can be run to a specified ductility or displacement.
        push_dir:                       int                Direction of pushover (1 = X; 2 = Y; 3 = Z)
        pflag:                       string                Flag to print (or not) the static pushover analysis steps
        num_steps:                      int                Number of spo analysis steps to perform
        ansys_soe:                   string                System of equations type
        constraints_handler:         string                The constraints handler object determines how the constraint equations are enforced in the analysis. Constraint equations enforce a specified value for a DOF, or a relationship between DOFs.
        numberer:                    string                The DOF numberer object determines the mapping between equation numbers and degrees-of-freedom – how degrees-of-freedom are numbered.
        test_type:                   string                This command is used to construct the LinearSOE and LinearSolver objects to store and solve the test of equations in the analysis
        init_tol:                     float                Tolerance criteria used to check for convergence.
        init_iter:                    float                Max number of iterations to check
        algorithm_type:              string                The integrator object determines the meaning of the terms in the system of equation object Ax=B
        
        Returns
        -------
        spo_disps:                    array                Displacements at each floor
        spo_rxn:                      array                Base shear as the sum of the reaction at the base 
    
        """        

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
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], nodeList[i]/len(pattern_nodes), 0.0, 0.0, 0.0, 0.0, 0.0)
            
            elif push_dir == 2:
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], 0.0, nodeList[i]/len(pattern_nodes), 0.0, 0.0, 0.0, 0.0)
    
            elif push_dir == 3:
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], 0.0, 0.0, nodeList[i]/len(pattern_nodes), 0.0, 0.0, 0.0)
                            
        # Set up the initial objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)        
        ops.algorithm(algorithm_type)
        
        # Set the integrator
        target_disp = float(ref_disp)*float(disp_scale_factor)
        delta_disp = target_disp/(1.0*num_steps)
        ops.integrator('DisplacementControl', control_node, push_dir, delta_disp)
        ops.analysis('Static')
        
        # Give some feedback if requested
        if pflag is True:
            print(f"\n------ Static Pushover Analysis of Node # {control_node} to {target_disp} ---------")
        # Set up the analysis
        ok = 0
        step = 1
        loadf = 1.0
        
        # Recording base shear
        spo_rxn = np.array([0.])
        # Recording top displacement
        spo_top_disp = np.array([ops.nodeResponse(control_node, push_dir,1)])
        # Recording all displacements to estimate drifts
        spo_disps = np.array([[ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]])
        
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
            spo_top_disp = np.append(spo_top_disp, ops.nodeResponse(
            control_node, push_dir, 1))
    
            spo_disps = np.append(spo_disps, np.array([
            [ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]
            ]), axis=0)
                
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
        
        ### Wipe the analysis objects
        ops.wipeAnalysis()
             
        return spo_disps, spo_rxn
    
    def do_cpo_analysis(self, ref_disp, mu, numCycles, push_dir, dispIncr, pflag=True, 
                        num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', 
                        numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-5, init_iter=1000,
                        algorithm_type='KrylovNewton'):
        """
        Perform cyclic pushover analysis on MDOF
    
        Parameters
        ----------
        ref_disp:                     float                Reference displacement to analyses are run. Corresponds to yield or equivalent other, such as 1mm.
        mu:                           float                Target ductility.
        numCycles:                    float                Number of cycles.
        dispIncr:                     float                Number of displacement increments.
        push_dir:                       int                Direction of pushover (1 = X; 2 = Y; 3 = Z).
        pflag:                       string                Flag to print (or not) the static pushover analysis steps.
        num_steps:                      int                Number of spo analysis steps to perform.
        ansys_soe:                   string                System of equations type.
        constraints_handler:         string                The constraints handler object determines how the constraint equations are enforced in the analysis. Constraint equations enforce a specified value for a DOF, or a relationship between DOFs.
        numberer:                    string                The DOF numberer object determines the mapping between equation numbers and degrees-of-freedom – how degrees-of-freedom are numbered.
        test_type:                   string                This command is used to construct the LinearSOE and LinearSolver objects to store and solve the test of equations in the analysis.
        init_tol:                     float                Tolerance criteria used to check for convergence.
        init_iter:                    float                Max number of iterations to check.
        algorithm_type:              string                The integrator object determines the meaning of the terms in the system of equation object Ax=B.
        
        Returns
        -------
        cpo_disps:                    array                Displacements at each floor.
        cpo_rxn:                      array                Base shear as the sum of the reaction at the base.
    
        """        

        # apply the load pattern
        ops.timeSeries("Linear", 1) # create timeSeries
        ops.pattern("Plain",1,1) # create a plain load pattern
                
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

     	#create the list of displacements
        dispList =                  [ref_disp*mu, -2.0*ref_disp*mu, ref_disp*mu]
        cycleDispList = (numCycles* [ref_disp*mu, -2.0*ref_disp*mu, ref_disp*mu])
        dispNoMax = len(cycleDispList)
        
        # Give some feedback if requested
        if pflag is True:
            print(f"\n------ Cyclic Pushover Analysis of Node # {control_node} for {numCycles} cycles to ductility: {mu}---------")
        
        # Recording base shear
        cpo_rxn = np.array([0.])
        # Recording top displacement
        cpo_top_disp = np.array([ops.nodeResponse(control_node, push_dir,1)])
        # Recording all displacements to estimate drifts
        cpo_disps = np.array([[ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]])

    
        for d in range(dispNoMax):
            numIncr = dispIncr
            dU = cycleDispList[d]/(1.0*numIncr)
            ops.integrator('DisplacementControl', control_node, push_dir, dU)
            ops.analysis('Static')
            
            for l in range(numIncr):
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
                if ok != 0:
                    print('Analysis Failed')
                    break 
                                
            # Give some feedback if requested
            if pflag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print('Currently pushed node ' + str(control_node) + ' to ' + str(curr_disp))
                            
            # Get the results
            cpo_top_disp = np.append(cpo_top_disp, ops.nodeResponse(
            control_node, push_dir, 1))
    
            cpo_disps = np.append(cpo_disps, np.array([
            [ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]
            ]), axis=0)
                
            ops.reactions()
            temp = 0
            for n in rxn_nodes:
                temp += ops.nodeReaction(n, push_dir)
            cpo_rxn = np.append(cpo_rxn, -temp)
        
        ### Wipe the analysis objects
        ops.wipeAnalysis()
                      
        return cpo_disps, cpo_rxn
                    
    # def do_nrha_analysis(self, fnames, dt_gm, sf, t_max, dt_ansys, drift_limit, 
    #                      outdir, pflag=False, ansys_soe='BandGeneral', 
    #                      constraints_handler='Transformation', numberer='RCM', 
    #                      test_type='EnergyIncr', init_tol=1.0e-8, init_iter=1000, 
    #                      algorithm_type='KrylovNewton'):
    #     """
    #     Method to carry out a NRHA of a 3D model
    
    #     Parameters
    #     ----------
    #     *fnames : list (str)
    #         List of the filepaths to the ground motions to be applied in the X Y and Z. At least the X direction is required.
    #     dt_gm : float
    #         time-step of the ground motions.
    #     sf : float
    #         scale factor to be applied to the records. THIS IS USUALLY GRAVITY.
    #     t_max : float
    #         duration of the record.
    #     dt_ansys : float
    #         time-step at which to conduct the analysis. Typically smaller than the record's.
    #     drift_limit : float
    #         Drift limit at which to stop the analysis (IN RADIANS).
    #     control_nodes : list (int)
    #         List of the nodes in sequential order to monitor.
    #     pflag : bool, optional
    #         Whether to print information on screen or not. The default is False.
    #     ansys_soe : TYPE, optional
    #         DESCRIPTION. The default is 'BandGeneral'.
    #     constraints_handler : TYPE, optional
    #         DESCRIPTION. The default is 'Transformation'.
    #     numberer : TYPE, optional
    #         DESCRIPTION. The default is 'RCM'.
    #     test_type : TYPE, optional
    #         DESCRIPTION. The default is 'EnergyIncr'.
    #     init_tol : TYPE, optional
    #         DESCRIPTION. The default is 1.0e-8.
    #     init_iter : TYPE, optional
    #         DESCRIPTION. The default is 1000.
    #     algorithm_type : TYPE, optional
    #         DESCRIPTION. The default is 'KrylovNewton'.
    
    #     Raises
    #     ------
    #     ValueError
    #         DESCRIPTION.
    
    #     Returns
    #     -------
    #     coll_index : int
    #         Collapse index (-1 for non-converged, 0 for stable, 1 for collapsed).
    #     peak_drift : numpy array (ratio, not in %)
    #         array of the peak drifts at each storey (determined as the storeys between the listed control_nodes) in the directions excited. IN RADIANS
    #     peak_accel : numpy array (in m/s2, divide by 9.81 to get in g)
    #         array of the peak accelerations at each floor (determined at the control_nodes) in the directions excited. IN TERMS OF G
    #     max_peak_drift : float (ratio, not in %)
    #         max in either dirction of peak_drift.
    #     max_peak_drift_dir : str
    #         direction of peak_drift.
    #     max_peak_drift_loc : int
    #         storey location of peak_drift.(Storeys = 1, 2, 3,...)
    #     max_peak_accel : float (in m/s2, divide by 9.81 to get in g)
    #         max in either dirction of peak_accel.
    #     max_peak_accel_dir : str
    #         direction of peak_accel.
    #     max_peak_accel_loc : ind
    #         storey location of peak_accel.(Floors = 0(G), 1, 2, 3,...)
    
    #     """
    #     # define control nodes
    #     control_nodes = ops.getNodeTags()
        
    #     # Define the timeseries and patterns first
    #     if len(fnames) > 0:
    #         nrha_tsTagX = 1
    #         nrha_pTagX = 1
    #         ops.timeSeries('Path', nrha_tsTagX, '-dt', dt_gm, '-filePath', fnames[0], '-factor', sf) 
    #         ops.pattern('UniformExcitation', nrha_pTagX, 1, '-accel', nrha_tsTagX)
    #         ops.recorder('Node', '-file', "floor_accel_X.txt", '-timeSeries', nrha_tsTagX, '-node', *control_nodes, '-dof', 1, 'accel')
    #     if len(fnames) > 1:
    #         nrha_tsTagY = 2
    #         nrha_pTagY = 2
    #         ops.timeSeries('Path', nrha_tsTagY, '-dt', dt_gm, '-filePath', fnames[1], '-factor', sf) 
    #         ops.pattern('UniformExcitation', nrha_pTagY, 2, '-accel', nrha_tsTagY)
    #         ops.recorder('Node', '-file', "floor_accel_Y.txt", '-timeSeries', nrha_tsTagY, '-node', *control_nodes, '-dof', 2, 'accel')
    #     if len(fnames) > 2:
    #         nrha_tsTagZ = 3
    #         nrha_pTagZ = 3
    #         ops.timeSeries('Path', nrha_tsTagZ, '-dt', dt_gm, '-filePath', fnames[2], '-factor', sf) 
    #         ops.pattern('UniformExcitation', nrha_pTagZ, 3, '-accel', nrha_tsTagZ)
                
    #     # Set up the initial objects
    #     ops.system(ansys_soe)
    #     ops.constraints(constraints_handler)
    #     ops.numberer(numberer)
    #     ops.test(test_type, init_tol, init_iter)        
    #     ops.algorithm(algorithm_type)
    #     ops.integrator('Newmark', 0.5, 0.25)
    #     ops.analysis('Transient')
        
    #     # Set up analysis parameters
    #     coll_index = 0   # Initially define the collapse index (-1 for non-converged, 0 for stable, 1 for collapsed)
    #     control_time = 0.0 
    #     ok = 0 # Set the convergence to 0 (initially converged)
        
    #     # Parse the data about the building
    #     top_nodes = control_nodes[1:]
    #     bottom_nodes = control_nodes[0:-1]
    #     h = []
    #     for i in np.arange(len(top_nodes)):
    #         topZ = ops.nodeCoord(top_nodes[i], 3)
    #         bottomZ = ops.nodeCoord(bottom_nodes[i], 3)
    #         dist = topZ - bottomZ
    #         if dist == 0:
    #             print("WARNING: Zero length found in drift check, using very large distance 1e9 instead")
    #             h.append(1e9)
    #         else:
    #             h.append(dist)
        
    #     # Create some arrays to record to
    #     peak_disp = np.zeros((len(control_nodes), 2))
    #     peak_drift = np.zeros((len(top_nodes), 2))
    #     peak_accel = np.zeros((len(top_nodes)+1, 2))
        
    #     # Run the actual analysis
    #     while coll_index == 0 and control_time <= t_max and ok == 0:
    #         ok = ops.analyze(1, dt_ansys)
    #         control_time = ops.getTime()
            
    #         if pflag is True:
    #             print('Completed {:.3f}'.format(control_time) + ' of {:.3f} seconds'.format(t_max) )
        
    #         # If the analysis fails, try the following changes to achieve convergence
    #         if ok != 0:
    #             print('FAILED at {:.3f}'.format(control_time) + ': Trying reducing time-step in half...')
    #             ok = ops.analyze(1, 0.5*dt_ansys)
    #         if ok != 0:
    #             print('FAILED at {:.3f}'.format(control_time) + ': Trying reducing time-step in quarter...')
    #             ok = ops.analyze(1, 0.25*dt_ansys)
    #         if ok != 0:
    #             print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iterations...')
    #             ops.test(test_type, init_tol*0.01, init_iter*10)
    #             ok = ops.analyze(1, 0.5*dt_ansys)
    #             ops.test(test_type, init_tol, init_iter)
    #         if ok != 0:
    #             print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iteration and Newton with initial then current...')
    #             ops.test(test_type, init_tol*0.01, init_iter*10)
    #             ops.algorithm('Newton', 'initialThenCurrent')
    #             ok = ops.analyze(1, 0.5*dt_ansys)
    #             ops.test(test_type, init_tol, init_iter)
    #             ops.algorithm(algorithm_type)
    #         if ok != 0:
    #             print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iteration and Newton with initial...')
    #             ops.test(test_type, init_tol*0.01, init_iter*10)
    #             ops.algorithm('Newton', 'initial')
    #             ok = ops.analyze(1, 0.5*dt_ansys)
    #             ops.test(test_type, init_tol, init_iter)
    #             ops.algorithm(algorithm_type)
    #         if ok != 0:
    #             print('FAILED at {:.3f}'.format(control_time) + ': Attempting a Hail Mary...')
    #             ops.test('FixedNumIter', init_tol*0.01, init_iter*10)
    #             ok = ops.analyze(1, 0.5*dt_ansys)
    #             ops.test(test_type, init_tol, init_iter)
                
    #         # Game over......
    #         if ok != 0:
    #             print('FAILED at {:.3f}'.format(control_time) + ': Exiting analysis...')
    #             coll_index = -1
        
    #         # For each of the nodes to monitor, get the current drift
    #         for i in np.arange(len(top_nodes)):
                
    #             # Get the current storey drifts - absolute difference in displacement over the height between them
    #             curr_drift_X = np.abs(ops.nodeDisp(top_nodes[i], 1) - ops.nodeDisp(bottom_nodes[i], 1))/h[i]
    #             curr_drift_Y = np.abs(ops.nodeDisp(top_nodes[i], 2) - ops.nodeDisp(bottom_nodes[i], 2))/h[i]
        
    #             # Check if the current drift is greater than the previous peaks at the same storey
    #             if curr_drift_X > peak_drift[i, 0]:
    #                 peak_drift[i, 0] = curr_drift_X
                
    #             if curr_drift_Y > peak_drift[i, 1]:
    #                 peak_drift[i, 1] = curr_drift_Y
                    
    #         # For each node to monitor, get is absolute displacement
    #         for i in np.arange(len(control_nodes)):
    #             curr_disp_X = np.abs(ops.nodeDisp(control_nodes[i], 1))
    #             curr_disp_Y = np.abs(ops.nodeDisp(control_nodes[i], 2))
                
    #             # Check if the current drift is greater than the previous peaks at the same storey
    #             if curr_disp_X > peak_disp[i, 0]:
    #                 peak_disp[i, 0] = curr_disp_X
                
    #             if curr_disp_Y > peak_disp[i, 1]:
    #                 peak_disp[i, 1] = curr_disp_Y
            
    #         # Check that the maximum in either direction hasn't exceeded the drift limit
    #         if np.max(peak_drift) > drift_limit:
    #             coll_index = 1
        
    #     # Now that the analysis is finished, get the maximum in either direction and report the location also
    #     max_peak_drift = np.max(peak_drift)
    #     ind = np.where(peak_drift == max_peak_drift)
    #     if ind[1][0] == 0:
    #         max_peak_drift_dir = 'X'
    #     elif ind[1][0] == 1:
    #         max_peak_drift_dir = 'Y'
    #     max_peak_drift_loc = ind[0][0]+1
        
    #     # Get the floor accelerations. Need to use a recorder file because a direct query would return relative values
    #     ops.wipe() # First wipe to finish writing to the file
        
    #     if len(fnames) > 0:
    #         temp1 = np.transpose(np.max(np.abs(np.loadtxt(f"{outdir}/floor_accel_X.txt")), 0))
    #         peak_accel[:,0] = temp1
    #         os.remove(f"{outdir}/floor_accel_X.txt")
        
    #     elif len(fnames) > 1:
            
    #         temp1 = np.transpose(np.max(np.abs(np.loadtxt(f"{outdir}/floor_accel_X.txt")), 0))
    #         temp2 = np.transpose(np.max(np.abs(np.loadtxt(f"{outdir}/floor_accel_Y.txt")), 0))
    #         peak_accel = np.stack([temp1, temp2], axis=1)
    #         os.remove(f"{outdir}/floor_accel_X.txt")
    #         os.remove(f"{outdir}/floor_accel_Y.txt")
        
    #     # Get the maximum in either direction and report the location also
    #     max_peak_accel = np.max(peak_accel)
    #     ind = np.where(peak_accel == max_peak_accel)
    #     if ind[1][0] == 0:
    #         max_peak_accel_dir = 'X'
    #     elif ind[1][0] == 1:
    #         max_peak_accel_dir = 'Y'
    #     max_peak_accel_loc = ind[0][0]
        
    #     # Give some feedback on what happened  
    #     if coll_index == -1:
    #         print('------ ANALYSIS FAILED --------')
    #     elif coll_index == 0:
    #         print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')
    #     elif coll_index == 1:
    #         print('======= ANALYSIS STOPPED FOR STRUCTURE COLLAPSE AT STOREY {:d} in {:s} ========'.format(max_peak_drift_loc, max_peak_drift_dir))
            
    #     if pflag is True:
    #         print('Final state = {:d} (-1 for non-converged, 0 for stable, 1 for collapsed)'.format(coll_index))
    #         print('Maximum peak storey drift {:.3f} radians at storey {:d} in the {:s} direction (Storeys = 1, 2, 3,...)'.format(max_peak_drift, max_peak_drift_loc, max_peak_drift_dir))
    #         print('Maximum peak floor acceleration {:.3f} g at floor {:d} in the {:s} direction (Floors = 0(G), 1, 2, 3,...)'.format(max_peak_accel, max_peak_accel_loc, max_peak_accel_dir))
                   
    #     # Give the outputs
    #     return control_nodes, coll_index, peak_drift, peak_accel, max_peak_drift, max_peak_drift_dir, max_peak_drift_loc, max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp
      
    def do_nrha_analysis(self, fnames, dt_gm, sf, t_max, dt_ansys, 
                         pflag=True, xi = 0.05, ansys_soe='BandGeneral', 
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
        peak_drift : numpy array (ratio, not in %)
            array of the peak drifts at each storey (determined as the storeys between the listed control_nodes) in the directions excited. IN RADIANS
        peak_accel : numpy array (in m/s2, divide by 9.81 to get in g)
            array of the peak accelerations at each floor (determined at the control_nodes) in the directions excited. IN TERMS OF G
        max_peak_drift : float (ratio, not in %)
            max in either dirction of peak_drift.
        max_peak_drift_dir : str
            direction of peak_drift.
        max_peak_drift_loc : int
            storey location of peak_drift.(Storeys = 1, 2, 3,...)
        max_peak_accel : float (in m/s2, divide by 9.81 to get in g)
            max in either dirction of peak_accel.
        max_peak_accel_dir : str
            direction of peak_accel.
        max_peak_accel_loc : ind
            storey location of peak_accel.(Floors = 0(G), 1, 2, 3,...)
    
        """
        # define control nodes
        control_nodes = ops.getNodeTags()
        
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
        conv_index = 0   # Initially define the collapse index (-1 for non-converged, 0 for stable)
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
        
        # Set damping 
        alphaM = 2*self.omega[0]*xi
        ops.rayleigh(alphaM,0,0,0)
        
        # Run the actual analysis
        while conv_index == 0 and control_time <= t_max and ok == 0:
            ok = ops.analyze(1, dt_ansys)
            control_time = ops.getTime()
            
            # if pflag is True:
            #     print('Completed {:.3f}'.format(control_time) + ' of {:.3f} seconds'.format(t_max) )
        
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
                conv_index = -1
        
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
            temp1 = np.transpose(np.max(np.abs(np.loadtxt("floor_accel_X.txt")), 0))
            peak_accel[:,0] = temp1
            os.remove("floor_accel_X.txt")
        
        elif len(fnames) > 1:
            
            temp1 = np.transpose(np.max(np.abs(np.loadtxt("floor_accel_X.txt")), 0))
            temp2 = np.transpose(np.max(np.abs(np.loadtxt("floor_accel_Y.txt")), 0))
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
        if conv_index == -1:
            print('------ ANALYSIS FAILED --------')
        elif conv_index == 0:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')
            
        if pflag is True:
            print('Final state = {:d} (-1 for non-converged, 0 for stable)'.format(conv_index))
            print('Maximum peak storey drift {:.3f} radians at storey {:d} in the {:s} direction (Storeys = 1, 2, 3,...)'.format(max_peak_drift, max_peak_drift_loc, max_peak_drift_dir))
            print('Maximum peak floor acceleration {:.3f} g at floor {:d} in the {:s} direction (Floors = 0(G), 1, 2, 3,...)'.format(max_peak_accel, max_peak_accel_loc, max_peak_accel_dir))
                   
        # Give the outputs
        return control_nodes, conv_index, peak_drift, peak_accel, max_peak_drift, max_peak_drift_dir, max_peak_drift_loc, max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp


##########################################################################
#                         SDOF-MDOF CALIBRATION                          #
##########################################################################

## load dependencies
import openseespy.opensees as ops
import pandas as pd
import numpy as np
import os
from units import *
from utils import *           
import matplotlib.pyplot as plt
import itertools
from scipy import optimize

class stickModelCalibration():

    def __init__(self, nst,capacity_array,gamma,storey_height=2.8):
        """
        Stick Modelling and Processing Tool
        :param n_storeys:         int                Number of storeys
        :param capacity_array:    array              Numpy array from SDOF capacity database
        :param gamma:             float              Transformation factor
        :param storey_height:     float              Storey Height (default set to 2.8m)
        """
        
        self.nst = nst
        self.capacity_array = capacity_array
        self.gamma = gamma
        self.storey_height = storey_height
        
    def computeModalShape(self):
        
        ### initialise the modal shape
        phi=np.zeros(int(self.nst))
        total_height=self.nst*self.storey_height
        storey_heights=[(i+1)*self.storey_height for i in range(int(self.nst))]
        
        ### define function for modal shape
        def func(x):
            for i in range(int(self.nst)):
                # Normalise top 
                if i==(self.nst-1):
                    phi[i]=1.0
                else:
                    phi[i]=x[i]
            # Recompute the tranformation during optimisation and check prediction/initial  
            computed_gamma=np.sum(phi)/np.sum(phi**2)        
            loss_squared=(self.gamma-computed_gamma)**2

            return loss_squared
        
        optimize.Bounds(lb=0, ub=1, keep_feasible=False)       
        solution=optimize.fmin(func,[storey_heights[i]/total_height for i in range(int(self.nst)-1)],disp=False)
        
        for i in range(int(self.nst)):
            
            if i==(int(self.nst)-1):
                phi[i]=1.0
            else:
                phi[i]=solution[i]
        
        return phi 
    
    def computeStoreyMasses(self, phi):
        
        if self.capacity_array.shape[0]>=4:
              say=self.capacity_array[2,1]*9.81
              sdy=self.capacity_array[2,0]
        else:
              say=self.capacity_array[1,1]*9.81
              sdy=self.capacity_array[1,0]     
              
        period=(2*np.pi)/(np.sqrt(say/sdy))

        # from EC8 annex B equations and assuming equal masses per storey
        
        m_star=((period/(2*np.pi))**2)/(sdy/say)
        
        if self.nst>1:
          m_storeys=m_star/(np.sum(phi))
        elif self.nst==1:
           m_storeys=1.0
        
        return m_storeys
    
    def compute_storey_force(total_base_shear,phi):
        
        # this function computes the lateral force at floor level assuming a distribution proportional to phi1
        # it outputs an array where node0=force at base (it is 0 always) and noden= force at top
        
        ratios=np.concatenate([np.array([0]),phi])
        force_at_top_floor=total_base_shear/np.sum(ratios)
        nodal_lateral_forces=force_at_top_floor*ratios   
        
        return nodal_lateral_forces
    
    def compute_M_at_floors(nodal_lateral_forces, nst, storey_height):
    
        storey_heights=storey_height*np.linspace(start=0,
                                                 stop=nst,
                                                 num=nst+1,endpoint=True)
        
        nodal_moment_value=np.zeros(storey_heights.shape)
        
        for i in reversed(range(nst)):
            
            nodal_moment_value[i]=np.sum(np.multiply(nodal_lateral_forces[i+1:],
                                                      storey_heights[i+1:]-storey_heights[i]))
        return nodal_moment_value
    
    def compute_M_M_integral(L,M1,M2,M3,M4):
        # this computes the M-M' integral assuming a trapesoidal distribution
        # M1 M2 are the moments for M
        # M3 M4 are the moments for M'    
        return L/6*(M1*(2*M3+M4)+M2*(M3+2*M4))
    
    
    def computeSpringParams(self,total_mass,phi):
               
        roof_displacement_array=self.capacity_array[:,0]*self.gamma
        base_shear_array=self.capacity_array[:,1]*9.81*total_mass*self.gamma
        
        # estimate different periods (i.e. T1, Ty, ...,Tult)
        periods_array=np.zeros(self.capacity_array.shape[0]-1)
        
        for i in range(1,self.capacity_array.shape[0]):
            
            periods_array[i-1]=np.sqrt((2*np.pi)**2/\
                            ((self.capacity_array[i,1]*9.81)/self.capacity_array[i,0]))
        
        stiff_storey_mult=np.ones([self.nst])
                
        # using the force method lets compute the EI value that leads to the roof displacement and base shear 
        # in the capacity curve (the lateral loads are assumed to be a triangular shape)
        # d_roof=1/EI*integral(M*M'dx)
        
        storey_heights=self.storey_height*np.linspace(start=0,
                                                 stop=self.nst,
                                                 num=self.nst+1,endpoint=True)
        
        springs_stiffness_matrix=np.zeros([self.capacity_array.shape[0],self.nst])
        EI_matrix=np.zeros([self.capacity_array.shape[0]-1])
        for i in range(1,self.capacity_array.shape[0]):
            
            ### Get the lateral loads
            lateral_loads=stickModelCalibration.compute_storey_force(base_shear_array[i],phi)
            ### Get the bending moment values
            m_values    =stickModelCalibration.compute_M_at_floors(lateral_loads, self.nst, self.storey_height)
            m_aux_values=stickModelCalibration.compute_M_at_floors(np.array([0 if _<self.nst else 1 for _ in range(self.nst+1)]),self.nst, self.storey_height)
                                             
            
            sum_accumulator=0
            
            for j in range(self.nst):
                
                M1=m_values[j]
                M2=m_values[j+1]
                
                M3=m_aux_values[j]
                M4=m_aux_values[j+1]
                
                L=storey_heights[j+1]-storey_heights[j]
                
                integral=stickModelCalibration.compute_M_M_integral(L,M1,M2,M3,M4)
                
                sum_accumulator+=(1/stiff_storey_mult[j])*integral
            
            EI=(1/roof_displacement_array[i])*sum_accumulator
            
            springs_stiffness_matrix[i,:]=np.array(stiff_storey_mult)*EI
            EI_matrix[i-1]=EI
        
        return springs_stiffness_matrix,periods_array,EI_matrix
    
    def compute_nodal_displacements(self,springs_stiffness_matrix,total_mass,phi):
        
        ### Initialise the nodal displacement matrix
        nodal_displacement_matrix=np.zeros([self.capacity_array.shape[0],self.nst+1])
        
        storey_heights=self.storey_height*np.linspace(start=0,
                                                 stop=self.nst,
                                                 num=self.nst+1,endpoint=True)
           
        base_shear_array=self.capacity_array[:,1]*9.81*total_mass*self.gamma
        
        for i in range(1,self.capacity_array.shape[0]):
            
            lateral_loads=stickModelCalibration.compute_storey_force(base_shear_array[i],phi)
            m_values=stickModelCalibration.compute_M_at_floors(lateral_loads,self.nst,self.storey_height)
            
            for j in range(self.nst+1):
                
                m_aux_values=stickModelCalibration.compute_M_at_floors(np.array([1 if _==j else 0 for _ in range(self.nst+1)]),
                                                 self.nst,self.storey_height)
                
                sum_accumulator=0
                
                for k in range(self.nst):
                    
                    M1=m_values[k]
                    M2=m_values[k+1]
                    
                    M3=m_aux_values[k]
                    M4=m_aux_values[k+1]
                    
                    L=storey_heights[k+1]-storey_heights[k]
                    
                    integral=stickModelCalibration.compute_M_M_integral(L,M1,M2,M3,M4)
                    
                    EIk=springs_stiffness_matrix[i,k]
                    
                    sum_accumulator+=(1/EIk)*integral
                    
                nodal_displacement_matrix[i,j]=sum_accumulator
                
        return nodal_displacement_matrix
    
    def compute_F_D_material_law(self,EI_matrix,displacements,x):
        
        # cantiliver beam P=(3EI/L**3)*D
        f_array=np.zeros(EI_matrix.shape[0])
        f_d_matrix=np.zeros([EI_matrix.shape[0],self.nst*2])
        for i in range(EI_matrix.shape[0]):
            
            for j in range(self.nst):
            
                f_array[i]=(x[j]*EI_matrix[i]/(self.nst*self.storey_height)**3)*displacements[i]
                
                f_d_matrix[i,j*2]=f_array[i]
                f_d_matrix[i,j*2+1]=displacements[i]
        
        return f_array,f_d_matrix
    
    
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
    
    def mdof_nodes(self,n_storeys,m_storeys,storey_height):
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
        while i <= n_storeys:
            nodeTag = i
            current_height = current_height + storey_height
            current_mass = m_storeys
            coords = [0.0, 0.0, current_height]
            masses = [current_mass, current_mass, current_mass, current_mass, current_mass, current_mass]
            ops.node(nodeTag,*coords)
            ops.mass(nodeTag,*masses)
            i+=1
            
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
                
    def mdof_loads(self,n_storeys,m_storeys):
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
        floor_loads = []
        for i in range(n_storeys):
            floor_loads.append(m_storeys*9.81)
        # get list of model nodes
        node_list = ops.getNodeTags()
        # create a plain load pattern with a linear timeseries
        ops.timeSeries('Linear', 101)
        ops.pattern('Plain',101,101)
        # load the nodes
        for i, node in enumerate(node_list):
            if i==0:
                pass
            else:
                ops.load(node,0.0,0.0,-floor_loads[i-1], 0.0, 0.0, 0.0)
                
                
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
    
    def mdof_material(self,n_storeys,f_array,d_array):
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
        #matDir = f'{self.gitDir}/raw/in/'   
        
        # get number of zerolength elements required
        nodeList = ops.getNodeTags()
        numEle = len(nodeList)-1
        
        # initialise some element params
        dirs = [1,2,3,4,5,6]
        D = np.abs(d_array)
        F = np.abs(f_array)
        for i in range(n_storeys):
            
            # define the material tag associated with each storey
            mat1Tag = int(f'1{i}00')
            mat2Tag = int(f'1{i}01')
            
            # get the backbone curve definition

    
            # create rigid elastic materials for the restrained dofs
            rigM = int(f'1{i}02')
            ops.uniaxialMaterial('Elastic', rigM, 1e6)
    
            # create the material
            createPinching4Material(mat1Tag, F, D)
            ops.uniaxialMaterial('MinMax', mat2Tag, mat1Tag, '-min', -D[-1], '-max', D[-1])
            
            # aggregate all material tags in one
            matTags = [mat2Tag, mat2Tag, rigM, rigM, rigM, rigM]            

            # define the connectivity parameters
            eleTag = int(f'200{i}')
            eleNodes = [i, i+1]
            
            # create the element
            #ops.element('zeroLength', eleTag, eleNodes, '-mat', matTags, '-dir', *dirs)
            ops.element('zeroLength', eleTag, eleNodes[0], eleNodes[1], '-mat', mat2Tag, mat2Tag, rigM, rigM, rigM, rigM, '-dir', 1, 2, 3, 4, 5, 6, '-doRayleigh', 1)
    
    
    def do_spo_analysis(self, target_disp, push_dir, pflag=True, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-8, init_iter=1000, algorithm_type='KrylovNewton'):
                
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
                         
        return spo_disps, spo_rxn
    
    def run_calibration(self,phi,m_storeys,EI_array,d_array):
        
        def func(x):
            total_mass=self.nst*m_storeys
            ops.wipe()
            stickModelCalibration.mdof_initialise(self)
            stickModelCalibration.mdof_nodes(self,self.nst,m_storeys,self.storey_height)
            stickModelCalibration.mdof_fixity(self)
            stickModelCalibration.mdof_loads(self,self.nst,m_storeys)
            
            # cantiliver beam P=(3EI/L**3)*D
            
            for i in range(self.nst):
            
                f_array=np.zeros(EI_array.shape[0])
                
                for j in range(EI_array.shape[0]):
                    
                    f_array[j]=np.abs((np.abs(x[i])*EI_array[j]/(self.nst*self.storey_height)**3)*d_array[j])
                    
                    
                D = np.abs(d_array)
                F = np.abs(f_array)
            # stickModelCalibration.mdof_material(self,n_storeys,f_array,d_array)
            
                # define the material tag associated with each storey
                mat1Tag = int(f'1{i}00')
                mat2Tag = int(f'1{i}01')
                
                # get the backbone curve definition
    
        
                # create rigid elastic materials for the restrained dofs
                rigM = int(f'1{i}02')
                ops.uniaxialMaterial('Elastic', rigM, 1e6)
        
                # create the material
                createPinching4Material(mat1Tag, F, D)
                ops.uniaxialMaterial('MinMax', mat2Tag, mat1Tag, '-min', -D[-1], '-max', D[-1])
                
                # aggregate all material tags in one
                matTags = [mat2Tag, mat2Tag, rigM, rigM, rigM, rigM]            
    
                # define the connectivity parameters
                eleTag = int(f'200{i}')
                eleNodes = [i, i+1]
                
                # create the element
                #ops.element('zeroLength', eleTag, eleNodes, '-mat', matTags, '-dir', *dirs)
                ops.element('zeroLength', eleTag, eleNodes[0], eleNodes[1], '-mat', mat2Tag, mat2Tag, rigM, rigM, rigM, rigM, '-dir', 1, 2, 3, 4, 5, 6, '-doRayleigh', 1)
            
            stickModelCalibration.do_gravity_analysis(self)
            
            spo_disps, spo_rxn=stickModelCalibration.do_spo_analysis(self,0.05*self.nst*self.storey_height,1)

            sds=np.array(spo_disps[:,-1])/self.gamma
            sas=((np.array(spo_rxn)/self.gamma)/total_mass)/9.81
            
            expected_sas=np.interp(sds, self.capacity_array[:,0], self.capacity_array[:,1])
            
            loss_squared=np.sum((expected_sas-sas)**2)

            return loss_squared
        optimize.Bounds(lb=3, ub=20, keep_feasible=False)
        initial_sol=np.linspace(12,3,self.nst).tolist()

        solution=optimize.fmin(func,initial_sol,disp=False)
        print(solution)
        return solution
        
##########################################################################
#                          MDOF MODELLING MODULE                         #
##########################################################################   
from utilities import *

def calibrateModel(nst, sdofCapArray, T_sdof, isInfilled, isSOS, isWall):    
    """
    Function to calibrate MDOF storey force-deformation relationships based on SDOF-based capacity functions
    -----
    Input
    -----
    :param nst:               int                MDOF system number of storeys
    :param sdofCapArray:    array                Array of SDOF spectral displacement and acceleration definition
    :param T_sdof:          float                Fundamental period of the SDOF system
    :param isInfilled:       bool                Boolean condition if building class is of the infilled class
    :param isSOS:            bool                Boolean condition if building class is soft-storey
    :param isWall:           bool                Boolean condition if building class uses walls as load-bearing

    ------
    Output
    ------
    flm_mdof:                list                MDOF floor masses
    stD_mdof:               array                MDOF storey displacements
    stF_mdof:               array                MDOF storey forces
    phi_mdof:                list                MDOF expected mode shape
       
    """
    ## If the building is infilled and it is NOT soft storey
    if (not isSOS and isInfilled) or isWall:
        
        if nst <= 4:        
            phi_mdof = np.zeros(nst)
            for i in range(nst):
                # These equations are based on empirical derivation of mode-shape vs number of storeys for infilled buildings with no soft-storey mechanism expected
                phi_mdof[i] = -0.9845*((i+1)/nst)**2 + 1.9747*((i+1)/nst) + 0.0041
            phi_mdof[-1] = 1.0
        elif nst > 4:
            phi_mdof = np.zeros(nst)
            for i in range(nst):
                # These equations are based on empirical derivation of mode-shape vs number of storeys for infilled buildings with no soft-storey mechanism expected
                phi_mdof[i] = -0.7348*((i+1)/nst)**2 + 1.7267*((i+1)/nst) + 0.0011                
            phi_mdof[-1] = 1.0 
    # If the building has a soft storey
    elif isSOS:
    
        if nst <= 4:
            phi_mdof = np.zeros(nst)
            for i in range(nst):
                # Initialise the mode shape (based on Priestley et al. 2007)
                phi_mdof[i] = -1.941*((i+1)/nst)**5 + 1.3378*((i+1)/nst)**4 + 5.8991*((i+1)/nst)**3 - 9.3336*((i+1)/nst)**2 + 5.0377*((i+1)/nst) + 0.000000008
            phi_mdof[-1] = 1.0
            
        elif nst > 4:            
            phi_mdof = np.zeros(nst)
            for i in range(nst):
                # Initialise the mode shape (based on Priestley et al. 2007)
                phi_mdof[i] = 6.7777*((i+1)/nst)**5 - 22.025*((i+1)/nst)**4 + 27.485*((i+1)/nst)**3 - 16.666*((i+1)/nst)**2 + 5.4282*((i+1)/nst) + 0.0004
            phi_mdof[-1] = 1.0                         

    # For all other building typologies
    else:
        
        if nst <= 4:
            # Initialise the mode shape
            phi_mdof = np.linspace(1/nst, 1, nst)
        elif nst > 4:
            phi_mdof = np.zeros(nst)            
            for i in range(nst):
                # Initialise the mode shape (based on Priestley et al. 2007)
                phi_mdof[i] = 4/3*(i+1)/nst*(1 - (i+1)/4/nst)

    
    # Calculate the sum of the squares of phi
    sum_square_phi = np.dot(phi_mdof, phi_mdof)

    # Calculate the sum of phi and then square it
    sum_phi_square = np.power(np.sum(phi_mdof), 2)
    
    # Calculate the sum of mode shape
    sum_phi = np.sum(phi_mdof)
    
    # Calculate the mass at each floor node knowing the mode shape, effective mass (1 unit ton) and transformation factor
    mass = sum_square_phi/sum_phi_square
    
    # mass = 1/(sum_square_phi*gamma)
    
    # Real Value of Gamma because of the asssumed mode shape
    gamma_real = sum_phi/sum_square_phi
            
    # Assign the MDOF mass
    flm_mdof = [mass]*nst

    ### Get the MDOF Capacity Curves Storey-Deformation Relationship
    rows, columns = np.shape(sdofCapArray)
    stD_mdof = np.zeros([nst,rows])
    stF_mdof = np.zeros([nst,rows])
    
    # Define the mass identity matrix (diagonal matrix that have 1). It assumes again that all masser are uniform
    I = np.identity(nst)
    
    # Define the stiffnes tri-diagonal matrix, which considers the stiffness to be uniform accross all stories
    ## Note: this may need to be changed later given that it does not apply to soft storeys
    
    # Initialize a zero matrix of size nst x nst
    K = np.zeros((nst, nst))
    
    # Fill the diagonal with 2k for all floors except the first and last, which get k
    np.fill_diagonal(K, 2)
    
    # For the last floors, the diagonal element is k, not 2k
    K[-1, -1] = 1

    # Fill the off-diagonal elements with -k (coupling between adjacent floors)
    for i in range(nst - 1):
        K[i, i + 1] = -1
        K[i + 1, i] = -1
    
    # Compute Lambda as per Lu et al (pay attention as one of the papers has a mistake)
    lamda = np.dot(np.dot(np.transpose(phi_mdof),I),phi_mdof)/np.dot(np.dot(np.transpose(phi_mdof),K),phi_mdof)
    
    # Compute the interstorey initial stiffness (in kN/m)
    k0 = lamda*4*pi**2*mass/T_sdof**2
    
    if len(sdofCapArray) == 3: # In case of trilinear capacity curve
        
        for i in range(nst):
            
            if i == 0:
                
                # get the force or spectral acceleration arrays at each storey
                # Note here that since we assume a mode shape and masses, then the real gamma
                # is different from the one in csv files. Technically, the multiplication outcome
                # of the factors after sdofCapArray should be equal to 1.0, which is the effective
                # mass of SDoF system
                stF_mdof[i,:] = sdofCapArray[:,1]*gamma_real*np.sum(phi_mdof)*mass
                
                # get the displacement or spectral displacement arrays at each storey
                stD_mdof[i,:] = sdofCapArray[:,0]*gamma_real*phi_mdof[i]
                
                # # Fix the initial stiffness to get the same period as the SDoF system
                # stD_mdof[i,0] = stF_mdof[i,0]/(k0/9.81)
                # ## NOTE: the k0 is divided by 9.81 to maintain unit consistency because
                # ## Moe multiplies all y-axis by g later in opensees
                
                # # Find the slope of the second branch of the capacity curve of first floor
                # # to use it for predicting displacements of the other floors
                # slope_2nd = (stF_mdof[0,1] - stF_mdof[0,0])/(stD_mdof[0,1] - stD_mdof[0,0])
                
            
            else:
                
                # Find the force contribution ratio, based on the mode shape (it works as
                # long as the masses are uniform across the floors).
                
                # This Ratio is disabled for now as we want the springs to have the full srength. The amount
                # of shear developing there will be determined by the analysis and force distribution. So it
                # is wrong to use the below ratio. Even Lu et al (2014) has removed this ratio in his later papers
                Ratio_force = np.sum(phi_mdof[i:])/np.sum(phi_mdof)
                
                # Multiply the Ratio with the Y-coordinates of the first floor (as it has the full base shear)
                stF_mdof[i,:] = stF_mdof[0,:]*Ratio_force 
                
                
                # Find the interstorey drift contribution ratio to be multiplied by the
                # interstorey drift of the first floor to give us the interstorey drift for
                # the remaining floors. If the mode shape is linear, this ratio will be
                # always one, so the storeys are drifting the same way. Note that floor
                # height is always assumed constant
                Ratio_disp = (phi_mdof[i]-phi_mdof[i-1])/phi_mdof[0]
                
                
                # Derive the displacements 
                stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
                
                # # Fix the initial stiffness to be the same as the first floor
                # stD_mdof[i,0] = stF_mdof[i,0]/(k0/9.81)
    
                # # Find the displacement of the second branch
                # stD_mdof[i,1] = stD_mdof[i,0] + (stF_mdof[i,1] - stF_mdof[i,0])/slope_2nd
                
                # The last displacement will stay the same
                
                # This is for the case that the ultimate displacement is less than
                # the previous ones. I basically scale the previous displacements
                # with the same scale of the ultimate displacement rather than
                # computing the previous displacements by maintaining the same
                # slopes as the first floor
                if stD_mdof[i,2] < stD_mdof[i,1]:
                    
                    stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
                
                
    elif len(sdofCapArray) == 2: # In case of bilinear capacity curve
                
                
        for i in range(nst):
            
            if i == 0:
                
                # get the force or spectral acceleration arrays at each storey
                # Note here that since we assume a mode shape and masses, then the real gamma
                # is different from the one in csv files. Technically, the multiplication outcome
                # of the factors after sdofCapArray should be equal to 1.0, which is the effective
                # mass of SDoF system
                stF_mdof[i,:] = sdofCapArray[:,1]*gamma_real*np.sum(phi_mdof)*mass
                
                
                # get the displacement or spectral displacement arrays at each storey
                stD_mdof[i,:] = sdofCapArray[:,0]*gamma_real*phi_mdof[i]
                
                # # Fix the initial stiffness to get the same period as the SDoF system
                # stD_mdof[i,0] = stF_mdof[i,0]/(k0/9.81)
                # ## NOTE: the k0 is divided by 9.81 to maintain unit consistency because
                # ## Moe multiplies all y-axis by g later in opensees
                
            else:
                
                # Find the force contribution ratio, based on the mode shape (it works as
                # long as the masses are uniform across the floors).
                
                # This Ratio is disabled for now as we want the springs to have the full srength. The amount
                # of shear developing there will be determined by the analysis and force distribution. So it
                # is wrong to use the below ratio. Even Lu et al (2014) has removed this ratio in his later papers
                Ratio_force = np.sum(phi_mdof[i:])/np.sum(phi_mdof)
                
                # Multiply the Ratio with the Y-coordinates of the first floor (as it has the full base shear)
                stF_mdof[i,:] = stF_mdof[0,:]*Ratio_force 
                                
                # Find the interstorey drift contribution ratio to be multiplied by the
                # interstorey drift of the first floor to give us the interstorey drift for
                # the remaining floors. If the mode shape is linear, this ratio will be
                # always one, so the storeys are drifting the same way. Note that floor
                # height is always assumed constant
                Ratio_disp = (phi_mdof[i]-phi_mdof[i-1])/phi_mdof[0]
                                
                # Derive the displacements 
                stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
                
                # # Fix the initial stiffness to be the same as the first floor
                # stD_mdof[i,0] = stF_mdof[i,0]/(k0/9.81)
                
                # This is for the case that the ultimate displacement is less than
                # the previous ones. I basically scale the previous displacements
                # with the same scale of the ultimate displacement rather than
                # computing the previous displacements by maintaining the same
                # slopes as the first floor
                if stD_mdof[i,1] < stD_mdof[i,0]:
                    
                    stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
   
 
    if len(sdofCapArray) == 4: # In case of quadrilinear capacity curve
        
        for i in range(nst):
            
            if i == 0:
                
                # get the force or spectral acceleration arrays at each storey
                # Note here that since we assume a mode shape and masses, then the real gamma
                # is different from the one in csv files. Technically, the multiplication outcome
                # of the factors after sdofCapArray should be equal to 1.0, which is the effective
                # mass of SDoF system
                stF_mdof[i,:] = sdofCapArray[:,1]*gamma_real*np.sum(phi_mdof)*mass
                
                
                # get the displacement or spectral displacement arrays at each storey
                stD_mdof[i,:] = sdofCapArray[:,0]*gamma_real*phi_mdof[i]
                
                # # Fix the initial stiffness to get the same period as the SDoF system
                # stD_mdof[i,0] = stF_mdof[i,0]/(k0/9.81)
                # ## NOTE: the k0 is divided by 9.81 to maintain unit consistency because
                # ## Moe multiplies all y-axis by g later in opensees
                
                # # Find the slope of the second branch of the capacity curve of first floor
                # # to use it for predicting displacements of the other floors
                # slope_2nd = (stF_mdof[0,1] - stF_mdof[0,0])/(stD_mdof[0,1] - stD_mdof[0,0])
                
                # # Find the slope of the third branch of the capacity curve of first floor
                # # to use it for predicting displacements of the other floors
                # slope_3rd = (stF_mdof[0,2] - stF_mdof[0,1])/(stD_mdof[0,2] - stD_mdof[0,1])
               
                
            
            else:
                
                # Find the force contribution ratio, based on the mode shape (it works as
                # long as the masses are uniform across the floors).
                
                # This Ratio is disabled for now as we want the springs to have the full srength. The amount
                # of shear developing there will be determined by the analysis and force distribution. So it
                # is wrong to use the below ratio. Even Lu et al (2014) has removed this ratio in his later papers
                Ratio_force = np.sum(phi_mdof[i:])/np.sum(phi_mdof)
                
                # Multiply the Ratio with the Y-coordinates of the first floor (as it has the full base shear)
                stF_mdof[i,:] = stF_mdof[0,:]*Ratio_force 
                
                
                # Find the interstorey drift contribution ratio to be multiplied by the
                # interstorey drift of the first floor to give us the interstorey drift for
                # the remaining floors. If the mode shape is linear, this ratio will be
                # always one, so the storeys are drifting the same way. Note that floor
                # height is always assumed constant
                Ratio_disp = (phi_mdof[i]-phi_mdof[i-1])/phi_mdof[0]
                
                
                # Derive the displacements 
                stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
                
                # # Fix the initial stiffness to be the same as the first floor
                # stD_mdof[i,0] = stF_mdof[i,0]/(k0/9.81)
    
                # # Find the displacement of the second branch
                # stD_mdof[i,1] = stD_mdof[i,0] + (stF_mdof[i,1] - stF_mdof[i,0])/slope_2nd
               
                # # Find the displacement of the third branch
                # stD_mdof[i,2] = stD_mdof[i,1] + (stF_mdof[i,2] - stF_mdof[i,1])/slope_3rd
                
                # # The last displacement will stay the same 


                # This is for the case that the ultimate displacement is less than
                # the previous ones. I basically scale the previous displacements
                # with the same scale of the ultimate displacement rather than
                # computing the previous displacements by maintaining the same
                # slopes as the first floor
                if stD_mdof[i,3] < stD_mdof[i,2]:
                    
                    stD_mdof[i,:] = stD_mdof[0,:]*Ratio_disp
                          
        
    return flm_mdof, stD_mdof, stF_mdof, phi_mdof


def createPinching4Material(mat1Tag, mat2Tag, F, D, degradation=False):   
    """
    Function to create Pinching4 Material Model used for the mdof_material object of stickModel
    -----
    Input
    -----
    :param mat1Tag:           int                Material Tag #1
    :param mat2Tag:           int                Material Tag #2
    :param F:               array                Array of storey forces
    :param D:               array                Array of storey displacements
    :param degradation:      bool                Boolean condition to enable/disable degradation in Pinching4

    -----
    Output
    -----
    None
 
    """

    f_vec=np.zeros([5,1])
    d_vec=np.zeros([5,1])
    
    # Bilinear
    if len(F)==2:
          #bilinear curve
          f_vec[1]=F[0]
          f_vec[4]=F[-1]
          
          d_vec[1]=D[0]
          d_vec[4]=D[-1]
          
          d_vec[2]=d_vec[1]+(d_vec[4]-d_vec[1])/3
          d_vec[3]=d_vec[1]+2*((d_vec[4]-d_vec[1])/3)
          
          f_vec[2]=np.interp(d_vec[2],D,F)
          f_vec[3]=np.interp(d_vec[3],D,F)
    
    # Trilinear
    elif len(F)==3:
          
          f_vec[1]=F[0]
          f_vec[4]=F[-1]
          
          d_vec[1]=D[0]
          d_vec[4]=D[-1]
          
          f_vec[2]=F[1]
          d_vec[2]=D[1]
          
          d_vec[3]=np.mean([d_vec[2],d_vec[-1]])
          f_vec[3]=np.interp(d_vec[3],D,F)
    
    # Quadrilinear
    elif len(F)==4:
          f_vec[1]=F[0]
          f_vec[4]=F[-1]
          
          d_vec[1]=D[0]
          d_vec[4]=D[-1]
          
          f_vec[2]=F[1]
          d_vec[2]=D[1]
          
          f_vec[3]=F[2]
          d_vec[3]=D[2]

    if degradation==True:
        matargs=[f_vec[1,0],d_vec[1,0],f_vec[2,0],d_vec[2,0],f_vec[3,0],d_vec[3,0],f_vec[4,0],d_vec[4,0],
                             -1*f_vec[1,0],-1*d_vec[1,0],-1*f_vec[2,0],-1*d_vec[2,0],-1*f_vec[3,0],-1*d_vec[3,0],-1*f_vec[4,0],-1*d_vec[4,0],
                             0.5,0.25,0.05,
                             0.5,0.25,0.05,
                             0,0.1,0,0,0.2,
                             0,0.1,0,0,0.2,
                             0,0.4,0,0.4,0.9,
                             10,'energy']
    else:   
        matargs=[f_vec[1,0],d_vec[1,0],f_vec[2,0],d_vec[2,0],f_vec[3,0],d_vec[3,0],f_vec[4,0],d_vec[4,0],
                             -1*f_vec[1,0],-1*d_vec[1,0],-1*f_vec[2,0],-1*d_vec[2,0],-1*f_vec[3,0],-1*d_vec[3,0],-1*f_vec[4,0],-1*d_vec[4,0],
                             0.5,0.25,0.05,
                             0.5,0.25,0.05,
                             0,0,0,0,0,
                             0,0,0,0,0,
                             0,0,0,0,0,
                             10,'energy']
    
    ops.uniaxialMaterial('Pinching4', mat1Tag,*matargs)
    ops.uniaxialMaterial('MinMax', mat2Tag, mat1Tag, '-min', -1*d_vec[4,0], '-max', d_vec[4,0])


def createHystereticMaterial(matTag, F, D, pinchX=0.80, pinchY=0.20, damageX= 0.01, damageY= 0.01):
    """
    Function to create Pinching4 Material Model used for the mdof_material object of stickModel
    -----
    Input
    -----

    :param matTag:            int                Material Tag 
    :param F:               array                Array of storey forces
    :param D:               array                Array of storey displacements
    :pinchX:                array                Pinching factor for strain (or deformation) during reloading
    :pinchY:                array                Pinching factor for stress (or force) during reloading
    :damageX:               array                Damage due to ductility: D1(mu-1)
    :damageY:               array                Damage due to energy: D2(Eii/Eult)

    -----
    Output
    -----
    None
    
    """

    # Bilinear
    if len(F)==2 and len(D)==2:
        # assign bilinear material
        ops.uniaxialMaterial('HystereticSM', matTag, '-posEnv', F[0], D[0], F[1], D[1], '-negEnv', -F[0], -D[0], -F[1], -D[1], '-pinch', pinchX, pinchY,'-damage', damageX, damageY, '-beta', 0)
    # Trilinear
    elif len(F)==3 and len(D)==3:
        # assign bilinear material
        ops.uniaxialMaterial('HystereticSM', matTag, '-posEnv', F[0], D[0], F[1], D[1], F[2], D[2], '-negEnv', -F[0], -D[0], -F[1], -D[1], -F[2], -D[2], '-pinch', pinchX, pinchY,'-damage', damageX, damageY, '-beta', 0)
    # Multilinear
    elif len(F)==4 and len(D)==4:
        # assign bilinear material
        ops.uniaxialMaterial('HystereticSM', matTag, '-posEnv', F[0], D[0], F[1], D[1], F[2], D[2], F[3], D[3],'-negEnv', -F[0], -D[0], -F[1], -D[1], -F[2], -D[2], -F[3], -D[3], '-pinch', pinchX, pinchY,'-damage', damageX, damageY, '-beta', 0)


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
        
        self.nst = nst 
        self.flh = flh
        self.flm = flm
        self.stF = stF 
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
        ops.node(0, *[0.0, 0.0, 0.0])   # is the star needed here?
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
        for i, node in enumerate(nodeList): # The enumerate helps to do a mixed loop (i and actual node name)
            if i==0:
                pass
            else:
                ops.load(node,0.0,0.0,-self.flm[i-1]*9.81, 0.0, 0.0, 0.0)

    def mdof_material(self):
        """
        Initialises the definition of MDOF storey material model
    
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
            mat1Tag = int(f'1{i}00') # hysteretic material tag
            mat2Tag = int(f'1{i}01') # min-max material tag
            
            ### get the backbone curve definition
            D = self.stD[i,:].tolist() # deformation capacity (i.e., storey displacement in m)
            F = self.stF[i,:].tolist() # strength capacity (i.e., storey base shear in kN)
                        
            ### Create rigid elastic materials for the restrained dofs
            rigM = int(f'1{i}02')
            ops.uniaxialMaterial('Elastic', rigM, 1e16)
                        
            ### Create the nonlinear material for the unrestrained dofs
            createPinching4Material(mat1Tag, mat2Tag, F, D, degradation = True)
            
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
        num_modes:                      int                Number of modes to consider (default is 3).
        solver:                      string                Type of solver (default is -genBandArpack).
        doRayleigh:                    bool                Flag to enable/disable Rayleigh damping
        pflag:                         bool                Flag to print (or not) the modal analysis report.
        
        Returns
        -------
        T:                            array                Periods of vibration.
        mode_shape                     list                First mode-based normalised mode-shape
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
        
        ## ADDED BY KARIM: MODE SHAPE FOR THE FIRST MODE
        mode_shape = []
        
        # Extract mode shapes for all nodes (displacements in x)
        for k in range(1, self.nst+1):
            ux = ops.nodeEigenvector(k, 1, 1)  # Displacement in x-direction
            mode_shape.append(ux)
        
        # Normalize the mode shape
        mode_shape = np.array(mode_shape)/mode_shape[-1]
        
        ### Wipe the analysis objects
        ops.wipeAnalysis()      
        
        return T, mode_shape
            
    def do_spo_analysis(self, ref_disp, disp_scale_factor, push_dir, phi, pflag=True, 
                        num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', 
                        numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-5, init_iter=1000, 
                        algorithm_type='KrylovNewton'):
        """
        Perform static pushover analysis on MDOF
    
        Parameters
        ----------
        ref_disp:                     float                Reference displacement to analyses are run. Corresponds to yield or equivalent other, such as 1mm.
        disp_scale_factor:            float                Multiple of ref_disp to which the push is run. So pushover can be run to a specified ductility or displacement.
        push_dir:                       int                Direction of pushover (1 = X; 2 = Y; 3 = Z)
        phi:                           list                Shape of lateral load applied, this is one (optional) output of the calibrateModel function and using it here assumes the applied pushover load is first-mode based
        pflag:                         bool                Flag to print (or not) the static pushover analysis steps
        num_steps:                      int                Number of spo analysis steps to perform (Default is 200).
        ansys_soe:                   string                System of equations type (Default is 'BandGeneral').
        constraints_handler:         string                The constraints handler object determines how the constraint equations are enforced in the analysis. Constraint equations enforce a specified value for a DOF, or a relationship between DOFs (Default is 'Transformation').
        numberer:                    string                The DOF numberer object determines the mapping between equation numbers and degrees-of-freedom – how degrees-of-freedom are numbered (Default is 'RCM').
        test_type:                   string                This command is used to construct the LinearSOE and LinearSolver objects to store and solve the test of equations in the analysis (Default is 'EnergyIncr').
        init_tol:                     float                Tolerance criteria used to check for convergence (Default is 1e-5).
        init_iter:                    float                Max number of iterations to check (Default is 1000).
        algorithm_type:              string                The integrator object determines the meaning of the terms in the system of equation object Ax=B (Default is 'KrylovNewton').
        
        Returns
        -------
        spo_disps:                    array                Displacements at each floor
        spo_rxn:                      array                Base shear as the sum of the reaction at the base 
        spo_disps_spring:             array                Displacements in the storey zero-length elements
        spo_forces_spring:            array                Shear forces in the storey zero-length elements
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
                    ops.load(pattern_nodes[i], phi[i], 0.0, 0.0, 0.0, 0.0, 0.0) ######### IT STARTS FROM ZERO

                    
                    # if len(pattern_nodes) <= 4:
                    #     ops.load(pattern_nodes[i], nodeList[i+1]/len(pattern_nodes), 0.0, 0.0, 0.0, 0.0, 0.0) ######### IT STARTS FROM ZERO
                    # elif len(pattern_nodes) >4:
                    #     ops.load(pattern_nodes[i], 4/3*nodeList[i+1]/len(pattern_nodes)*(1 - nodeList[i+1]/4/len(pattern_nodes)), 0.0, 0.0, 0.0, 0.0, 0.0) ######### IT STARTS FROM ZERO                        
            elif push_dir == 2:
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], 0.0, phi[i], 0.0, 0.0, 0.0, 0.0)
    
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
                
        # Get a list of all the element tags (zero-length springs)
        elementList = ops.getEleTags()
        
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
        
        # Recording displacements and forces in non-linear zero-length springs [the zero is needed to get the exact required value]
        spo_disps_spring = np.array([[ops.eleResponse(ele, 'deformation')[0] for ele in elementList]])
        spo_forces_spring = np.array([[ops.eleResponse(ele, 'force')[0] for ele in elementList]])

        
        # Start the adaptive convergence scheme
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
                ops.test('FixedNumIter', init_iter*10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            
            # This feature of disabling the possibility of having a negative loading has been included.
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
    
           
            spo_disps_spring = np.append(spo_disps_spring, np.array([
            [ops.eleResponse(ele, 'deformation')[0] for ele in elementList]
            ]), axis=0)
            
            
            spo_forces_spring = np.append(spo_forces_spring, np.array([
            [ops.eleResponse(ele, 'force')[0] for ele in elementList]
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
             
        return spo_disps, spo_rxn, spo_disps_spring, spo_forces_spring
    
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
        pflag:                         bool                Flag to print (or not) the static pushover analysis steps (Default is 'True').
        num_steps:                      int                Number of cpo analysis steps to perform (Default is 200).
        ansys_soe:                   string                System of equations type. (Default is 'BandGeneral')
        constraints_handler:         string                The constraints handler object determines how the constraint equations are enforced in the analysis. Constraint equations enforce a specified value for a DOF, or a relationship between DOFs (Default is 'Transformation').
        numberer:                    string                The DOF numberer object determines the mapping between equation numbers and degrees-of-freedom – how degrees-of-freedom are numbered (Default is 'RCM').
        test_type:                   string                This command is used to construct the LinearSOE and LinearSolver objects to store and solve the test of equations in the analysis (Default is 'NormDispIncr').
        init_tol:                     float                Tolerance criteria used to check for convergence (Default is 1e-5).
        init_iter:                    float                Max number of iterations to check (Default is 1000).
        algorithm_type:              string                The integrator object determines the meaning of the terms in the system of equation object Ax=B (Default is 'KrylovNewton').
        
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
                    ops.test('FixedNumIter', init_iter*10)
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
                    
    def do_nrha_analysis(self, fnames, dt_gm, sf, t_max, dt_ansys, nrha_outdir,
                         pflag=True, xi = 0.05, ansys_soe='BandGeneral', 
                         constraints_handler='Plain', numberer='RCM', 
                         test_type='NormDispIncr', init_tol=1.0e-6, init_iter=50, 
                         algorithm_type='Newton'):
        """
        Perform nonlinear time-history analysis on MDOF
    
        Parameters
        ----------
        fnames:                        list                List of the filepaths to the ground motions to be applied in the X Y and Z. At least the X direction is required.
        dt_gm:                        float                Time-step of the ground motions.
        sf:                           float                Scale factor to be applied to the records (Typically equal to 9.81).
        t_max:                        float                Duration of the record.
        dt_ansys:                     float                Time-step at which to conduct the analysis (Typically smaller than the record dt).
        nrha_outdir:                 string                Filepath where "TEMPORARY" files are saved and then deleted
        pflag:                         bool                Flag to print (or not) the nonlinear time-history analysis steps (Default is 'True').
        xi:                           float                The value of inherent damping (Default is 5% '0.05').
        ansys_soe:                   string                System of equations type. (Default is 'BandGeneral')
        constraints_handler:         string                The constraints handler object determines how the constraint equations are enforced in the analysis. Constraint equations enforce a specified value for a DOF, or a relationship between DOFs (Default is 'Plain').
        numberer:                    string                The DOF numberer object determines the mapping between equation numbers and degrees-of-freedom – how degrees-of-freedom are numbered (Default is 'RCM').
        test_type:                   string                This command is used to construct the LinearSOE and LinearSolver objects to store and solve the test of equations in the analysis (Default is 'NormDispIncr').
        init_tol:                     float                Tolerance criteria used to check for convergence (Default is 1e-6).
        init_iter:                    float                Max number of iterations to check (Default is 50).
        algorithm_type:              string                The integrator object determines the meaning of the terms in the system of equation object Ax=B (Default is 'Newton').
                        
        Returns
        -------
        control_nodes:                 list                List of MDOF system floor nodes
        conv_index:                    list                List containing whether or not analysis has converged (collapse index)
        peak_drift:                   array                Peak storey drift values (i.e., all storeys per record)
        peak_accel:                   array                Peak floor acceleration values (i.e., all floors per record)
        max_peak_drift:               array                Maximum peak storey drift values (i.e., maximum of all storeys per record)
        max_peak_drift_dir:           array                Direction of maximum peak storey drift value (i.e., X or Y)
        max_peak_drift_loc:           array                Location of maximum peak storey drift values (i.e., storey ID)
        max_peak_accel:               array                Maximum peak floor acceleration values (i.e., maximum of all floors per record)
        max_peak_accel_dir:           array                Direction of maximum peak floor acceleration value (i.e., X or Y)
        max_peak_accel_loc:           array                Location of maximum peak floor acceleration values (i.e., floor ID) 
        peak_disp:                    array                Peak displacement values (i.e., all floors per record) 
    
        """        
    
        # define control nodes
        control_nodes = ops.getNodeTags()

        # Define the timeseries and patterns first
        if len(fnames) > 0:
            nrha_tsTagX = 1
            nrha_pTagX = 1
            ops.timeSeries('Path', nrha_tsTagX, '-dt', dt_gm, '-filePath', fnames[0], '-factor', sf) 
            ops.pattern('UniformExcitation', nrha_pTagX, 1, '-accel', nrha_tsTagX)
            ops.recorder('Node', '-file', f"{nrha_outdir}/floor_accel_X.txt", '-timeSeries', nrha_tsTagX, '-node', *control_nodes, '-dof', 1, 'accel')
        if len(fnames) > 1:
            nrha_tsTagY = 2
            nrha_pTagY = 2
            ops.timeSeries('Path', nrha_tsTagY, '-dt', dt_gm, '-filePath', fnames[1], '-factor', sf) 
            ops.pattern('UniformExcitation', nrha_pTagY, 2, '-accel', nrha_tsTagY)
            ops.recorder('Node', '-file', f"{nrha_outdir}/floor_accel_Y.txt", '-timeSeries', nrha_tsTagY, '-node', *control_nodes, '-dof', 2, 'accel')
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
                ops.test('FixedNumIter', init_iter*10)
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
            temp1 = np.transpose(np.max(np.abs(np.loadtxt(f"{nrha_outdir}/floor_accel_X.txt")), 0))
            peak_accel[:,0] = temp1
            os.remove(f"{nrha_outdir}/floor_accel_X.txt")
        
        elif len(fnames) > 1:
            
            temp1 = np.transpose(np.max(np.abs(np.loadtxt(f"{nrha_outdir}/floor_accel_X.txt")), 0))
            temp2 = np.transpose(np.max(np.abs(np.loadtxt(f"{nrha_outdir}/floor_accel_Y.txt")), 0))
            peak_accel = np.stack([temp1, temp2], axis=1)
            os.remove(f"{nrha_outdir}/floor_accel_X.txt")
            os.remove(f"{nrha_outdir}/floor_accel_Y.txt")
        
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

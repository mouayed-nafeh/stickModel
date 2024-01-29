def mdof_model():
    """
    Initialising the definition of an MDOF system

    Parameters
    ----------
    None.
    
    Returns
    -------
    None.

    """    
    # import necessary libraries
    import openseespy.opensees as ops

    # set model builder
    ops.wipe() # wipe existing model
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    
def mdof_nodes(nst, floor_height, floor_mass):
    """
    Initialising the definition of MDOF nodes and their corresponding mass

    Parameters
    ----------
    nst: integer 
        Number of stories (e.g., for three-storey buildings --> nst = 3)
    
    floor_height: list of floats
        Floor heights (in m) (e.g., for three-storey buildings --> floor_height = [2.8, 3.0, 3.0])
    
    floor_mass: list of floats
        Floor masses (in tonnes) (e.g. for three-storey buildings --> floor_masses = [300, 300, 300])
    
    Returns
    -------
    None.

    """    
    # import necessary libraries
    import openseespy.opensees as ops
    
    # define base node (tag = 0)
    ops.node(0, *[0.0, 0.0, 0.0])
    # define floor nodes (tag = 1+)
    i = 1
    current_height = 0.0
    while i <= nst:
        print(i)
        nodeTag = i
        current_height = current_height + floor_height[i-1]
        current_mass = floor_mass[i-1]
        coords = [0.0, 0.0, current_height]
        masses = [current_mass, current_mass, current_mass, current_mass, current_mass, current_mass]
        ops.node(nodeTag,*coords)
        ops.mass(nodeTag,*masses)
        i=i+1
    
    # for i in range(nst):
    #     nodeTag = i+1
    #     coords = [0.0, 0.0, (i+1)*floor_height[i]]
    #     masses = [floor_mass[i], floor_mass[i], floor_mass[i], floor_mass[i], floor_mass[i], floor_mass[i]]
    #     ops.node(nodeTag,*coords)
    #     ops.mass(nodeTag,*masses)


def mdof_fixity():
    """
    Initialising the definition of MDOF nodes' boundary conditions

    Parameters
    ----------
    None.
    
    Returns
    -------
    None.

    """    
    
    # import necessary libraries
    import openseespy.opensees as ops
    
    # get list of model nodes
    nodeList = ops.getNodeTags()
    
    # impose boundary conditions
    for i in nodeList:
        
        if i==0:
            ops.fix(i,1,1,1,1,1,1)
        else:
            ops.fix(i,0,0,1,0,0,0)
 
def mdof_elastic_elements(A,E,G,Jxx,Iy,Iz):

    # good for testing
    
    # import necessary libraries
    import openseespy.opensees as ops
    
    nodeList = ops.getNodeTags()
    
    i = 0
    while i < len(nodeList)-1:
        #print(i)
        # define the connectivity parameters
        eleTag = i
        eleNodes = [i, i+1]
       
        transfTag = int(f'9{eleTag}')
        ops.geomTransf('Linear', transfTag, 0, 1, 0)
        # create the element
        ops.element('elasticBeamColumn',eleTag,*eleNodes,A,E,G,Jxx,Iy,Iz,transfTag)
        i=i+1
        
def mdof_loads(A):
    """
    Initialising the definition of MDOF loads based on floor mass

    Parameters
    ----------
    A: float
     floor surface area (in sq.m) 
     this assumes that all floor areas are identical along the height of the structure, therefore only
     a single float argument is required (e.g., for three-storey buildings, A = 2000)
        
    
    Returns
    -------
    None.

    """        
    # import necessary libraries
    import openseespy.opensees as ops
    
    # get floor mass
    FloorLoad = 5.0; # assuming a floor load 
    massFl = A*FloorLoad/9.81;
        
    # get list of model nodes
    nodeList = ops.getNodeTags()
    nst = len(nodeList)-1
    
    # create a plain load pattern with a linear timeseries
    ops.timeSeries('Linear', 101)
    ops.pattern('Plain',101,101)
    
    # load the nodes
    for i in nodeList:
        if i==0:
            pass
        else:
            ops.load(i,0.0,0.0,-massFl*9.81, 0.0, 0.0, 0.0)
            

def create_storeyCap_material(eleTag, matDef, F, D):
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
    # import necessary libraries
    import openseespy.opensees as ops
    
    
    if matDef == 'Bilinear' or 'bilinear':
        
        # define the material tag
        mat1Tag = int(f'98{eleTag}')
        print(mat1Tag)
        # define the material                
        ops.uniaxialMaterial('Steel01', mat1Tag, F[0], F[0]/D[0], 1e-6)
        # define the min max tag
        mat2Tag = int(f'99{eleTag}')
        # define the material
        ops.uniaxialMaterial('MinMax', mat2Tag, mat1Tag, '-min', -D[1], '-max', D[1])
        
        
    elif matDef == 'Trilinear' or 'trilinear':
        
        # define the material tag
        mat1Tag = int(f'98${eleTag}')
        # define the material                
        ops.uniaxialMaterial('HystereticSM', mat1Tag, '-posEnv', F[0], D[0], F[1], D[1], F[2], D[2], '-pinch', 1,1, '-damage', 1,1)
        
        # define the min max tag
        mat2Tag = int(f'99${eleTag}')
        # define the material
        ops.uniaxialMaterial('MinMax', mat2Tag, mat1Tag, '-min', -D[2], '-max', D[2])
        
    elif matDef == 'Quadlinear':
        
        # define the material tag
        mat1Tag = int(f'98${eleTag}')
        # define the material                
        ops.uniaxialMaterial('HystereticSM', mat1Tag, '-posEnv', F[0], D[0], F[1], D[1], F[2], D[2], F[3], D[3], '-pinch', 1,1, '-damage', 1,1)
        
        # define the min max tag
        mat2Tag = int(f'99${eleTag}')
        # define the material
        ops.uniaxialMaterial('MinMax', mat2Tag, mat1Tag, '-min', -D[3], '-max', D[3])
    
    return mat2Tag
    
def mdof_storeyCap(F,D,matDef):
    """
    Initialising the definition of MDOF storey capacities

    Parameters
    ----------
    A: float
     floor surface area (in sq.m) 
     this assumes that all floor areas are identical along the height of the structure, therefore only
     a single float argument is required (e.g., for three-storey buildings, A = 2000)
        
    Returns
    -------
    None.

    """
    # import necessary libraries
    import openseespy.opensees as ops
    
    # get list of model nodes
    nodeList = ops.getNodeTags()
    numEle = len(nodeList)-1
    dirs = [1,2,3,4,5,6]


    i = 0
    while i < numEle:

        # define the connectivity parameters
        eleTag = i
        eleNodes = [i, i+1]
        
        # define the force-deformation
        matTag = create_storeyCap_material(eleTag, matDef, F, D)
        
        # create rigid elastic materials for the restrained dofs
        rigM = int(f'100{eleTag}')
        ops.uniaxialMaterial('Elastic', rigM, 1e6)
        matTags = [matTag, matTag, rigM, rigM, rigM, rigM]
        
        # create the element
        #ops.element zeroLength $eleTag $iNode $jNode -mat $matTag -dir $dir <-doRayleigh $rFlag> <-orient $x $yp>
        ops.element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, '-dir', *dirs)
        i=i+1
    
    
    
    
    
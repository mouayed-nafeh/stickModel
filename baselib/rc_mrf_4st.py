import openseespy.opensees as ops
import numpy as np
from basics import units, rect_section
from analysis import analysis
import vfo.vfo as vfo

class rc_mrf_4st:
    
    """
    This is a simple script to create an RC MRF model and analyse
    The structure was designed for EC8 in L'Aquila
    It has been subsequently base isolated with friction pendulum bearings
    The structure is modelled here as elastic 
    
    TO DO
    Add spo
    Add nrha
    Add non-linearity
    
    """
    def __init__(self, isolated=False):
        self.isolated = isolated
        self.fc = 25.*units.MPa
        self.Ec = (3320.*np.power(self.fc/units.MPa, 0.5) + 6900.)*units.MPa
        self.muF = 0.05
        self.Reff = 5.0
        self.gap = 100*units.mm
        
        # Define the geometry
        self.bx = [6.0, 4.5, 3.5, 6.0, 3.5, 4.5, 6.0]
        self.by = [6.0, 4.0, 6.0]
        self.hz = [4.5, 3.5, 3.5, 3.5]
        
        # Define the sections
        self.h_beams = [0.6, 0.6, 0.55, 0.55]
        self.b_beams = [0.4, 0.4, 0.35, 0.35]
        self.hb_cols_ext = [0.4, 0.4, 0.35, 0.35]
        self.hb_cols_int = [0.45, 0.45, 0.4, 0.4]
        
        # Set the floor pressures
        self.q_live_typ = 2.0*units.kNm2
        self.q_snow = 2.2*units.kNm2
        
        self.gamma_conc = 25*units.kNm3
        self.tk_slab = 0.2*units.m
        self.g_slab = self.gamma_conc*self.tk_slab
        self.tk_slab_base = 0.3*units.m
        self.g_slab_base = self.gamma_conc*self.tk_slab_base
        self.g_finish = 1.0*units.kNm2
        g_k = 1.35
        q_k = 1.5
        p_k = 1.2
        psi = 0.3
        
        # Factor up the loads
        self.q_typ = g_k*(self.g_slab + self.g_finish) + psi*q_k*self.q_live_typ
        self.q_base = g_k*(self.g_slab_base + self.g_finish) + psi*q_k*self.q_live_typ
        self.q_roof = g_k*(self.g_slab) + psi*p_k*self.q_snow

        # Create the model
        ops.wipe() # Remove existing model
        ops.model('basic', '-ndm', 3, '-ndf', 6) # Set modelbuilder
        self.grid_point_weight = np.zeros((len(self.bx)+1, len(self.by)+1))
        self.base_nodes = []
        self.pattern_nodes_X = []
        self.pattern_nodes_Y = []
        self.ref_loads_X = []
        self.ref_loads_Y = []
        self.control_nodes = []
        
        # Define the nodes
        for k in np.arange(len(self.hz)+1):
            rigid_dia_nodes = []
            
            for i in np.arange(len(self.bx)+1):
                for j in np.arange(len(self.by)+1):
                    # Create the node tage
                    node_tag = int(str(i+1) + str(j+1) + str (k+1))
        
                    # Create the node
                    ops.node(node_tag, sum(self.bx[:i]), sum(self.by[:j]), sum(self.hz[:k]))
                    
                    # Append node tage if it is on the right gridlines
                    if i == 0:
                        self.pattern_nodes_X.append(node_tag)
                        self.ref_loads_X.append(k/len(self.hz))
                    if j == 0:
                        self.pattern_nodes_Y.append(node_tag)
                        self.ref_loads_Y.append(k/len(self.hz))
                        
                    
                    if self.isolated is True and k == 0:
                        # Create an additional set of base nodes
                        node_tag_base = int(str(i+1) + str(j+1) + str(k))
                        ops.node(node_tag_base, sum(self.bx[:i]), sum(self.by[:j]), sum(self.hz[:k]))
                        
                    if i == 3 and j == 1:
                        if self.isolated is True and k == 0:
                            self.control_nodes.append(node_tag_base)
                        self.control_nodes.append(node_tag)

                    # Set fixity
                    if self.isolated is False:
                        if k == 0:
                            ops.fix(node_tag, 1, 1, 1, 1, 1, 1)
                            self.base_nodes.append(node_tag)
                        elif k > 0:
                            rigid_dia_nodes.append(node_tag)
                    elif self.isolated is True:
                        if k == 0:
                            ops.fix(node_tag_base, 1, 1, 1, 1, 1, 1)
                            rigid_dia_nodes.append(node_tag)
                            self.base_nodes.append(node_tag_base)
                        elif k > 0:
                            rigid_dia_nodes.append(node_tag)
                    
                        
                    # get the tributary widths
                    if i == 0:
                        x_trib = 0.5*self.bx[i]
                    elif i == len(self.bx):
                        x_trib = 0.5*self.bx[i-1]
                    else:
                        x_trib = 0.5*(self.bx[i-1] + self.bx[i])
                        
                    if j == 0:
                        y_trib = 0.5*self.by[j]
                    elif j == len(self.by):
                        y_trib = 0.5*self.by[j-1]
                    else:
                        y_trib = 0.5*(self.by[j-1] + self.by[j])
                    
                    # Assign the mass
                    if self.isolated is False:
                        if k < len(self.hz):
                            q = self.q_typ
                        else:
                            q = self.q_roof                      
                    elif self.isolated is True:
                        if k == 0:
                            q = self.q_base
                        elif k > 0 and k < len(self.hz):
                            q = self.q_typ
                        else:
                            q = self.q_roof
                          
                    node_mass = q*x_trib*y_trib/units.g
                    ops.mass(node_tag, node_mass, node_mass, 0.0, 0.0, 0.0, 0.0)
                    self.grid_point_weight[i, j] += q*x_trib*y_trib # Add the weight of the gridlines to the overall array
                    
            # Set the rigid diaphram
            if k > 0:
                rNode = rigid_dia_nodes[13]
                rigid_dia_nodes.remove(rNode)
                ops.rigidDiaphragm(3, rNode, *rigid_dia_nodes)
                
        # Define the geomteric transform tag
        col_geo_tag = 1
        beam_x_geo_tag = 2
        beam_y_geo_tag = 3
        
        ops.geomTransf('Corotational', col_geo_tag, 0, 1, 0)
        ops.geomTransf('Linear', beam_x_geo_tag, 0, 1, 0)
        ops.geomTransf('Linear', beam_y_geo_tag, -1, 0, 0)
        
        def create_elastic_rc_element(eleTag, eleNodes, h, b, E_mod, transfTag):
            G_mod = 0.42*E_mod
            Area = rect_section.area(b,h)        
            Iy = rect_section.second_moment(h, b)
            Iz = rect_section.second_moment(b, h)
            Jxx = rect_section.torsional_constant(b, h)
            ops.element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, G_mod, Jxx, Iy, Iz, transfTag)  
        
        def create_fpb_isolator(eleTag, eleNodes, muF, Reff, W):
            Kinit = 1e5
            iso_tag = int('1' + str(eleTag))
            fix_tag = int('2' + str(eleTag))
            ops.uniaxialMaterial('Elastic', fix_tag, 1e20) # Use this for all other fixed dofs
            ops.uniaxialMaterial('Steel01', iso_tag, W*muF, Kinit, W/Reff/Kinit) # Use this when you want to actually model the isolator
            # ops.uniaxialMaterial('Elastic', iso_tag, W/Reff) # Use this when you want to get isolated period
            matTags = [iso_tag, iso_tag, fix_tag, fix_tag, fix_tag, fix_tag]
            dirs = [1, 2, 3, 4, 5, 6]
            ops.element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, '-dir', *dirs, '-doRayleigh', 0)


        # Define the columns
        for k in np.arange(len(self.hz)):
            for i in np.arange(len(self.bx)+1):
                for j in np.arange(len(self.by)+1):
                    
                    # Create the external columns
                    if i == 0 or i == len(self.bx) or j == 0 or j == len(self.by):
                        hb_col = self.hb_cols_ext
                    else:
                        hb_col = self.hb_cols_int
                        
                    eleNodes = [int(str(i+1) + str(j+1) + str(k+1)), int(str(i+1) + str(j+1) + str(k+2))]
                    eleTag = int('7' + str(i+1) + str(j+1) + str(k+1))
                    create_elastic_rc_element(eleTag, eleNodes, hb_col[k], hb_col[k], self.Ec, col_geo_tag)
                        
        # Define the beams
        for k in np.arange(len(self.hz)):
            for i in np.arange(len(self.bx)+1):
                for j in np.arange(len(self.by)+1): 
                    
                    eleNodes_x = [int(str(i+1) + str(j+1) + str(k+2)), int(str(i+2) + str(j+1) + str(k+2))]
                    eleNodes_y = [int(str(i+1) + str(j+1) + str(k+2)), int(str(i+1) + str(j+2) + str(k+2))]
                    eleTag_x = int('5' + str(i+1) + str(j+1) + str(k+2))
                    eleTag_y = int('6' + str(i+1) + str(j+1) + str(k+2))
                    
                    if i < len(self.bx):
                        create_elastic_rc_element(eleTag_x, eleNodes_x, self.h_beams[k], self.b_beams[k], self.Ec, beam_x_geo_tag)
                    if j < len(self.by):
                        create_elastic_rc_element(eleTag_y, eleNodes_y, self.h_beams[k], self.b_beams[k], self.Ec, beam_y_geo_tag)
        
        # Define the isolator elements
        if isolated == True:
            # Define the moat wall's impact material
            gap_tag = 20
            K1 = 1e9
            K2 = 1e9 
            sigy = -10000.0
            ops.uniaxialMaterial('ImpactMaterial', gap_tag, K1, K2, sigy, -self.gap)
            
            ops.node(1, 0.0, 0.0, 0.0)
            ops.node(2, sum(self.bx), 0.0, 0.0)
            ops.node(3, 0.0, sum(self.by), 0.0)
            ops.fix(1, 1, 1, 1, 1, 1, 1)
            ops.fix(2, 1, 1, 1, 1, 1, 1)
            ops.fix(3, 1, 1, 1, 1, 1, 1)
            
            # Position isolators
            for i in np.arange(len(self.bx)+1):
                for j in np.arange(len(self.by)+1): 
                    eleNodes = [int(str(i+1) + str(j+1) + '0'), int(str(i+1) + str(j+1) + '1')]
                    eleTag = int('9' + str(i+1) + str(j+1) + '0')
                    create_fpb_isolator(eleTag, eleNodes, self.muF, self.Reff, self.grid_point_weight[i, j])
                    
                    if i == 0 and j == 0:
                        ops.element('zeroLength', 201, *[1, int(str(i+1) + str(j+1) + '1')], '-mat', *[gap_tag, gap_tag], '-dir', *[1, 2], '-doRayleigh', 0)
                        ops.element('zeroLength', 202, *[int(str(i+1) + str(j+1) + '1'), 1], '-mat', *[gap_tag, gap_tag], '-dir', *[1, 2], '-doRayleigh', 0)
                    elif i == len(self.bx) and j == 0:
                        ops.element('zeroLength', 203, *[2, int(str(i+1) + str(j+1) + '1')], '-mat', *[gap_tag], '-dir', *[1], '-doRayleigh', 0)
                        ops.element('zeroLength', 204, *[int(str(i+1) + str(j+1) + '1'), 2], '-mat', *[gap_tag], '-dir', *[1], '-doRayleigh', 0)
                    elif i == 0 and j == len(self.by):
                        ops.element('zeroLength', 205, *[3, int(str(i+1) + str(j+1) + '1')], '-mat', *[gap_tag], '-dir', *[2], '-doRayleigh', 0)
                        ops.element('zeroLength', 206, *[int(str(i+1) + str(j+1) + '1'), 3], '-mat', *[gap_tag], '-dir', *[2], '-doRayleigh', 0)

                    
        # Define the damping to be applied
        self.T, self.omega = analysis.modal(6)

        # Use a rayleigh damping model
        w_i = self.omega[0]    # Use the first and third modes
        w_j = self.omega[2] 
        xi_i = 0.05
        xi_j = 0.05
        alpha_m = 2*w_i*w_j/(w_j*w_j-w_i*w_i)*(w_j*xi_i-w_i*xi_j)
        beta_k = 2*w_i*w_j/(w_j*w_j-w_i*w_i)*(-xi_i/w_j+xi_j/w_i)
        ops.rayleigh(alpha_m, 0.0, beta_k, 0.0)


    def spo(self, disp_mm, drn):
        control_node = self.control_nodes[-1]
        
        if drn == "X":
            push_dir = 1
            pattern_nodes = self.pattern_nodes_X
            ref_loads = self.ref_loads_X
        elif drn == "Y":
            push_dir = 2
            pattern_nodes = self.pattern_nodes_Y
            ref_loads = self.ref_loads_Y
        
        self.spo_disp, self.spo_rxn = analysis.spo(units.mm, disp_mm, control_node, push_dir, pattern_nodes, ref_loads, self.base_nodes, pflag=False)
        
    def nrha(self, fnames, dt_gm, sf, t_max, dt_ansys, drift_limit):
        self.coll_index, self.peak_drift, self.peak_accel, self.max_peak_drift, self.max_peak_drift_dir, self.max_peak_drift_loc, self.max_peak_accel, self.max_peak_accel_dir, self.max_peak_accel_loc, self.peak_disp = \
            analysis.nrha(fnames, dt_gm, sf, t_max, dt_ansys, drift_limit, self.control_nodes, pflag=False)
        
    def plot_model(self):
        vfo.plot_model(model='none', show_nodes='yes', show_nodetags='yes', show_eletags='yes', font_size=10, setview='3D', elementgroups=None, line_width=1, filename=None)
        
    def plot_modeshape(self, modenumber, scale):
        vfo.plot_modeshape(modenumber=modenumber, scale=scale)
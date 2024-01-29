# This is a simple script to create and analyse an SDOF
import openseespy.opensees as ops
import numpy as np
from analysis import analysis
from basics import units

class sdof:
    """
    Create an SDOF oscillator object in OpenSeesPy
    Author: Gerard J. O'Reilly

    """
    
    def __init__(self, F_y, mu, b, T, m=1.0, xi=0.05, damp_model='Tangent'):
        """
        Initialising the definition of an SDOF oscillator

        Parameters
        ----------
        F_y : float
            Yield strength of the SDOF system
        mu : float
            Ductility capacity at which the strength becomes zero in either direction.
        b : float
            Hardening ratio to the initial stiffness.
        T : float
            Initial period of the system in seconds.
        m : float, optional
            Mass of the SDOF. The default is 1.0.
        xi : TYPE, optional
            Fraction of critical damping to apply. The default is 0.05.
        damp_model : str, optional
            Damping model to use. Currently either 'Initial' or 'Tangent' stiffness proportional Rayleigh damping. The default is 'Tangent'.

        Returns
        -------
        None.

        """

        self.F_y = F_y
        self.mu = mu
        self.b = b
        self.T = T
        self.m = m
        self.fix_node = 1
        self.control_node = 2
        
        
        # Create an SDOF
        ops.wipe() # Remove existing model
        ops.model('basic', '-ndm', 1, '-ndf', 1) # Set modelbuilder

        # Create nodes and fixity
        ops.node(self.fix_node, 0.0)
        ops.node(self.control_node, 0.0,'-mass', m)
        ops.fix(self.fix_node, 1)
    
        # Define materials
        self.D_y = np.power(self.T/2.0/units.pi,2)*self.F_y/self.m
        D_u = self.mu*self.D_y
        
        ops.uniaxialMaterial('Steel01', 1, self.F_y, self.F_y/self.D_y, self.b)
        ops.uniaxialMaterial('MinMax', 2, 1,'-min',-D_u,'-max', D_u)
    
        # Define elements
        ops.element("zeroLength", 1, self.fix_node, self.control_node, '-mat', 2, '-dir', 1, '-doRayleigh', 1)
        
        # Define the damping
        omega = 2*units.pi/self.T
        alpha_m = 0.0
        beta_k = 2*xi/omega
        
        # apply the mode
        if damp_model == 'Initial':
            ops.rayleigh(alpha_m, 0.0, beta_k, 0.0)
        elif damp_model == 'Tangent':
            ops.rayleigh(alpha_m, 0.0, 0.0, beta_k)
        

    def modal_analysis(self, num_modes=1, solver='-fullGenLapack'):
        """
        Parameters
        ----------
        num_modes : int, optional
            Number of modes to request. The default is 1.
        solver : str, optional
            The type of solver to use. The default is '-genBandArpack'.

        Returns
        -------
        None.

        """
        self.T_model, self.omega_model = analysis.modal(num_modes, solver)
        
    def spo(self, mu):
        """
        
        Do a simple pushover using the method in analysis
        
        Parameters
        ----------
        mu : float
            This is the multiple of the yield displacement that the SDOF is pushed to.

        Returns
        -------
        spo_disp: numpy array
            This is the deformations of the SDOF.
            
        spo_rxn: numpy array
            This is the reaction forces of the SDOF

        """
        
        self.spo_disp, self.spo_rxn = analysis.spo_sdof(self.D_y, mu, self.control_node, self.fix_node, pflag=False, num_steps=200)
                
    def nrha(self, fname, dt_gm, sf, t_max, dt_ansys, def_limit):
        """
        
        Do a simple non-linear time history analysis using the method in analysis
        

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

        Returns
        -------
        coll_index : int
            Collapse index (-1 for non-converged, 0 for stable, 1 for collapsed).
        peak_def : float
            peak deformation
        peak_accel : numpy array
            array of the peak accelerations at each floor (determined at the control_nodes) in the directions excited. IN TERMS OF G

        """
        
        self.nrha_coll_index, self.nrha_peak_def, self.nrha_peak_accel = analysis.nrha_sdof(fname, dt_gm, sf, t_max, dt_ansys, def_limit, self.control_node)
        
    
##########################################################################
#                  INTENSITY MEASURE CALCULATOR MODULE                   #
##########################################################################
import numpy as np
import eqsig 

class intensityMeasure():
    
    def __init__(self, acc, dt, damping = 0.05):
        """
        Get intensity measures from a ground-motion record
        Parameters
        ----------
        acc: List
            Accelerations
        dt: float
            Time step of the accelerogram
        damping: float
            Damping
        """
        self.acc = acc
        self.dt  = dt
        self.damping = damping
        
    def get_spectrum(self):
        """
        Get response spectrum of a ground-motion record
        Parameters
        ----------
        acc: List
            Accelerations
        dt: float
            Time step of the accelerogram
        damping: float
            Damping
        
        Returns
        -------
        record.response_times: list of periods
        record.s_a: spectral accelerations
        """
            
        periods = np.linspace(0.0, 4.0, 200)  # compute the response for 100 periods between T=0.2s and 5.0s
        record = eqsig.AccSignal(self.acc, self.dt)
        record.generate_response_spectrum(response_times=periods)
        
        return record.response_times, record.s_a
    
    def get_sa(self, period):
        """
        Get spectral acceleration at a period
        Parameters
        ----------
        period: float
            Period at which to calculate the SA
        acc: List
            Accelerations
        dt: float
            Time step of the accelerogram
        
        Returns
        -------
        sa: float
            Spectral acceleration at the given period
        """        
        prd,sas = self.get_spectrum()
        return np.interp(period, prd, sas)
        
    def get_saavg(self, period):
    
        ### get period range
        period_range = np.linspace(0.2*self.period,1.5*self.period,10)
        
        ### get spectrum
        sas = [self.get_sa(i) for i in period_range]
        
        ### calculate saavg        
        saavg = np.prod(sas)**(1/10)
        
        return saavg
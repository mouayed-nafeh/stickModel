##########################################################################
#                                                                        #
#                          POSTPROCESSING MODULE                         #
#                                                                        #
##########################################################################

from utilities import *
import eqsig 

##########################################################################
#                                                                        #
#                     INTENSITY MEASURE PROCESSING                       #
#                                                                        #
##########################################################################

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
        periods = np.linspace(0.0, 4.0, 100)  # compute the response for 100 periods between T=0.0s and 4.0s
        record = eqsig.AccSignal(self.acc, self.dt)
        record.generate_response_spectrum(response_times=periods)
                
        return record.response_times, record.s_a
            
    def get_sa(self, prd, sas, period):
        """
        Get spectral acceleration for a period value
        Parameters
        ----------
        prd: list
            List of periods (obtained from get_spectrum method)
        sas: List
            List of spectral acceleration values (obtained from get_spectrum method)
        period: Float
            Conditioning period (for PGA, period = 0.0)
        
        Returns
        -------
        sa: float
            Spectral acceleration at the given period
        """
        
        ### Return the interpolated spectral acceleration value
        return np.interp(period, prd, sas)
    
    def get_saavg(self, prd, sas, period):
        """
        Get average spectral acceleration for a range of periods
        Parameters
        ----------
        prd: list
            List of periods (obtained from get_spectrum method)
        sas: List
            List of spectral acceleration values (obtained from get_spectrum method)
        period: Float
            Conditioning period
        
        Returns
        -------
        saavg: float
            Average spectral acceleration at the given period
        """        
        
        ### Define the period range
        period_range = np.linspace(0.2*period,1.5*period, 10)
        
        ### Interpolate for spectral acceleration values at each period within the range
        sa = [np.interp(i,prd,sas) for i in period_range]
        
        ### Return the average spectral acceleration
        return np.prod(sa)**(1/10)
    
    def get_FIV3(self, period, alpha, beta):
        """
        References:
        Dávalos H, Miranda E. Filtered incremental velocity: A novel approach in intensity measures for
        seismic collapse estimation. Earthquake Engineering & Structural Dynamics 2019; 48(12): 1384–1405.
        DOI: 10.1002/eqe.3205.

        Get the filtered incremental velocity IM for a ground motion
        Parameters
        ----------
            Period:     Float
            Period [s]
            alpha:      Float
            Period factor (see Figure 6)
            beta:       Float
            Cut-off frequency factor (see Figure 6)

        Returns:
            FIV3:       Intensity measure FIV3 (as per Eq. (3) of Davalos and Miranda (2019))
            FIV:        Filtered incremental velocity (as per Eq. (2) of Davalos and Miranda (2019))
            t:          Time series of FIV
            ugf:        Filtered acceleration time history
            pks:        Three peak values used to compute FIV3
            trs:        Three trough values used to compute FIV3
        """
        # Import required packages

        # Create the time series of the signal
        tim = self.dt*range(len(self.acc))

        # Apply a 2nd order Butterworth low pass filter to the ground motion
        Wn = beta/period/(0.5/self.dt)
        b, a = signal.butter(2, Wn, 'low')
        ugf = signal.filtfilt(b, a, self.acc)

        # Get the filtered incremental velocity
        FIV = np.array([])
        t = np.array([])
        for i in range(len(tim)):
            # Check if there is enough length in the remaining time series
            if tim[i] < tim[-1] - alpha*period:
                # Get the snippet of the filtered acceleration history
                ugf_pc = ugf[i:i+int(np.floor(alpha*period/self.dt))]

                # Integrate the snippet
                FIV = np.append(FIV, self.dt*integrate.trapz(ugf_pc))

                # Log the time
                t = np.append(t, tim[i])

        # Convert
        # Find the peaks and troughs of the FIV array
        pks_ind, _ = signal.find_peaks(FIV)
        trs_ind, _ = signal.find_peaks(-FIV)

        # Sort the values
        pks_srt = np.sort(FIV[pks_ind])
        trs_srt = np.sort(FIV[trs_ind])

        # Get the peaks
        pks = pks_srt[-3:]
        trs = trs_srt[0:3]

        # Compute the FIV3
        FIV3 = np.max([np.sum(pks), np.sum(trs)])

        return FIV3, FIV, t, ugf, pks, trs
    
    def get_amplitude_ims(self):
        
        """
        Get amplitude-based intensity measures of a ground-motion record
        Parameters
        ----------
        None        
        Returns
        -------
        record.pga: peak ground acceleration
        record.pgv: peak ground velocity
        record.pgd: peak ground displacement
        """
        record = eqsig.AccSignal(self.acc, self.dt)
        record.generate_displacement_and_velocity_series()
        return record.pga, record.pgv, record.pgd
    

    def get_arias_intensity(self, start = 0.05, end = 0.95):
        
        record = eqsig.AccSignal(self.acc, self.dt)    
        
        # Get the 5% and 95% time stamps
        t_start, t_end  = eqsig.im.calc_sig_dur_vals(self.acc, self.dt, start=start, end=end, se=True)
        # Get the index of the steps
        step_start = int(t_start/self.dt)
        step_end   = int(t_end/self.dt)
        
        # Get the arias intensity time-history
        AI = eqsig.im.calc_arias_intensity(record)
        # Get the arias intensity between the percentiles requested       
        
        return AI[step_end]-AI[step_start]


    def get_duration_ims(self):
        """
        Get duration-based intensity measures of a ground-motion record
        Parameters
        ----------
        None        
        Returns
        -------
        record.arias_intensity: arias intensity (AI)
        record.cav: cumulative absolute velocity (CAV)
        record.t_959: 5%-95% significant duration
        
        """        
        record = eqsig.AccSignal(self.acc, self.dt)
        record.generate_cumulative_stats()
        record.generate_duration_stats()
        return record.arias_intensity, record.cav, record.t_595

    def get_significant_duration(self, start = 0.05, end = 0.95):
        """
        Get significant duration of a ground-motion record
        Parameters
        ----------
        None        
        Returns
        -------
        record.arias_intensity: arias intensity (AI)
        record.cav: cumulative absolute velocity (CAV)
        record.t_959: 5%-95% significant duration
        
        """        
        return eqsig.im.calc_sig_dur_vals(self.acc, self.dt, start = start, end = end, se=False)
        
##########################################################################
#                                                                        #
#                  FRAGILITY/VULNERABILITY PROCESSING                    #
#                                                                        #
##########################################################################

def get_fragility_function(intensities, theta, beta_total):
    """
    Calculate the damage state lognormal CDF for a set of median seismic intensity and associated dispersion
    Parameters
    ----------
    intensities:                   list                Intensity measure levels 
    theta:                        float                Median seismic intensity given edp-based damage threshold.
    beta_total:                   float                Total uncertainty (i.e. accounting for record-to-record and modelling variabilities).
    
    Returns
    -------
    p:                             list                Probabilities of damage exceedance.
    """
    
    ### calculate probabilities of exceedance for a range of intensity measure levels
    p = stats.lognorm.cdf(intensities, s=beta_total, loc=0, scale=theta)
        
    return p


def do_cloud_analysis(imls,edps,damage_thresholds,lower_limit,censored_limit,sigma_build2build=0.3):

    """
    Function to perform censored cloud analysis to a set of engineering demand parameters and intensity measure levels
    Processes cloud analysis and fits linear regression after due consideration of collapse
    -----
    Input
    -----
    imls:                    list          Intensity measure levels 
    edps:                    list          Engineering demand parameters (e.g., maximum interstorey drifts, maximum peak floor acceleration, top displacements, etc.)
    damage_thresholds        list          EDP-based damage thresholds associated with slight, moderate, extensive and complete damage
    lower_limit             float          Minimum EDP below which cloud records are filtered out (Typically equal to 0.1 times the yield capacity which is a proxy for no-damage)
    censored_limit          float          Maximum EDP above which cloud records are filtered out (Typically equal to 1.5 times the ultimate capacity which is a proxy for collapse)
    sigma_build2build       float          Building-to-building variability or modelling uncertainty (Default is set to 0.3)

    ------
    Output
    ------
    cloud_dict:                     dict         Cloud analysis outputs (regression coefficients and data, fragility parameters and functions)
    
    """    

    # Convert to numpy array type
    if isinstance(imls, np.ndarray):
        pass
    else:
        imls = np.array(imls)
    if isinstance(edps, np.ndarray):
        pass
    else:
        edps = np.array(edps)

      
    x_array=np.log(imls)
    y_array=edps
    
    # remove displacements below lower limit
    bool_y_lowdisp=edps>=lower_limit
    x_array = x_array[bool_y_lowdisp]
    y_array = y_array[bool_y_lowdisp]
    
    # checks if the y value is above the censored limit
    bool_is_censored=y_array>=censored_limit
    bool_is_not_censored=y_array<censored_limit
    
    # creates an array where all the censored values are set to the limit
    observed=np.log((y_array*bool_is_not_censored)+(censored_limit*bool_is_censored))
    
    y_array=np.log(edps)
    
    def func(x):
          p = np.array([stats.norm.pdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed))],dtype=float)
          return -np.sum(np.log(p))
    sol1=optimize.fmin(func,[1,1,1],disp=False)
    
    def func2(x):
          p1 = np.array([stats.norm.pdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed)) if bool_is_censored[i]==0],dtype=float)
          p2 = np.array([1-stats.norm.cdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed)) if bool_is_censored[i]==1],dtype=float)
          return -np.sum(np.log(p1[p1 != 0]))-np.sum(np.log(p2[p2 != 0]))
    
    p_cens=optimize.fmin(func2,[sol1[0],sol1[1],sol1[2]],disp=False)
    
    # reproduce the fit
    xvec = np.linspace(np.log(min(imls)),np.log(max(imls)),endpoint=True)
    yvec = p_cens[0]*xvec+p_cens[1]
    
    # calculate probabilities of exceedance
    intensities = np.round(np.geomspace(0.05, 10.0, 50), 3) # sample the intensities
    thetas = [np.exp((np.log(x)-p_cens[1])/p_cens[0]) for x in damage_thresholds] # calculate the median seismic intensities via the regression coefficients
    betas = [np.sqrt((p_cens[2]/p_cens[0])**2+sigma_build2build**2)]*4            # calculate the total uncertainty accounting for the modelling uncertainty
    poes = np.zeros((len(intensities),len(damage_thresholds)))                    # initialise and calculate the probabilities of exceedances associated with each damage state
    for i in range((len(damage_thresholds))):
        poes[:,i] = get_fragility_function(intensities, thetas[i], betas[i])

    ## Package the outputs
    cloud_dict =     {'imls': imls,                                              # Input intensity measure levels
                      'edps': edps,                                              # Input engineering demand parameters
                      'lower_limit': lower_limit,                                # Input lower censoring limit
                      'upper_limit': censored_limit,                             # Input upper censoring limit
                      'damage_thresholds': damage_thresholds,                    # Input damage thresholds
                      'fitted_x': np.exp(xvec),                                  # fitted intensity measure range
                      'fitted_y': np.exp(yvec),                                  # fitted edps 
                      'intensities': intensities,                                # sampled intensities for fragility analysis
                      'poes': poes,                                              # probabilities of exceedance of each damage state (DS1 to DSi)
                      'medians': thetas,                                         # median seismic intensities (in g)
                      'betas_total': betas,                                      # associated total dispersion (accounting for building-to-building and modelling uncertainties)
                      'b1': p_cens[0],                                           # cloud analysis regression parameter (a in EDP = aIM^b)
                      'b0': p_cens[1],                                           # cloud analysis regression parameter (b in EDP = aIM^b)
                      'sigma': p_cens[2]}                                        # the standard error in the fitted regression
                                                
    return cloud_dict
    

def calculate_vulnerability(intensities,poes,consequenceModel):
    """
    Calculate the vulnerability function associated with a building class given the probabilities of exceedance and a consequence model

    Parameters
    ----------
    intensities:                   list                Intensity measure levels
    poes:                         array                Probabilities of exceedance associated with the damage states considered (size = Intensity measure levels x nDS)
    consequenceModel:              list                Damage-to-loss ratios
    
    Returns
    -------
    vulnCurve:                    array                Vulnerability functions containing two columns, first is intensities (fixed range) and expected loss ratios (the uncertainty is modelled separately)
    """    
    
    ### Do some consistency checks
    if len(consequenceModel)!=np.size(poes,1):
          raise Exception('Mismatch between the fragility consequence models!')
    if len(intensities)!=np.size(poes,0):
          raise Exception('Mismatch between the number of IMLs and fragility models!')
    
    lossArray=np.zeros([len(intensities),1])
    for i in range(len(intensities)):
          for j in range(0,np.size(poes,1)):
                if j==(np.size(poes,1)-1):
                      lossArray[i,0]=lossArray[i,0]+poes[i,j]*consequenceModel[j]
                else:
                      lossArray[i,0]=lossArray[i,0]+(poes[i,j]-poes[i,j+1])*consequenceModel[j]
                          
    tmp=np.column_stack((intensities,lossArray))
    vulnCurve=tmp[tmp[:,0].argsort()]
    
    return vulnCurve

def calculate_structural_vulnerability(intensities, thetas, betas, consequenceModel):
    """
    Calculate the structural vulnerability function associated with a building class given the median seismic intensities and associated dispersion and a consequence model
    The probabilities of exceedance are calculated internally in this routine

    Parameters
    ----------
    intensities:                   list                Intensity measure levels
    thetas:                        list                Median seismic intensities corresponding to a 50% exceedance of any damage state
    betas:                         list                Total uncertainty associated with the median seismic intensity of each damage state
    consequenceModel:              list                Damage-to-loss models
    
    Returns
    -------
    structural_vulnerability:     array                Structural vulnerability functions containing two columns, first is intensities (fixed range) and expected loss ratios (the uncertainty is modelled separately)
    """    
    
    ### Assert data format
    if isinstance(thetas, np.ndarray):
        thetas = list(thetas)
    else:
        pass
    if isinstance(betas, np.ndarray):
        betas = list(betas)
    else:
        pass
        
    ### Initialise array to store damage probabilities
    poes = np.zeros((len(intensities),len(thetas)))
    
    ### Loop over the damage states: slight, moderate, extensive, complete
    for i in range(len(thetas)):
        poes[:,i]= get_fragility_function(intensities, thetas[i], betas[i])
        # check for NaNs 
        if np.isnan(poes[:,i]).all():
            print('NaN values encountered')
            poes[:,i]=0.0
        else:
            pass
        
    structural_vulnerability = calculate_vulnerability(intensities,poes,consequenceModel)
    
    return structural_vulnerability


def calculate_contents_vulnerability(intensities, theta, beta):
    """
    Calculate the contents vulnerability function associated with a building class given the median seismic intensities and associated dispersion associated
    with complete damage. The probabilities of exceedance are calculated internally in this routine

    Parameters
    ----------
    intensities:                  list                Intensity measure levels
    theta:                        list                Median seismic intensities corresponding to a 50% exceedance of any damage state
    beta:                         list                Total uncertainty associated with the median seismic intensity of each damage state
    
    Returns
    -------
    content_vulnerability:        array                Content vulnerability functions containing two columns, first is intensities (fixed range) and expected loss ratios (the uncertainty is modelled separately)
    """        
            
    ### Initialise array to store damage probabilities
    poes = np.zeros((len(intensities),1))
    
    ### Calculate complete damage probabilities
    poes[:,0]= get_fragility_function(intensities, theta[0], beta[0])
    # check for NaNs 
    if np.isnan(poes[:,0]).all():
        print('NaN values encountered')
        poes[:,0]=0.0
    else:
        pass
    
    content_vulnerability = calculate_vulnerability(intensities,poes,[1])
    
    return content_vulnerability


def calculate_business_interruption_vulnerability(intensities, thetas, betas, consequenceModel, normalised = False):
    """
    Calculate the (absolute or relative) business interruption vulnerability function associated with a building class given the median seismic
    intensities and associated dispersion corresponding to structural damage and a consequence model expressed typically in terms of downtime    
    The probabilities of exceedance are calculated internally in this routine

    Parameters
    ----------
    intensities:                   list                Intensity measure levels
    thetas:                        list                Median seismic intensities corresponding to a 50% exceedance of any damage state
    betas:                         list                Total uncertainty associated with the median seismic intensity of each damage state
    consequenceModel:              list                Downtimes associated with each structural DS 
    normalised:                    bool                Flag to return normalised (True) or absolute (False) values on the vulnerability function's y-axis

    Returns
    -------
    business_vulnerability:       array                Business interruption vulnerability functions containing two columns, first is intensities (fixed range) and expected loss ratios (the uncertainty is modelled separately)
    """        

    ### Assert data format
    if isinstance(thetas, np.ndarray):
        thetas = list(thetas)
    else:
        pass
    if isinstance(betas, np.ndarray):
        betas = list(betas)
    else:
        pass
        
    ### Initialise array to store damage probabilities
    poes = np.zeros((len(INTENSITIES),len(thetas)))
    
    ### Loop over the damage states: slight, moderate, extensive, complete
    for i in range(len(thetas)):
        poes[:,i]= get_fragility_function(intensities, thetas[i], betas[i])
        # check for NaNs 
        if np.isnan(poes[:,i]).all():
            print('NaN values encountered')
            poes[:,i]=0.0
        else:
            pass
        
    business_vulnerability = calculate_vulnerability(intensities,poes,consequenceModel)
    
    if normalised: 
        business_vulnerability[:,1] = business_vulnerability[:,1]/np.max(business_vulnerability[:,1]) 
    else:
        pass
    
    return business_vulnerability 

def calculate_building_vulnerability(structural_vulnerability, nonstructural_vulnerability, occupancyType):
    """
    Calculate the building vulnerability function associated with a building class via a factor-based combination of
    the structural and non-structural vulnerability functions and for an occupancy type (i.e., residential, commercial 
    and industrial)
    
    Parameters
    ----------
    structural_vulnerability:                   array                Y-axis value of the structural vulnerability
    nonstructural_vulnerability:                array                Y-axis value of the nonstructural vulnerability
    occupancyType:                             string                Occupancy type (i.e., RES, COM, IND)

    Returns
    -------
    building_vulnerability:                     array                Business interruption vulnerability functions containing two columns, first is intensities (fixed range) and expected loss ratios (the uncertainty is modelled separately)
    """        
    
    if occupancyType.lower() == 'res':
        sv  = 3.0/8.0 
        nsv = 5.0/8.0
    elif occupancyType.lower() == 'com':
        sv = 2.0/5.0
        nsv = 3.0/5.0
    elif occupancyType.lower() == 'ind':
        sv  = 3.0/8.0 
        nsv = 5.0/8.0
    
    building_vulnerability = np.zeros((struct_vulnerability.shape[0],1)) 
    building_vulnerability = sv*structural_vulnerability + nsv*nonstruct_vulnerability
    
    return building_vulnerability

def calculate_total_vulnerability(structural_vulnerability, nonstructural_vulnerability, contents_vulnerability, occupancyType):
    """
    Calculate the total vulnerability function associated with a building class via a factor-based combination of
    the structural, non-structural and contents vulnerability functions and for an occupancy type (i.e., residential, commercial 
    and industrial)
    
    Parameters
    ----------
    structural_vulnerability:    array                Y-axis value of the structural vulnerability
    nonstructural_vulnerability: array                Y-axis value of the nonstructural vulnerability
    contents_vulnerability:      array                Y-axis value of the contents vulnerability
    occupancyType:               string                Occupancy type (i.e., RES, COM, IND)

    Returns
    -------
    total_vulnerability:          array                Business interruption vulnerability functions containing two columns, first is intensities (fixed range) and expected loss ratios (the uncertainty is modelled separately)
    """        

    if occupancyType.lower() == 'res':
        sv  = 0.30 
        nsv = 0.50
        cv  = 0.20
    elif occupancyType.lower() == 'com':
        sv  = 0.20 
        nsv = 0.30
        cv  = 0.50
    elif occupancyType.lower() == 'ind':
        sv  = 0.15 
        nsv = 0.25
        cv  = 0.60 
        
    total_vulnerability= np.zeros((structural_vulnerability.shape[0],1))
    total_vulnerability = sv*structural_vulnerability + nsv*nonstructural_vulnerability + cv*contents_vulnerability
    
    return total_vulnerability

def calculate_sigma_loss(mean_loss_ratio):
    """
    Function to calculate the sigma in loss estimates according to:
    'Silva, V. (2019) Uncertainty and correlation in seismic vulnerability 
    functions of building classes. Earthquake Spectra. 
    DOI: 10.1193/013018eqs031m.'

    -----
    Input
    -----
    :param mean_loss_ratio:         list          List containing the mean loss ratios 

    ------
    Output
    ------
    sigma_loss_ratio:               list          The uncertainty associated with mean loss ratio
    a_beta_dist:                   float          coefficient of the beta-distribution
    b_beta_dist:                   float          coefficient of the beta_distribution
    
    """    
    
    # calculates the uncertainty in the mean loss ratio
    if mean_loss_ratio==0:
          sigma_loss_ratio=np.array([0])
    elif mean_loss_ratio==1:
          sigma_loss_ratio=np.array([1])
    else:
          sigma_loss_ratio=np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
          
    # calculates a and b parameters for beta distribution
    a_beta_dist=((1-mean_loss_ratio)/sigma_loss_ratio**2-(1/mean_loss_ratio))*mean_loss_ratio**2
    b_beta_dist=a_beta_dist*((1/mean_loss_ratio)-1)
    
    return sigma_loss_ratio,a_beta_dist,b_beta_dist


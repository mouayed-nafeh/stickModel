##########################################################################
#                          POSTPROCESSING MODULE                         #
##########################################################################
import numpy as np
from utils import *
from scipy import stats, optimize, interpolate
import piecewise_regression
from scipy.optimize import minimize

def cloudAnalysis(im,edp):
    """
    Processes cloud analysis results adaptively by fitting either a piecewise or 1st order linear regression

    Parameters
    ----------
    im:                            list                List of intensity measures.
    edp:                           list                List of engineering demand parameters (e.g., maximum peak storey drift).
    
    Returns
    -------
    im_fitted:                     list                List of predicted intensity measures.
    edp_fitted:                    list                List of equally spaced edp values sampled based on the minimum and maximum observed edps.
    """        
    
    ### Fit the piecewise regression in the log-log space
    pw_fit = piecewise_regression.Fit(np.log(edp), np.log(im), n_breakpoints=1)
    
    ### Print the summary of the regression
    pw_fit.summary()    
    pw_results = pw_fit.get_results()
    
    ### Calculate variability and resulting regression
    if pw_results['converged']:
            
        ### Reproduce the function
        xx = np.linspace(np.log(np.min(edp)), np.log(np.max(edp)), 100)
        yy = pw_fit.predict(xx)
        
        ### Return the values
        im_fitted = np.exp(yy)
        edp_fitted= np.exp(xx)    

    else:
        
        ### Fit linear regression
        p = np.polyfit(np.log(edp), np.log(im), 1)
        b = p[0]
        a = p[1]
                    
        ### Reproduce the function
        xx = np.linspace(np.log(np.min(edp)), np.log(np.max(edp)), 100)
        yy = b*xx+a
        
        ### Return the values
        im_fitted = np.exp(yy)
        edp_fitted = np.exp(xx)
        
    return im_fitted, edp_fitted

def calculateFragParams(edp, im, edp_fitted, im_fitted, damageThreshold, beta_build2build =0.3):
    """
    Calculates the median seismic intensity associated with an edp-based damage threshold based on cloud analysis regression

    Parameters
    ----------
    im:                            list                List of intensity measures.
    edp:                           list                List of engineering demand parameters (e.g., maximum peak storey drift).
    damageThreshold:              float                EDP-based damage threshold (e.g. for 5% maximum peak storey drift, damageThreshold = 0.05).
    beta_build2build:             float                Building-to-building variability or modelling uncertainty (default is 0.3).
    
    Returns
    -------
    theta:                        float                Median seismic intensity given edp-based damage threshold.
    beta_total:                   float                Total uncertainty (i.e. accounting for record-to-record and modelling variabilities). 

    """        
    
    
    ### Fit the piecewise regression in the log-log space
    pw_fit = piecewise_regression.Fit(np.log(edp), np.log(im), n_breakpoints=1)
    
    ### Print the summary of the regression
    pw_fit.summary()    
    pw_results = pw_fit.get_results()
    
    ### Calculate variability and resulting regression
    if pw_results['converged']:
        
        print('Piecewise Regression Converged')
        
        ### Reproduce the function
        xx = np.linspace(np.log(np.min(edp)), np.log(np.max(edp)), 100)
        yy = pw_fit.predict(xx)
        
        ### Calculate the standard deviation
        y_true = np.log(im)
        y_pred = pw_fit.predict(np.log(edp))
        beta = RSE(y_true, y_pred)
    
        ### Calculate the total variability accounting for the modelling uncertainty
        beta_total = np.sqrt(beta**2+beta_build2build**2)

    
    else:
        
        print('Piecewise Regression Did Not Converge...Fitting Simple Linear Regression')
       
        ### Fit linear regression
        p = np.polyfit(np.log(edp), np.log(im), 1)
        b = p[0]
        a = p[1]
                    
        ### Reproduce the function
        xx = np.linspace(np.log(np.min(edp)), np.log(np.max(edp)), 100)
        yy = b*xx+a
        
        ### Calculate the standard deviation
        y_true = np.log(im)
        y_pred = b*np.log(edp)+a
        beta = RSE(y_true, y_pred)
    
        ### Calculate the total variability accounting for the modelling uncertainty
        beta_total = np.sqrt(beta**2+beta_build2build**2)
            
    f = interpolate.interp1d(edp_fitted, im_fitted, fill_value='extrapolate')
    theta = f(damageThreshold)
        
    return theta, beta_total    

def getDamageProbability(theta, beta_total):
    """
    Calculate the damage state lognormal CDF for a set of median seismic intensity and associated dispersion

    Parameters
    ----------
    theta:                        float                Median seismic intensity given edp-based damage threshold.
    beta_total:                   float                Total uncertainty (i.e. accounting for record-to-record and modelling variabilities).
    
    Returns
    -------
    imls:                          list                Intensity measure levels.
    p:                             list                Probabilities of damage exceedance.
    """        

    ### calculate probabilities of exceedance for a range of intensity measure levels
    imls = np.linspace(0.0,5.0,1000)
    p = stats.norm.cdf(np.log((np.linspace(0.0, 5.0, 1000))/ theta) / beta_total, loc=0, scale=1)
    
    return imls, p


def mle_fit(IM = [], trials = [], obs = [], x0 = [0.8, 0.4]):
  """
  Fits a lognormal CDF curve according to Baker, J. W. (2015). Earthquake Spectra, 31(1), 579-599.

  INPUTS:
  
  IM-------------1xn              IM Levels of interest      
  
  trials---------1x1 or 1xn       number of ground motions @ IM level 
  
  obs------------1xn              number of occurrences observed @ IM level
  
  x0-------------[theta, beta]    initial solution (optional)
  
  OUTPUTS:
  
  array[theta, beta]              median and dispersion of fragility function
  
  EXAMPLE: (Same as on Baker's video)
  
  mle_fit(IM=[0.2,0.3,0.4,0.6,0.7,0.8,0.9,1.0], 
          trials = 40,
          obs = [0,0,0,4,6,13,12,16])
  
  --output: array([1.07611623, 0.42923779])
  """

  if len(trials if type(trials) is list else [trials]) == 1:
    trials = np.repeat(trials, len(IM))

  def func(x):
    theta, beta = x
    poes = obs/trials #calculates the observed probability of exceedance based on number of occurrences over number of records tested
    poes_fitted = stats.lognorm.cdf(IM, beta, scale = theta) #evaluates the probability calculated under the initial assumption of theta and beta
    likelihood = stats.binom.pmf(k = obs, n = trials, p = poes_fitted) #calculates the likelihood of the parameters being correct
    return -np.sum(np.log(likelihood)) #sum of the logs of the likelihood to be minimized

  sol = minimize(func, x0 = x0, method = 'SLSQP', bounds= ((0,None),(0, None))) #this finds a solution, bounds prevent negative values, check documentation for other settings

  return sol.x
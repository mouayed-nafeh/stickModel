##########################################################################
#                          POSTPROCESSING MODULE                         #
##########################################################################
import numpy as np
import math
from utils import *
from scipy import stats, optimize, interpolate
import piecewise_regression
from scipy.optimize import minimize

def getDamageProbability(sampled_ims, theta, beta_total):
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
    p = stats.norm.cdf(np.log(sampled_ims/ theta) / beta_total, loc=0, scale=1)
    
    return p


def cloudAnalysis(imls,edps,damage_thresholds,sigma_build2build=0.3):
    """
    Processes cloud analysis and fits linear regression after due consideration of collapse

    Parameters
    ----------
    imls:                          list                List of intensity measures.
    edps:                          list                List of engineering demand parameters (e.g., maximum peak storey drift).
    damage_thresholds:             list                List of edp-based damage thresholds
    
    Returns
    -------
    fitted_regression:             array               Array of edps (i.e., fitted_regression[:,0])  vs im (i.e., fitted regression[:,1])
    fragility_array:               array               Array of sampled ims vs probability of damage (first column of the array are the intensity measure levels and the remaining columns are the probability of damage values for each DS)
    theta:                         array               Array of median seismic intensities [g]
    beta_total:                    float               Total uncertainty (considering record-to-record and building-to-building variabilities)
    """        
    
    censored_limit = 1.50*damage_thresholds[-1]
    min_edp = 0.1*damage_thresholds[0]
    
    # Convert to numpy array type
    if isinstance(imls, np.ndarray):
        pass
    else:
        imls = np.array(imls)
    if isinstance(edps, np.ndarray):
        pass
    else:
        edps = np.array(edps)
        
    # Filter the arrays
    imls=imls[edps>=min_edp]
    edps=edps[edps>=min_edp]

    x_array=np.log(imls)
    y_array=edps
    
    # checks if the y value is above the censored limit
    bool_is_censored=y_array>=censored_limit
    bool_is_not_censored=y_array<censored_limit
    
    # creates an array where all the censored values are set to the limit
    observed=np.log((y_array*bool_is_not_censored)+(censored_limit*bool_is_censored))
    
    y_array=np.log(edps)
    
    def func(x):
          p = np.array([stats.norm.pdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed))],dtype=float)
          return -np.sum(np.log(p[p!= 0]))
    sol1=optimize.fmin(func,[1,1,1],disp=False)
    
    def func2(x):
          p1 = np.array([stats.norm.pdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed)) if bool_is_censored[i]==0],dtype=float)
          
          p2 = np.array([1-stats.norm.cdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed)) if bool_is_censored[i]==1],dtype=float)
          return -np.sum(np.log(p1[p1 != 0]))-np.sum(np.log(p2[p2 != 0]))
    
    p_cens=optimize.fmin(func2,[sol1[0],sol1[1],sol1[2]],disp=False)
    
    
    beta_edp=np.array([p_cens[0],p_cens[1]])
          
    sse=np.empty([len(y_array),1])
    ssto=np.empty([len(y_array),1])
    for i in range(len(y_array)):
        sse[i]=(y_array[i]-(p_cens[0]*x_array[i]+p_cens[1]))**2
        ssto[i]=(y_array[i]-np.mean(y_array,dtype=float))**2
    
    r_square=np.ones(len(damage_thresholds))*(1-(np.sum(sse)/np.sum(ssto)))
    sigma_edp = math.sqrt((1-r_square[0])**2+sigma_build2build)
    
    x_vec=np.linspace(np.log(0.05),np.log(15),endpoint=True)
    probability_damage_state=np.zeros([len(x_vec),len(damage_thresholds)])
    for i in range(len(x_vec)):
          mu=p_cens[0]*x_vec[i]+p_cens[1]
          for j in range(len(damage_thresholds)):
                probability_damage_state[i][j]=1-(stats.norm.cdf(np.log(damage_thresholds[j]),loc=mu,scale=sigma_edp))
                
    
    # Reproduce the function
    xx = np.linspace(np.log(np.min(imls)), np.log(np.max(imls)), 100)
    yy = p_cens[0]*xx+p_cens[1]
    
    # Pack the regression array
    xx = np.exp(xx)
    yy = np.exp(yy)
    regression_array = np.column_stack((xx,yy))
    
    # Get median intensities and dispersion for each DS
    theta = np.zeros([1,len(damage_thresholds)])
    beta_total = np.zeros([1,len(damage_thresholds)])
    for i in range(len(damage_thresholds)):
        theta[0,i] = np.exp(p_cens[0]*np.log(damage_thresholds[i])+p_cens[1])
        beta_total[0,i] = sigma_edp
    
    # Get fragility functions for each damage state
    sampled_ims = np.linspace(0,10,1000)
    poes = np.zeros([len(sampled_ims),len(damage_thresholds)])
    for i in range(len(damage_thresholds)):
        poes[:,i] = getDamageProbability(sampled_ims,theta[0,i], beta_total[0,i])
    
    # Pack the fragility array
    fragility_array = np.column_stack((sampled_ims, poes))  

    return fragility_array, regression_array, theta, beta_total

## load dependencies
import openseespy.opensees as ops
import pandas as pd
import numpy as np
import os
from mdof_units import g
from utils import *           
import matplotlib.pyplot as plt
import itertools

class stickModelCalibration():

    def __init__(self, ):
        """
        Stick Modelling and Processing Tool
        :param nst:         int                Number of storeys
        :param flh:        list               List of floor heights in metres (e.g. [2.5, 3.0])
        :param flm:        list               List of floor masses in tonnes (e.g. [1000, 1200])
        :param fll:        list               List of floor loads in kN/m2 (e.g. [5.0, 5.0])
        :param fla:        list               List of floor areas in m2 (e.g. [350, 350])
        :param structTypo:  str               String from GEM taxonomy (e.g. 'CR_LDUAL+CDH+DUH_H1')
        :param gitDir:      str               String corresponding to the GitHub file directory

        """
                
        # run some tests
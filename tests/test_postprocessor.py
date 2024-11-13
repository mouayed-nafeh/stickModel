import sys
import unittest
import os

# set paths
github_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       '..'))
sys.path.append(github_path)
os.chdir(os.path.join(github_path, 'src'))

# import the class to test
from postprocessor import *


class TestPostprocessor(unittest.TestCase):
        
    def test_get_fragility_function(self):
        
        # initialise the class
        pp = postprocessor()        

        # case #1: median check
        result = pp.get_fragility_function(0.50,  0.30, intensities = 0.50)        
        self.assertEqual(result, 0.50)
        
        # case #2: 16-th percentile check
        decimalPlace = 2
        result = pp.get_fragility_function(0.50, 0.30, 0.36976575137098666)
        self.assertAlmostEqual(result,0.16, decimalPlace)
        
        # case #3: 84-th percentile check
        decimalPlace = 2
        result = pp.get_fragility_function(0.50, 0.30, 0.6747982255787905)
        self.assertAlmostEqual(result,0.84, decimalPlace)
    
    def test_get_vulnerability_function(self):
        
        # initialise the class
        pp = postprocessor()
        
        # case #1: convolving a fragility function with a unit consequence models 
        # yields expected loss ratio values equal to the probabilities of exceedance
        poes = pp.get_fragility_function(0.50, 0.30).reshape((50, 1))
        consequence_model = [1.00]
        result = pp.get_vulnerability_function(poes, consequence_model)
        self.assertIsNone(np.testing.assert_array_equal(result, poes))
    
    
    def test_get_building_vulnerability(self):
        
        # initialise the class
        pp = postprocessor()
        
        # define case-independent variables
        structural_loss    = np.linspace(0.00,1.00,50)
        nonstructural_loss = np.linspace(0.00,1.00,50)
        
        # case #1: residential class
        occupancy_type = 'RES'
        result = pp.get_building_vulnerability(structural_loss, 
                                               nonstructural_loss, 
                                               occupancy_type)
        self.assertEqual(result[2], (3.00/8.00)*structural_loss[2]+(5.00/8.00)*nonstructural_loss[2])
        
        # case #2: commercial class
        occupancy_type = 'COM'
        result = pp.get_building_vulnerability(structural_loss, 
                                               nonstructural_loss, 
                                               occupancy_type)
        self.assertEqual(result[2], (2.00/5.00)*structural_loss[2]+(3.00/5.00)*nonstructural_loss[2])
        
        # case #3: industrial class
        occupancy_type = 'IND'
        result = pp.get_building_vulnerability(structural_loss, 
                                               nonstructural_loss, 
                                               occupancy_type)
        self.assertEqual(result[2], (3.00/8.00)*structural_loss[2]+(5.00/8.00)*nonstructural_loss[2])
        
    
    def test_get_total_building_vulnerability(self):
        
        # initialise the class
        pp = postprocessor()
        
        # define case-independent variables
        structural_loss    = np.linspace(0.00,1.00,50)
        nonstructural_loss = np.linspace(0.00,1.00,50)
        contents_loss      = np.linspace(0.00,1.00,50)        

        # case #1: residential class
        occupancy_type = 'RES'
        result = pp.get_total_vulnerability(structural_loss, 
                                             nonstructural_loss, 
                                             contents_loss,
                                             occupancy_type)
        self.assertEqual(result[2], 0.30*structural_loss[2]+0.50*nonstructural_loss[2]+0.20*contents_loss[2])
                
        # case #2: commercial class
        occupancy_type = 'COM'
        result = pp.get_total_vulnerability(structural_loss, 
                                             nonstructural_loss, 
                                             contents_loss,
                                             occupancy_type)
        self.assertEqual(result[2], 0.20*structural_loss[2]+0.30*nonstructural_loss[2]+0.50*contents_loss[2])

        # case #2: industrial class
        occupancy_type = 'IND'
        result = pp.get_total_vulnerability(structural_loss, 
                                             nonstructural_loss, 
                                             contents_loss,
                                             occupancy_type)
        self.assertEqual(result[2], 0.15*structural_loss[2]+0.25*nonstructural_loss[2]+0.60*contents_loss[2])
        
    
    def test_calculate_sigma_loss(self):
        
        # initialise the class
        pp = postprocessor()
        
        # case#1: if loss = 0.0, sigma = 0.0
        result = pp.calculate_sigma_loss(np.array([0.0]))
        self.assertEqual(result[0][0], 0.0)
        
        # case#2: if loss = 1.0, sigma = 1.0
        result = pp.calculate_sigma_loss(np.array([1.0]))
        self.assertEqual(result[0][0], 1.0)
        
        # case#3: if loss = 0.49, sigma = 0.37
        decimalPlace = 2
        result = pp.calculate_sigma_loss(np.array([0.49]))
        self.assertAlmostEqual(result[0][0], 0.37, decimalPlace)
        
    
if __name__ == '__main__':
    unittest.main()     
    

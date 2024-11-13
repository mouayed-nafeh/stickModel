import sys
import unittest
import os
import pandas as pd

# set paths
github_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       '..'))
sys.path.append(github_path)
os.chdir(os.path.join(github_path, 'src'))

# import the class to test
from calibration import *
from modeller import *
from units import *

class TestModeller(unittest.TestCase):
    
    def test_calibrate_model(self):
        
        # case: single-degree-of-freedom (CR_LFINF+DUL_H1_SOS)
        number_storeys = 1
        gamma = 1.33
        storey_height = 2.8
        sdof_period = 0.077
        isFrame = False
        isSOS = True
        sdof_capacity = np.array([[3.947368421052630406e-04,3.157894736842104758e-03,1.294736842105263103e-02,2.273684210526315599e-02],
                                  [2.679272838761374009e-01,5.358545677522748019e-01,3.215127406513648700e-01,3.247278680578785104e-01]]).T # omitting the initial zeros
        
        floor_masses, storey_disps, storey_forces, mdof_phi = calibrate_model(number_storeys, gamma, sdof_capacity, sdof_period, isFrame, isSOS)
        self.assertEqual(floor_masses[0], 1.00)
        self.assertIsNone(np.testing.assert_array_equal(storey_disps.flatten(),sdof_capacity[:,0]))
        self.assertIsNone(np.testing.assert_array_equal(storey_forces.flatten(),sdof_capacity[:,1]))
        self.assertEqual(mdof_phi, 1.00)

    
    def test_mdof_initialise(self):
        
        # case: single-degree-of-freedom (CR_LFINF+DUL_H1_SOS)
        number_storeys = 1
        gamma = 1.33
        storey_height = 2.8
        floor_heights = [storey_height]*number_storeys
        sdof_period = 0.077
        isFrame = False
        isSOS = True
        sdof_capacity = np.array([[3.947368421052630406e-04,3.157894736842104758e-03,1.294736842105263103e-02,2.273684210526315599e-02],
                                  [2.679272838761374009e-01,5.358545677522748019e-01,3.215127406513648700e-01,3.247278680578785104e-01]]).T # omitting the initial zeros
        
        floor_masses, storey_disps, storey_forces, mdof_phi = calibrate_model(number_storeys, gamma, sdof_capacity, sdof_period, isFrame, isSOS)
        model = modeller(number_storeys,floor_heights, floor_masses, storey_disps, storey_forces*units.g)
        model.mdof_initialise() 
        model.mdof_nodes()                                                          # Construct the nodes
        model.mdof_fixity()                                                         # Set the boundary conditions 
        model.mdof_material()                                                       # Assign the nonlinear storey material
        model.do_gravity_analysis()                                                 # Do gravity analysis
        T, _ = model.do_modal_analysis(num_modes = number_storeys)                  # Do modal analysis and get period of vibration
        
        self.assertEqual(T[0],sdof_period)
        self.assertEqual(model.number_storeys, 1)
        self.assertEqual(model.floor_heights,[2.8])
        self.assertEqual(model.floor_masses,[1.0])
        self.assertIsNone(np.testing.assert_array_equal(model.storey_disps.flatten(),sdof_capacity[:,0]))
        self.assertIsNone(np.testing.assert_array_equal(model.storey_forces.flatten(),sdof_capacity[:,1]*units.g))
     
        
if __name__ == '__main__':
    unittest.main()     


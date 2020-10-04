import unittest
import run
import numpy as np
import surfaces_and_fields.system_cylinder1D as system
import random
import math
import copy

class TestRun(unittest.TestCase):

  def setUp(self):
    pass

  def tearDown(self):
    # self.widget.dispose()
    pass
  
  def test_loop_wavenumber_kappa_dims(self):
    """

    """
    range1 = np.arange(0.005, .5, .3)
    range2 = np.arange(0, .2, .3)
    n_steps = 1
    converge_stop = True

if __name__ == '__main__':
    unittest.main()

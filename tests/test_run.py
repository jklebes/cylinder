import unittest
import run
import numpy as np
import system
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
    amplitude_results = run.loop_wavenumber_kappa(wavenumber_range=range1, kappa_range=range2, n_steps=n_steps)
    self.assertEqual(amplitude_results[0].shape, (len(range1), len(range2)), "dimensions of amplitude output")

if __name__ == '__main__':
    unittest.main()

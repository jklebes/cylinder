import unittest
import run
import numpy as np
import surfaces_and_fields.system_cylinder1D as system
import metropolisengine
import random
import math
import copy

class TestMetropolis(unittest.TestCase):

  def setUp(self):
    self.me = metropolisengine.MetropolisEngine(method='sequential', num_field_coeffs=3, sampling_dist=random.gauss, sampling_widths=.5, temp=0)

  def tearDown(self):
    # self.widget.dispose()
    pass
  
if __name__ == '__main__':
  unittest.main()

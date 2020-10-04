import unittest
import run
import numpy as np
import surfaces_and_fields.system_cylinder2D as system
import metropolisengine
import random
import math
import copy

class TestBaseMetropolis(unittest.TestCase):

  def setUp(self): 
    self.field_coeffs=dict([(i, 0+0j) for i in range(-3,4)])
    #self.me = metropolisengine.MetropolisEngine(self.field_coeffs)
    self.sys_basic = system.System(radius=1, wavenumber=1, kappa=1, gamma=1, alpha=-1, u=1, C=1, n=1)

  def tearDown(self):
    pass
  
  
if __name__ == '__main__':
  unittest.main()

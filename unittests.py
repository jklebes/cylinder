import unittest
import run
import numpy as np

class TestRun(unittest.TestCase):

  def setUp(self):
    pass

  def tearDown(self):
    # self.widget.dispose()
    pass

  def test_loop_wavenumber_kappa_dims(self):
    """
    runs, returns two arrays of expected dimensions
    :return:
    """
    range1 = np.arange(0.005, .5, .3)
    range2 = np.arange(0, .2, .3)
    amp_steps = 3
    converge_stop = True
    fieldsteps_per_ampstep = 1
    converge_time, amplitude_results = run.loop_wavenumber_kappa(wavenumber_range=range1, kappa_range=range2,
                                                   amp_steps=amp_steps
                                                   , fieldsteps_per_ampstep=fieldsteps_per_ampstep,converge_stop=converge_stop,
                                                  )
    #self.assertEqual(converge_time.shape, (len(range1), len(range2)), "dimensions of convergetime output")
    self.assertEqual(amplitude_results.shape, (len(range1), len(range2)), "dimensions of amplitude output")

  def test_decide_change1(self):
    old_energy = 12.45
    new_energy = -1451.0
    temp= .001
    self.assertTrue(run.decide_change(temp, old_energy, new_energy), "note: very rarely True by chance")

  def test_decide_change2(self):
    old_energy = -145.2
    new_energy = -145.3
    temp = 0
    self.assertTrue(run.decide_change(temp, old_energy, new_energy))

  def test_decide_change3(self):
    old_energy = 0
    new_energy = 0
    temp=0
    self.assertTrue(run.decide_change(temp, old_energy, new_energy), "equal energies defined as accepted")

  def test_decide_change4(self):
    old_energy = 14.3
    new_energy = 14.3
    temp=1.5
    self.assertTrue(run.decide_change(temp, old_energy, new_energy), "equal energies defined as accepted")

class TestCalcEnergy(unittest.TestCase):

  def setUp(self):
    #self.widget = Widget('The widget')
    pass
  
  def tearDown(self):
    #self.widget.dispose()
    pass

  def test_sum(self):
    self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()

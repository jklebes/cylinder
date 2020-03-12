import unittest
import run
import numpy as np
import calc_energy as ce
import random
import math
import copy

class TestRun(unittest.TestCase):

  def setUp(self):
    pass

  def tearDown(self):
    # self.widget.dispose()
    pass
  """
  def test_loop_wavenumber_kappa_dims(self):
    range1 = np.arange(0.005, .5, .3)
    range2 = np.arange(0, .2, .3)
    amp_steps = 1
    converge_stop = True
    fieldsteps_per_ampstep = 1
    converge_time, amplitude_results = run.loop_wavenumber_kappa(wavenumber_range=range1, kappa_range=range2,
                                                   amp_steps=amp_steps
                                                   , fieldsteps_per_ampstep=fieldsteps_per_ampstep,converge_stop=converge_stop,
                                                  )
    #self.assertEqual(converge_time.shape, (len(range1), len(range2)), "dimensions of convergetime output")
    self.assertEqual(amplitude_results.shape, (len(range1), len(range2)), "dimensions of amplitude output")
  """
  def test_decide_change1(self):
    old_energy = 12.45
    new_energy = -1451.0
    temp= .001
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy), "note: very rarely True by chance")

  def test_decide_change2(self):
    old_energy = -145.4
    new_energy = -145.3
    temp = 0
    self.assertFalse(run.metropolis_decision(temp, old_energy, new_energy))

  def test_decide_change2b(self):
    old_energy = -145.4
    new_energy = -145.5
    temp = 0
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy))

  def test_decide_change3(self):
    old_energy = 0
    new_energy = 0
    temp= 0
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy), "equal energies defined as accepted")

  def test_decide_change4(self):
    old_energy = 14.3
    new_energy = 14.3
    temp=1.5
    self.assertTrue(run.metropolis_decision(temp, old_energy, new_energy), "equal energies defined as accepted")

class TestCalcEnergy(unittest.TestCase):

  def setUp(self):
    #self.system_energy = ce.System_Energy()
    pass
  def tearDown(self):
    #self.widget.dispose()
    pass

  def test_sqrt_g_theta_flat(self):
    radius=1
    amplitude=0
    wavenumber=1
    system_energy = ce.System_Energy()
    for z in np.arange(0, 2 * math.pi / wavenumber, .3):
      flat = system_energy.sqrt_g_theta(radius, amplitude, wavenumber, z)
      self.assertEqual(flat, 1)
      small_radius = system_energy.sqrt_g_theta(.5, amplitude, wavenumber, z)
      self.assertLess(small_radius, 1)

  def test_sqrt_g_z(self):
    radius = 1
    amplitude = 0
    wavenumber = 1
    system_energy = ce.System_Energy()
    for z in np.arange(0, 2 * math.pi / wavenumber, .3):
      flat = system_energy.sqrt_g_z(radius, amplitude, wavenumber, z)
      self.assertEqual(flat, 1)
      perturbed = system_energy.sqrt_g_z(radius, .3, wavenumber, z)
      self.assertGreater(perturbed, 1)
      perturbed2 = system_energy.sqrt_g_z(radius, -.5, wavenumber, z)
      self.assertGreater(perturbed2, 1)

  def test_n_A_theta(self):
    system_energy = ce.System_Energy()
    radius=1
    amplitude=0
    wavenumber=1
    n=1
    for z in np.arange(0, 2 * math.pi / wavenumber, .3):
      flat = system_energy.n_A_theta_squared(radius, amplitude, wavenumber, z, n=n)
      self.assertEqual(flat, 0)
      perturbed = system_energy.n_A_theta_squared(radius, .12, wavenumber, z, n=n)
      self.assertGreater(perturbed, 0)
      perturbed2 = system_energy.n_A_theta_squared(radius, -.3, wavenumber, z, n=n)
      self.assertGreater(perturbed2, 0)
      perturbed_n6 = system_energy.n_A_theta_squared(radius, -.3, wavenumber, z, n=6)
      self.assertGreater(perturbed_n6, perturbed2)

  def test_radius_rescale_factor(self):
    system_energy = ce.System_Energy()
    flat = system_energy.radius_rescaled(amplitude=0,radius=1)
    self.assertEqual(flat, 1)
    perturbed = system_energy.radius_rescaled(amplitude=-.2,radius=1)
    self.assertLess(perturbed, 1)
    perturbed2 = system_energy.radius_rescaled(amplitude=0.9,radius=1)
    self.assertLess(perturbed2, 1)

  def test_A_integrand_psi0(self):
    diff = 0
    radius = 1
    amplitude = 0
    wavenumber = 1
    system_energy = ce.System_Energy()
    for z in np.arange(0, 2 * math.pi / wavenumber, .2):
      img = system_energy.A_integrand_img_part( diff, amplitude, z, wavenumber, radius)
      real = system_energy.A_integrand_real_part(diff, amplitude, z, wavenumber, radius)
      self.assertEqual(img, 0)
      self.assertEqual(real, 1) # surface area per unit area? without factor of 2pi

  def test_evaluate_A_integrals_zerofield(self):
    system_energy = ce.System_Energy()
    num_field_coeffs=1
    amplitude= 0
    radius=1
    wavenumber=1
    field_coeffs=dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    self.assertDictEqual(system_energy.A_integrals, dict([])) #an empty dict before using anything
    system_energy.evaluate_A_integrals(amplitude, wavenumber, field_coeffs, radius)
    self.assertNotEqual(system_energy.A_integrals, dict([]))
    self.assertEqual(system_energy.A_integrals[0], complex(2*math.pi, 0))
    self.assertAlmostEqual(system_energy.A_integrals[1], complex(0, 0))
    self.assertAlmostEqual(system_energy.A_integrals[-1], complex(0, 0))
    #evaluate at new ammpitude
    system_energy.evaluate_A_integrals(0.3, wavenumber, field_coeffs, radius)
    #pertrubed  - should change value to different surface area
    self.assertNotEqual(system_energy.A_integrals[0].real, complex(2 * math.pi, 0).real)
    #low wavenumber - greater surface area
    system_energy.evaluate_A_integrals(0.3, 0.5, field_coeffs, radius)
    self.assertGreater(system_energy.A_integrals[0].real, complex(2 * math.pi, 0).real)
    self.assertAlmostEqual(system_energy.A_integrals[0].imag, complex(2 * math.pi, 0).imag)
    #high wavenumber - less surface area
    system_energy.evaluate_A_integrals(0.3, 1.5, field_coeffs, radius)
    self.assertLess(system_energy.A_integrals[0].real, complex(2 * math.pi, 0).real)
    self.assertAlmostEqual(system_energy.A_integrals[0].imag, complex(2 * math.pi, 0).imag)

  def test_B_integrand_psi0(self):
    # TODO : debug B integrands
    i,j = 0,0
    radius = 1
    amplitude = 0
    wavenumber = 1
    n=1
    system_energy = ce.System_Energy()
    for z in np.arange(0, 2 * math.pi / wavenumber, .2):
      img = system_energy.B_integrand_img_part( i,j, amplitude, z, wavenumber, radius)
      real = system_energy.B_integrand_real_part(i,j, amplitude, z, wavenumber, radius, n)
      self.assertEqual(img, 0)
      self.assertEqual(real, 0) # surface area per unit area? without factor of 2pi

  def test_evaluate_B_integrals_zerofield(self):
    system_energy = ce.System_Energy()
    num_field_coeffs = 1
    amplitude = 0
    radius = 1
    wavenumber = 1
    n=1
    field_coeffs = dict([(i, 0 + 0j) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    self.assertDictEqual(system_energy.B_integrals, dict([]))  # an empty dict before using anything
    system_energy.evaluate_B_integrals(amplitude, wavenumber, field_coeffs, radius, n=n)
    self.assertNotEqual(system_energy.B_integrals, dict([])) #not an empty dict
    #expected values at a=0, c_i=0
    self.assertEqual(system_energy.B_integrals[(0,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(1,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(-1,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(1, 1)], complex(radius*2*math.pi/wavenumber, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(-1, -1)], complex(radius*2*math.pi/wavenumber, 0))
    # evaluate at new ammpitude
    new_amplitude=.243
    system_energy.evaluate_B_integrals(new_amplitude, wavenumber, field_coeffs, radius, n=n)
    # pertrubed  - should change value to greater
    self.assertNotEqual(system_energy.B_integrals[(0,0)].real, complex(0, 0).real)

  def test_evaluate_B_integrals_field_zeroamplitude(self):
    system_energy = ce.System_Energy()
    num_field_coeffs = 1
    amplitude = 0
    radius = 1
    wavenumber = 1
    n=1
    field_coeffs = dict([(i, complex(random.uniform(-1,1), random.uniform(-1,1))) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    self.assertDictEqual(system_energy.B_integrals, dict([]))  # an empty dict before using anything
    system_energy.evaluate_B_integrals(amplitude, wavenumber, field_coeffs, radius, n=n)
    self.assertNotEqual(system_energy.B_integrals, dict([])) #not an empty dict
    #expected values at a=0, c_i=0
    self.assertEqual(system_energy.B_integrals[(0,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(1,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(-1,0)], complex(0, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(1, 1)], complex(radius*2*math.pi/wavenumber, 0))
    self.assertAlmostEqual(system_energy.B_integrals[(-1, -1)], complex(radius*2*math.pi/wavenumber, 0))
    # evaluate at new ammpitude
    new_amplitude=.243
    system_energy.evaluate_B_integrals(new_amplitude, wavenumber, field_coeffs, radius, n=n)
    # pertrubed  - should change value to greater
    self.assertNotEqual(system_energy.B_integrals[(0,0)].real, complex(0, 0).real)

  def test_evaluate_B_integrals_field_zerofield_compare(self):
    system_energy = ce.System_Energy()
    num_field_coeffs = 3
    amplitude = .03
    radius = 1
    wavenumber = 1
    n=1
    field_coeffs_zero = dict([(i, complex(0,0)) for i in
                         range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    field_coeffs_random = dict([(i, complex(random.uniform(-1,1), random.uniform(-1,1))) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    self.assertDictEqual(system_energy.B_integrals, dict([]))  # an empty dict before using anything
    system_energy.evaluate_B_integrals(amplitude, wavenumber, field_coeffs_zero, radius, n=n)
    b_integrals_zerofield = copy.copy(system_energy.B_integrals)
    system_energy.evaluate_B_integrals(amplitude, wavenumber, field_coeffs_random, radius, n=n)
    b_integrals_randomfield = copy.copy(system_energy.B_integrals)
    self.assertGreaterEqual(b_integrals_randomfield[(0,0)].real, b_integrals_zerofield[(0,0)].real)
    self.assertGreaterEqual(b_integrals_randomfield[(1, 0)].real, b_integrals_zerofield[(1, 0)].real)
    self.assertGreaterEqual(b_integrals_randomfield[(-1,-1)].real, b_integrals_zerofield[(-1, -1)].real)
    self.assertGreaterEqual(b_integrals_randomfield[(1, 3)].real, b_integrals_zerofield[(1, 3)].real)
    self.assertGreaterEqual(b_integrals_randomfield[(1, 2)].real, b_integrals_zerofield[(1, 2)].real)
    self.assertGreaterEqual(b_integrals_randomfield[(2, 2)].real, b_integrals_zerofield[(2, 2)].real)

  def test_evaluate_B_integrals_problem(self):
    system_energy = ce.System_Energy()
    amplitude = 0
    radius = 1
    wavenumber = 1
    n=1
    field_coeffs_3 = {-3: (0 +0j), -2: (0+0j),
                    -1: (0+0j), 0: (0+0j),
                    1: (0+0j), 2: (0+0j),
                    3: (0+0j)}
    self.assertDictEqual(system_energy.B_integrals, dict([]))  # an empty dict before using anything
    system_energy.evaluate_B_integrals(amplitude, wavenumber, field_coeffs_3, radius, n=n)
    b_integrals_flat = copy.copy(system_energy.B_integrals)
    amplitude=0.1
    system_energy.evaluate_B_integrals(amplitude, wavenumber, field_coeffs_3, radius, n=n)
    b_integrals_perturbed = copy.copy(system_energy.B_integrals)
    self.assertGreaterEqual(b_integrals_perturbed[(0,0)].real, b_integrals_flat[(0,0)].real)
    self.assertGreaterEqual(b_integrals_perturbed[(1, 1)].real, b_integrals_flat[(1, 1)].real)
    self.assertGreaterEqual(b_integrals_perturbed[(-1,-1)].real, b_integrals_flat[(-1, -1)].real)
    # tODO: this fails
    self.assertGreaterEqual(b_integrals_perturbed[(2, 2)].real, b_integrals_flat[(2, 2)].real)
    self.assertGreaterEqual(b_integrals_perturbed[(3, 3)].real, b_integrals_flat[(3, 3)].real)
    self.assertGreaterEqual(b_integrals_perturbed[(-3, -3)].real, b_integrals_flat[(-3, -3)].real)

  def test_evaluate_B_integrals_problem_22(self):
    system_energy = ce.System_Energy()
    amplitude = 0
    radius = 1
    wavenumber = 1
    n=1
    field_coeffs_3 = { -2: (0+0j),
                    -1: (0+0j), 0: (0+0j),
                    1: (0+0j), 2: (0+0j)}
    self.assertDictEqual(system_energy.B_integrals, dict([]))  # an empty dict before using anything
    system_energy.evaluate_B_integrals(amplitude, wavenumber, field_coeffs_3, radius, n=n)
    b_integrals_flat = copy.copy(system_energy.B_integrals)
    self.assertAlmostEqual(b_integrals_flat[(2, 2)].real, 2*2*math.pi*2/wavenumber) #expected zero amplitude value ijSA
    amplitude = 0.1
    system_energy.evaluate_B_integrals(amplitude, wavenumber, field_coeffs_3, radius, n=n)
    b_integrals_perturbed = copy.copy(system_energy.B_integrals)
    # tODO: this fails
    self.assertGreaterEqual(b_integrals_perturbed[(2, 2)].real, b_integrals_flat[(2, 2)].real)

  def test_calc_field_energy_zerofield(self):
    num_field_coeffs = 3
    field_coeffs= dict([(i, 0+0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
    amplitude = 0
    wavenumber=1
    n=1
    radius = 1
    alpha= -1
    C=1
    u=1
    system_energy = ce.System_Energy()
    energy = system_energy.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_zerofield2(self):
    num_field_coeffs = 3
    field_coeffs= dict([(i, 0+0j) for i in range(-1*num_field_coeffs, num_field_coeffs+1)])
    amplitude = .3
    wavenumber=1
    n=1
    radius = 1
    alpha= -1
    C=1
    u=1
    system_energy = ce.System_Energy()
    energy = system_energy.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_1(self):
    num_field_coeffs = 3
    field_coeffs = dict([(i, complex(random.uniform(-1,1), random.uniform(-1,1))) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    amplitude = 0
    wavenumber = 1
    n = 1
    radius = 1
    alpha = 0
    C = 0
    u = 0
    system_energy = ce.System_Energy()
    energy = system_energy.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    self.assertEqual(energy, 0)

  def test_calc_field_energy_2(self):
    num_field_coeffs = 3
    field_coeffs = dict([(i, complex(random.uniform(-1,1), random.uniform(-1,1))) for i in range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    print(field_coeffs)
    wavenumber = .99
    n = 1
    radius = 1
    alpha = -1
    C = 1
    u = 1
    system_energy = ce.System_Energy()
    energy_flat = system_energy.calc_field_energy(field_coeffs, 0, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    energy_perturbed = system_energy.calc_field_energy(field_coeffs, .1, wavenumber, radius, n, alpha, C, u,
                                       amplitude_change=True)
    # TODO : this sometimes fails
    self.assertGreater(energy_perturbed, energy_flat, "less field energy on perturbation, C=1 case")

    #try C=0
    energy_flat_0C = system_energy.calc_field_energy(field_coeffs, 0, wavenumber, radius, n, alpha, 0, u,
                                                  amplitude_change=True)
    energy_perturbed_0C = system_energy.calc_field_energy(field_coeffs, .1, wavenumber, radius, n, alpha, 0, u,
                                                       amplitude_change=True)
    # TODO : this also occasioanlly fails
    self.assertGreater(energy_perturbed_0C, energy_flat_0C, "less field energy on perturbation, C=0 case")
    # succeeds -> problem comes from B integrals

  def test_calc_field_energy_problem(self):
    #perturbation lowers field energy
    field_coeffs ={-3: (0.9465116508232707 - 0.5387120173258799j), -2: (-0.718084567517205 + 0.5901035013776461j),
    -1: (0.07036136682408078 - 0.4089994218359341j), 0: (-0.5299508553843084 + 0.07924608399296607j),
    1: (-0.9320012671910849 - 0.8412360311955076j), 2: (0.9430104049475898 + 0.10038282895878625j),
    3: (-0.019608152141024737 + 0.5573601919676952j)} #coeffs that reproduce the problem
    wavenumber = .99
    n = 1
    radius = 1
    alpha = -1
    C = 1
    u = 1
    system_energy = ce.System_Energy()
    energy_flat = system_energy.calc_field_energy(field_coeffs, 0, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    energy_perturbed = system_energy.calc_field_energy(field_coeffs, .1, wavenumber, radius, n, alpha, C, u,
                                       amplitude_change=True)
    # TODO : this fails
    self.assertGreater(energy_perturbed, energy_flat, "less field energy on perturbation, C=1 case")
    # check surface area increased
    self.AssertGreater(system_energy.A_integrals[0].real, radius*2*math.pi/wavenumber, "surface area on perturbation")
    #try C=0
    energy_flat_0C = system_energy.calc_field_energy(field_coeffs, 0, wavenumber, radius, n, alpha, 0, u,
                                                  amplitude_change=True)
    energy_perturbed_0C = system_energy.calc_field_energy(field_coeffs, .1, wavenumber, radius, n, alpha, 0, u,
                                                       amplitude_change=True)
    self.assertGreater(energy_perturbed_0C, energy_flat_0C, "less field energy on perturbation, C=0 case")

  def test_calc_field_energy_problem_psi0(self):
    #perturbation lowers field energy
    field_coeffs ={-3: (0.9465116508232707 - 0.5387120173258799j), -2: (-0.718084567517205 + 0.5901035013776461j),
    -1: (0.07036136682408078 - 0.4089994218359341j), 0: (-0.5299508553843084 + 0.07924608399296607j),
    1: (-0.9320012671910849 - 0.8412360311955076j), 2: (0.9430104049475898 + 0.10038282895878625j),
    3: (-0.019608152141024737 + 0.5573601919676952j)} #coeffs that reproduce the problem
    field_coeffs_0 = dict([(0, field_coeffs[0])])
    wavenumber = 0.99
    n = 1
    radius = 1
    alpha = -1
    C = 1
    u = 1
    system_energy = ce.System_Energy()
    energy_flat = system_energy.calc_field_energy(field_coeffs_0, 0, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    energy_perturbed = system_energy.calc_field_energy(field_coeffs_0, .1, wavenumber, radius, n, alpha, C, u,
                                       amplitude_change=True)
    self.assertGreater(energy_perturbed, energy_flat, "less field energy on perturbation, C=1 case")

  def test_calc_field_energy_problem_psi2(self):
    #example reproding the problem
    field_coeffs ={-2: (-1+ 0j),
    -1: (0 - 1j), 0: (0+ 0j),
    1: (0+0j), 2: (0+ 0j)}
    wavenumber = .99
    n = 1
    radius = 1
    alpha = -1
    C = 1
    u = 1
    system_energy = ce.System_Energy()
    energy_flat = system_energy.calc_field_energy(field_coeffs, 0, wavenumber, radius, n, alpha, C, u, amplitude_change=True)
    energy_perturbed = system_energy.calc_field_energy(field_coeffs, .1, wavenumber, radius, n, alpha, C, u,
                                       amplitude_change=True)
    print(system_energy.A_integrals)
    print(system_energy.B_integrals)
    print(field_coeffs[-1] * field_coeffs[-1].conjugate() * system_energy.A_integrals[-1 +1])
    print(field_coeffs[-2] * field_coeffs[-1].conjugate() * system_energy.A_integrals[-2 + 1])
    print(field_coeffs[-1]*field_coeffs[-2].conjugate()*system_energy.A_integrals[-1+2])
    print(system_energy.A_integrals[-2 + 1],system_energy.A_integrals[-1+2])
    print(field_coeffs[-2] * field_coeffs[-2].conjugate() * system_energy.A_integrals[-2 + 2])
    self.assertGreater(energy_perturbed, energy_flat, "less field energy on perturbation")

  def test_calc_field_energy_problem_psi3(self):
    # perturbation lowers field energy
    field_coeffs_3 = {-3: (0.9 - 0.5j), -2: (0+0j),
                    -1: (0+0j), 0: (0+0j),
                    1: (0+0j), 2: (0+0j),
                    3: (0+0j)}
    wavenumber = 0.9999
    n = 1
    radius = 1
    alpha = -1
    C = 1
    u = 1
    system_energy = ce.System_Energy()
    energy_flat = system_energy.calc_field_energy(field_coeffs_3, 0, wavenumber, radius, n, alpha, C, u,
                                                  amplitude_change=True)
    energy_perturbed = system_energy.calc_field_energy(field_coeffs_3, .3, wavenumber, radius, n, alpha, C, u,
                                                       amplitude_change=True)
    # TODO : this fails
    self.assertGreater(energy_perturbed, energy_flat, "less field energy on perturbation, C=1 case")

  def test_calc_field_energy_problem_zerofield(self):
    # perturbation lowers field energy
    field_coeffs_zero = {-3: (0+ 0j), -2: (0+0j),
                    -1: (0+0j), 0: (0+0j),
                    1: (0+0j), 2: (0+0j),
                    3: (0+0j)}
    wavenumber = 0.9999
    n = 1
    radius = 1
    alpha = -1
    C = 1
    u = 1
    system_energy = ce.System_Energy()
    energy_flat = system_energy.calc_field_energy(field_coeffs_zero, 0, wavenumber, radius, n, alpha, C, u,
                                                  amplitude_change=True)
    energy_perturbed = system_energy.calc_field_energy(field_coeffs_zero, .3, wavenumber, radius, n, alpha, C, u,
                                                       amplitude_change=True)
    # doesnt fail -both 0 energy
    self.assertGreaterEqual(energy_perturbed, energy_flat, "less field energy on perturbation, C=1 case")

  def test_calc_field_energy_3(self):
    """
    should be more energy cost when there is a nonzero field bending rigidity
    :return:
    """
    num_field_coeffs = 3
    field_coeffs = dict([(i, complex(random.uniform(-1, 1), random.uniform(-1, 1))) for i in
                           range(-1 * num_field_coeffs, num_field_coeffs + 1)])
    wavenumber = 1
    n = 1
    radius = 1
    alpha = -1
    amplitude = 0.1
    u = 1
    system_energy = ce.System_Energy()
    energy_0C = system_energy.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, 0, u, amplitude_change=True)
    energy_C = system_energy.calc_field_energy(field_coeffs, amplitude, wavenumber, radius, n, alpha, 1, u,
                                              amplitude_change=True)
    self.assertGreater(energy_C, energy_0C)

if __name__ == '__main__':
    unittest.main()

import On_lattice_simple as l
import math
import random

"""
trial for how ot find effective alpha, C, u as a function of lattice spacing
trying on unperturbed cylinders = plane
"""

def u(u0, dims):
    return u0#/dims[0]**2

def C(C0, dims):
    #a number: difference in sum over ln(|q|) for each lattice vector q
    #for C0 there is only one lattice vector q=0 and this is left out in both
    D=0
    #then ln|q| to D
    for i in range(-dims[0]//2, dims[0]//2+1):
        for j in range(-dims[1]//2, dims[1]//2+1):
            try:
                D+=math.log(math.sqrt(i**2+j**2))
            except ValueError:
                #skip ln(0)
                pass
    N=dims[0]*dims[1]
    return 2*math.pi*N*(C0/(2*math.pi))**(1/N)*math.exp(2*D/N)

def alpha(alpha0, dims):
    return 2*math.pi*dims[0]*dims[1]*(alpha0/(2*math.pi))**(1/float(dims[0]*dims[1]))

if __name__=="__main__":
    n_steps=3000
    alpha0=1
    C0=1
    u0=0

    spacings= [(20,20),(25,25), (10,10), (5,5)]#, (50,50)]
    energies={}
    energies2={}
    profiles={}
    """
    temp=.01
    for dims in spacings:
        alpha_eff = alpha(alpha0, dims)
        lattice=[0]*(dims[0]*dims[1])
        area = 4*math.pi**2/(dims[0]*dims[1])
        for i in range(n_steps):
            n_substeps=dims[0]*dims[1]
            for j in range(n_substeps):
                pos=random.randint(0, n_substeps-1)
                r=random.gauss(lattice[pos], .01)
                old_energy = lattice[pos]**2*alpha_eff
                new_energy = r**2*alpha_eff
                diff = old_energy-new_energy
                #metropolissteps
                if random.uniform(0,1) <= math.exp(diff/temp):
                    lattice[pos]=r
                    #
                    # print(new_energy)
        energy = 0
        for x in lattice:
            energy += x**2*alpha_eff*area
        print(alpha_eff, lattice[0], energy)

    """
    #for each lattice spacing, make a lattice object
    for dims in spacings:
        lattice=l.Lattice(amplitude=0, wavenumber=1, radius=1, gamma=0, kappa=0,
                          intrinsic_curvature=0, alpha=alpha(alpha0, dims), u=u(u0, dims), 
                          C=C(C0, dims), n=1, temperature=1000, temperature_lattice=1000, dims=dims)
        #trial MC simulation for each
        #run without updating amplitude away from 0
        for i in range(n_steps):
            lattice.measure_avgs()
            print(i, lattice.surface_field_energy(amplitude=0))
            for i in range(lattice.n_substeps):
                lattice.step_lattice(amplitude=0)
        lattice.plot_save('.',"renormalizetrial_"+str(dims[0]))
        energies[dims]=lattice.surface_field_energy(amplitude=0)
        profiles[dims]=lattice.field_average
        #compare energies
        #do the energies myself for alpha0=1, C0=0, uo=0
        energy_snapshot = 0+0j
        alpha_eff=alpha(alpha0,dims)

    print(energies)
    #print(profiles)
    for dims in spacings:
      print("alpha'", alpha(alpha0, dims))
      print("C'", C(C0, dims))
    
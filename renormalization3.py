import random
import math
import surfaces_and_fields.On_lattice_simple as lattice

"""
experimentally get mean, variance of large blocks Nx x Ny
each composed of the same size pixels 2pi lx0, 2pi ly0 with microscopic alpha0, C0

"""
def squared(c):
    return c*c.conjugate()

def get_mean_variance(Nx, Ny):
    """just run a simulation"""
    n_substeps=Nx*Ny

    #2.simulate a 3Nx3N fine grained lattice with bare alpha, C, F_0
    #collect the averaages and statisitcs of 9 sub regions
    l_2 = lattice.Lattice(amplitude=0, alpha=alpha0, C=C0,
                        wavenumber=1, radius=3*Nx*lx0, gamma=3*Ny*ly0, kappa=0,
                        n=1, dims=(3*Nx,3*Ny), temperature_lattice=t, intrinsic_curvature=0, 
                        u=0, temperature=1)

    #set values in most squares
    for i in range(Nx):
        for j in range(Ny): 
            l_2.lattice[i,j] = 0+0j
            l_2.lattice[Nx+i, j] = leftBC
            l_2.lattice[2*Nx+i,j] = 0+0j

            l_2.lattice[i,Ny+j] = upBC
            l_2.lattice[2*Nx+i,Ny+j] = downBC

            l_2.lattice[i,2*Ny+j] = 0+0j
            l_2.lattice[Nx+i, 2*Ny+j] = rightBC
            l_2.lattice[2*Nx+i,2*Ny+j] = 0+0j

    
    #force reevaluation of gradients etc
    for i in range(Nx*3):
        for j in range(Ny*3):
            l_2.psi_squared[i, j] = l_2.squared(l_2.lattice[i,j]) 
    #fill derivatives
    for z_index in range(l_2.z_len):
      for th_index in range(l_2.th_len):    
        #dz
        value= l_2.lattice[z_index, th_index]
        left_value_z = l_2.lattice[z_index-1, th_index]
        #just a left (backwards) derivative
        l_2.dz[z_index, th_index]  = value-left_value_z
        #dth
        left_value_th = l_2.lattice[z_index, th_index-1]
        l_2.dth[z_index, th_index]  =   value-left_value_th
    l_2.dz/= l_2.z_pixel_len
    l_2.dth/= l_2.th_pixel_len

    #metropolis step everything in the central square
    psi_history=[]
    for i in range(n_steps):
        for j in range(n_substeps):
          z_index = Nx+random.randrange(0, Nx)
          th_index = Ny+random.randrange(0, Ny)
          l_2.step_lattice_loc(amplitude=0, index_th=th_index, index_z=z_index)
        #record an avg of central NxN pixels
        avg = sum(sum(l_2.lattice[Nx:2*Nx, Ny:2*Ny]))/float(Nx*Ny)
        psi_history.append(avg)
        #print(i, avg)
        #print(l_2.lattice[Nx:2*Nx, Ny:2*Ny])
        #energy_2_history.append(energy_avg)#??
    #collect its averages and statistics
    #print(i, avg, l_2.lattice[12, 14])
    
    
    #compare

    print(leftBC, rightBC, upBC, downBC, "mean BC", sum([leftBC, rightBC, upBC, downBC])/4.0)
    mean = sum(psi_history)/len(psi_history)
    var= sum(squared(i - mean) for i in psi_history) / len(psi_history) 
    print(Nx, "mean", mean)
    print(Nx, "variance ", var)
    return (mean, var)
    

if __name__ == "__main__":
    lx0=200
    ly0=200
    C0= 1
    alpha0=1
    n_steps=1000
    t=1

    leftBC = 0#random.uniform(-5,5)+random.uniform(-5,5)*1j
    rightBC = 0#random.uniform(-5,5)+random.uniform(-5,5)*1j
    upBC = 0#random.uniform(-5,5)+random.uniform(-5,5)*1j
    downBC = 0#random.uniform(-5,5)+random.uniform(-5,5)*1j
    
    means=[]
    variances=[]
    for N in [1,2,5,10,20,50,100]:
        mean, var = get_mean_variance(N, N)
        means.append(mean)
        variances.append(var)
    print(means)
    print(variances)
    
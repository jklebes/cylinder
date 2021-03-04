import random
import math
import surfaces_and_fields.On_lattice_simple as lattice

def alpha(alpha0, C0, lx, ly):
    return ((math.pi*2)**2-32)*lx*ly*alpha0 #7.48 lxly alpha0

def F(F0, Nx, Ny):
    pass

def squared(c):
    return c*c.conjugate()

def C(C0, alpha0, lx, ly):
    return 0#alpha0*8*math.pi *lx * math.pi *4*4*lx*ly + C0 *2*math.pi /lx

if __name__ == "__main__":
    Nx=10
    Ny=Nx
    C0= 0
    alpha0=10 
    n_steps=500
    n_substeps=Nx*Ny
    t=1
    leftBC = random.uniform(-5,5)+random.uniform(-5,5)*1j
    rightBC = random.uniform(-5,5)+random.uniform(-5,5)*1j
    upBC = random.uniform(-5,5)+random.uniform(-5,5)*1j
    downBC = random.uniform(-5,5)+random.uniform(-5,5)*1j

    #1.simulate a 3x3 periodic lattice with alpha', C', F_0' retrieved
    #from renomarlization attempt
    #fixed values in all but central block?
    l = lattice.Lattice(amplitude=0, alpha=alpha(alpha0=alpha0,C0=C0,lx=1, ly=1), 
                        C= C(C0=C0, alpha0=alpha0, lx=1, ly=1), 
                        wavenumber=1, radius=1, gamma=1, kappa=0,
                        n=1, dims=(3,3), temperature_lattice=t, intrinsic_curvature=0, u=0, temperature=1)

    #set values in most squares
    l.lattice[0,0] = 0+0j
    l.lattice[1,0] = leftBC
    l.lattice[2,0] = 0+0j
    l.lattice[0,1] = upBC
    l.lattice[2,1] = downBC
    l.lattice[0,2] = 0+0j
    l.lattice[1,2] = rightBC
    l.lattice[2,2] = 0+0j
    #force reevaluation of gradients etc
    for i in range(3):
        for j in range(3):
            l.psi_squared[i, j] = l.squared(l.lattice[i,j]) 
    #fill derivatives
    for z_index in range(l.z_len):
      for th_index in range(l.th_len):    
        #dz
        value= l.lattice[z_index, th_index]
        left_value_z = l.lattice[z_index-1, th_index]
        #just a left (backwards) derivative
        l.dz[z_index, th_index]  = value-left_value_z
        #dth
        left_value_th = l.lattice[z_index, th_index-1]
        l.dth[z_index, th_index]  =   value-left_value_th
    l.dz/= l.z_pixel_len
    l.dth/= l.th_pixel_len

    #metropolis step the central square
    psi_history=[]
    for i in range(n_steps):
        l.step_lattice_loc(amplitude=0, index_th=1, index_z=1)
        psi_history.append(l.lattice[1,1])
        print("big lattice", i, l.lattice[1,1])
    #collect its averages and statistics

    #2.simulate a 3Nx3N fine grained lattice with bare alpha, C, F_0
    #collect the averaages and statisitcs of 9 sub regions
    l_2 = lattice.Lattice(amplitude=0, alpha=alpha(alpha0=alpha0, C0=C0, lx=1.0/Nx, ly=1.0/Ny), 
                        C=C(alpha0=alpha0, C0=C0, lx=1.0/Nx, ly=1.0/Ny),
                        wavenumber=1, radius=1, gamma=1, kappa=0,
                        n=1, dims=(3*Nx,3*Ny), temperature_lattice=t, intrinsic_curvature=0, u=0, temperature=1)

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
    psi_2_history=[]
    for i in range(n_steps):
        for j in range(n_substeps):
          z_index = Nx+random.randrange(0, Nx)
          th_index = Ny+random.randrange(0, Ny)
          l_2.step_lattice_loc(amplitude=0, index_th=th_index, index_z=z_index)
        #record an avg of central NxN pixels
        avg = sum(sum(l_2.lattice[Nx:2*Nx, Ny:2*Ny]))/float(Nx*Ny)
        psi_2_history.append(avg)
        print(i, avg)
        #print(l_2.lattice[Nx:2*Nx, Ny:2*Ny])
        #energy_2_history.append(energy_avg)#??
    #collect its averages and statistics
    #print(i, avg, l_2.lattice[12, 14])
    
    
    #compare

    print(leftBC, rightBC, upBC, downBC, "mean BC", sum([leftBC, rightBC, upBC, downBC])/4.0)
    print("alpha'", alpha(alpha0=alpha0,C0=C0, lx=1, ly=1))
    print("C'",C(alpha0=alpha0,C0=C0, lx=1, ly=1))
    print("alpha'2", alpha(alpha0=alpha0,C0=C0, lx=1.0/Nx, ly=1.0/Ny))
    print("C'2",C(alpha0=alpha0,C0=C0, lx=1.0/Nx, ly=1.0/Ny))
    #print("1/(|alpha|+C)'", 1.0/(alpha(alpha0=alpha0,C0=C0, Nx=Nx, Ny=Ny)+C(alpha0=alpha0,C0=C0, Nx=Nx, Ny=Ny)))
    mean = sum(psi_history)/len(psi_history)
    var= sum((i - mean) ** 2 for i in psi_history) / len(psi_history) 
    print("mean big lattice", mean)
    print("variance ", var)
    mean = sum(psi_2_history)/len(psi_2_history)
    var= sum((i - mean) ** 2 for i in psi_2_history) / len(psi_2_history) 
    print("mean explicit simulation", mean)
    print("measured variance", var)
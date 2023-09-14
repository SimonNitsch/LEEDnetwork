import numpy as np
import LEED_generate_pictures_functions as leed
import pandas as pd
import numba
import os
import matplotlib.pyplot as plt

@numba.jit(nopython=True)
def gen_square():
    r = np.random.rand() * 2 + 6
    ar = np.array([r,r])
    aphi = np.array([0,np.pi/2])
    return ar, aphi

@numba.jit(nopython=True)
def gen_rect():
    ar = np.random.rand(2) * 2 + 6
    aphi = np.array([0,np.pi/2])
    return ar, aphi

@numba.jit(nopython=True)
def gen_hex():
    r = np.random.rand() * 2 + 6
    ar = np.array([r,r])
    aphi = np.array([0,np.pi*2/3])
    return ar, aphi

@numba.jit(nopython=True)
def gen_oblique():
    ar = np.random.rand(2) * 2 + 6
    aphi = np.random.rand(2) * np.pi
    aphi[np.abs(aphi-np.pi/2) <= np.pi/30] += np.pi/15
    aphi[np.abs(aphi-np.pi*2/3) <= np.pi/30] += np.pi/15
    return ar, aphi



def rand_numbers(number,max_atoms=4):
    r = np.random.randint(2,high=np.power(2,max_atoms)+1,size=number)
    rlog = np.log2(r)
    rfinal = np.array(np.ceil(rlog),dtype=int)
    return (max_atoms + 1 - rfinal)

@numba.jit(nopython=True)
def generate_a_b(R):
    dist = 3
    a1 = np.zeros(2)
    a2 = np.zeros(2)

    while dist > 1.6:
        gen = np.random.randint(low=0,high=4)
        if gen == 0:
            ar, aphi = gen_square()
        if gen == 1:
            ar, aphi = gen_rect()
        if gen == 2:
            ar, aphi = gen_hex()
        if gen == 3:
            ar, aphi = gen_oblique()

        a1[0] = ar[0] * np.cos(aphi[0])
        a1[1] = ar[0] * np.sin(aphi[0])
        a2[0] = ar[1] * np.cos(aphi[1])
        a2[1] = ar[1] * np.sin(aphi[1])
        b1, b2 = leed.calculate_b(a1, a2, R)
        b1_max = np.linalg.norm(b1,np.inf)
        b2_max = np.linalg.norm(b2,np.inf)
        dist = min(b1_max,b2_max)

    return a1, a2, b1, b2, gen




def Generate_Pictures(number,folder,types=98,max_atoms=4):
    
    gen_list = np.zeros((number,4))
    atom_types = leed.formfactors.index.values
    atom_types = np.append(atom_types,[" "])
    sizes = rand_numbers(number,max_atoms=max_atoms)

    zero_atoms = np.zeros((4,types+1))
    zero_atoms[:,-1] = 1
    atom_list = np.zeros((4,types+1))

    R = np.array([[0,-1],[1,0]])
    os.mkdir(folder)

    for i in range(number):
        ats = np.random.randint(0,high=types,size=4)
        atom_list_proto = np.zeros((4,types+1))

        atom_list_proto[0,ats[0]] = 1
        atom_list_proto[1,ats[1]] = 1
        atom_list_proto[2,ats[2]] = 1
        atom_list_proto[3,ats[3]] = 1

        atom_list = zero_atoms
        atom_list[0:sizes[i],:] = atom_list_proto[0:sizes[i],:]
        atoms = atom_types[ats][0:sizes[i]]

        coords = np.random.rand(sizes[i],2)
        a1, a2, b1, b2, gen = generate_a_b(R)
        gen_list[i,gen] = 1

        p = leed.generate_data_point(a1,a2,b1,b2,atoms,coords)

        plt.imsave(os.path.join(folder,"LEEDImage_%s%s.jpg" %((len(str(number))-len(str(i)))*"0" ,i)),p,cmap="gray")
    
    np.save(os.path.join(folder,"Type"),gen_list)




if __name__ == "__main__":
    Generate_Pictures(2000,"Test3",types=9,max_atoms=3)

        




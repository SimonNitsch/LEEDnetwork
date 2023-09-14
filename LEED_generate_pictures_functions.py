import numpy as np
import pandas as pd
import numba


formfactors = pd.read_excel("formfacs.xlsx",index_col=0,header=None,names=["a1","b1","a2","b2","a3","b3","a4","b4","a5","b5"]).astype(np.double)


@numba.jit(nopython=True)
def formfac(G,ab):
    f = 0
    GG = 0
    for i in G:
        GG += i*i
    for i in range(5):
        f += ab[2*i] * np.exp(-ab[2*i+1] * GG/(16*np.pi**2))
        
    return f


@numba.jit(nopython=True)
def pre_calc(b1,b2,h_much_b1,h_much_b2):
    border_list = np.zeros(h_much_b1 * 2 + 1)
    border_list[h_much_b1] = h_much_b2

    for i in range(1,h_much_b1+1):
        b1mult = i*b1
        dist = np.linalg.norm(b1mult,ord=np.inf)
        b2_nums = -1
        while dist < 5:
            b1mult += b2
            dist = np.linalg.norm(b1mult,ord=np.inf)
            b2_nums+=1
        border_list[h_much_b1+i] = b2_nums
        border_list[h_much_b1-i] = b2_nums
    return border_list


@numba.jit(nopython=True)
def acccalc(b1,b2,ab,pos,h_much_b1,h_much_b2,borderlist): 
    vals = np.zeros((h_much_b1*2+1,h_much_b2*2+1,2))
    for h in range(-h_much_b1,h_much_b1+1):
        for k in range(-borderlist[h+h_much_b1],borderlist[h+h_much_b1]+1):
            G = h*b1 + k*b2
            Gpos = 0
            for (i,j) in zip(G,pos):
                Gpos += i*j

            f = formfac(G,ab)
            res = f * np.exp(-1j* Gpos)
            vals[h+h_much_b1,k+h_much_b2,0] = np.real(res)
            vals[h+h_much_b1,k+h_much_b2,1] = np.imag(res)
    return vals


@numba.jit(nopython=True)
def point_source(I):
    pic_I = I * 255 / 1000 * np.sqrt(2)
    light = np.zeros((16,16))
    for i in range(1,9):
        for j in range(1,9):
            circle_norm = np.sqrt(i**2+j**2)
            if circle_norm <= 8:       
                current_I = min(pic_I*(9-circle_norm)/8,255)
                light[8-i,8-j] = current_I
                light[7+i,8-j] = current_I
                light[8-i,7+j] = current_I
                light[7+i,7+j] = current_I
            else:
                light[8-i,8-j] = 0
                light[7+i,8-j] = 0
                light[8-i,7+j] = 0
                light[7+i,7+j] = 0

    return light


@numba.jit(nopython=True)
def generate_picture(b1,b2,I):
    picture = np.zeros((1016,1016))
    h_m = int((I.shape[0]-1)/2)
    k_m = int((I.shape[1]-1)/2)

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            k_coords = (i-h_m) * b1 + (j-k_m) * b2
            k_coords[1] *= -1
            k_inds = np.flip(k_coords) *100 + 500
            k_x = int(k_inds[0])
            k_y = int(k_inds[1])
            picture[k_x:k_x+16,k_y:k_y+16] += point_source(I[i,j])

    return picture[8:1008,8:1008]


@numba.jit(nopython=True)
def calculate_b(a1, a2, R):
    c = 2 * np.pi / custom_vecmul(a1, custom_matmul(R,a2))
    b1 = custom_matmul(R,a2) * c
    b2 = custom_matmul(R,a1) * c
    return b1, b2

@numba.jit(nopython=True)
def custom_matmul(R,a):
    b = np.zeros(R.shape[0])
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            b[i] += R[i,j] * a[j]
    return b

@numba.jit(nopython=True)
def custom_vecmul(a,b):
    c = 0
    for i in range(a.shape[0]):
        c += a[i] * b[i]
    return c


            


def generate_data_point(a1,a2,b1,b2,atoms,coords):
    b1_max = np.linalg.norm(b1,ord=np.inf)
    b2_max = np.linalg.norm(b2,ord=np.inf)
    h_much_b1 = int(np.floor(5/b1_max))
    h_much_b2 = int(np.floor(5/b2_max))
    result = np.zeros((h_much_b1*2+1,h_much_b2*2+1,2))
    borderlist = pre_calc(b1,b2,h_much_b1,h_much_b2)

    for index in range(atoms.size):
        ab = formfactors.loc[atoms[index]].values
        pos = coords[index,0]*a1 + coords[index,1]*a2
        result += acccalc(b1,b2,ab,pos,h_much_b1,h_much_b2,borderlist)

    Intensity = np.abs(result[:,:,0])**2 + np.abs(result[:,:,1])**2
    p = generate_picture(b1,b2,Intensity)

    return p


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    atoms = np.array(["Cs","Cl"])
    coords = np.array([[0.0,0.0],[0.5,0.5]])
    a1 = np.array([4.12,0])
    a2 = np.array([2,4.12])
    R = np.array([[0,-1],[1,0]])
    b1, b2 = calculate_b(a1,a2,R)
    p = generate_data_point(a1,a2,b1,b2,atoms,coords)

    plt.figure(figsize=[10,10])
    plt.imshow(p,cmap="inferno")
    plt.xticks([])
    plt.yticks([])
    plt.show()




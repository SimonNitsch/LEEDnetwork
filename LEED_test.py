import os
try:
    os.add_dll_directory("C:\\Programme\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")
    print("Cuda importing finished")
except:
    print("Cuda importing failed")

import numpy as np
import tensorflow as tf
from tensorflow import keras as kr
import matplotlib.pyplot as plt

LEEDmodel = kr.models.load_model("LEEDmodel")


type = np.load(os.path.join("Test3","Type.npy"))
tcount = np.arange(type.shape[1])

dataset_length = type.shape[0]
dataset_digits = len(str(dataset_length))

vals = [np.array([])]*type.shape[1]

for i in range(dataset_length):
    picname = "LEEDImage_%s%s.jpg" %((dataset_digits-len(str(i)))*"0",i)
    rgb = np.array(plt.imread(os.path.join("Test3",picname)),dtype=np.float64)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    picture = 0.2989 * r + 0.5870 * g + 0.1140 * b
    p2 = np.expand_dims(picture,2)
    p = np.expand_dims(p2,0)

    result = LEEDmodel.predict(p)
    ind = int(type[i,:]@tcount)

    vals[ind] = np.append(vals[ind],np.array(result[0,ind]))
    
    if i%10==0:
        print(i)

os.mkdir("Test3res")
np.save(os.path.join("Test3res","type1"),vals[0])
np.save(os.path.join("Test3res","type2"),vals[1])
np.save(os.path.join("Test3res","type3"),vals[2])
np.save(os.path.join("Test3res","type4"),vals[3])


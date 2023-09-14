import numpy as np
import matplotlib.pyplot as plt
import os

v1 = np.load(os.path.join("Test3res","type1.npy"))
v2 = np.load(os.path.join("Test3res","type2.npy"))
v3 = np.load(os.path.join("Test3res","type3.npy"))
v4 = np.load(os.path.join("Test3res","type4.npy"))


fig = plt.figure()
fig.suptitle("Distribution of Results")

ax1 = fig.add_subplot(2,2,1)
ax1.hist(v1,bins=50)
ax1.set_xlim(left=0,right=1.005)
ymin, ymax = ax1.get_ylim()
ax1.plot([0.75,0.75],[ymin,ymax],"r--")
v1_rate = np.sum(v1>=0.75) / v1.shape[0]
ax1.set_title("Square Lattice, Det. Rate: %s%%" % int(v1_rate*100))

ax2 = fig.add_subplot(2,2,2)
ax2.hist(v2,bins=50)
ax2.set_xlim(left=0,right=1.005)
ymin, ymax = ax2.get_ylim()
ax2.plot([0.75,0.75],[ymin,ymax],"r--")
v2_rate = np.sum(v2>=0.75) / v2.shape[0]
ax2.set_title("Rectangle Lattice, Det. Rate: %s%%" % int(v2_rate*100))

ax3 = fig.add_subplot(2,2,3)
ax3.hist(v3,bins=3)
ax3.set_xlim(left=0,right=1.005)
ymin, ymax = ax3.get_ylim()
ax3.plot([0.75,0.75],[ymin,ymax],"r--")
v3_rate = np.sum(v3>=0.75) / v3.shape[0]
ax3.set_title("Hexagonal Lattice, Det. Rate: %s%%" % int(v3_rate*100))

ax4 = fig.add_subplot(2,2,4)
ax4.hist(v4,bins=50)
ax4.set_xlim(left=0,right=1.005)
ymin, ymax = ax4.get_ylim()
ax4.plot([0.75,0.75],[ymin,ymax],"r--")
v4_rate = np.sum(v4>=0.75) / v4.shape[0]
ax4.set_title("Oblique Lattice, Det. Rate: %s%%" % int(v4_rate*100))

plt.show()
    
    

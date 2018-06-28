import numpy as np
import glob

stuff = glob.glob('*.dat')
for thing in stuff:
    data = np.loadtxt(thing)
    new_col = data[:,0] * 2
    np.savetxt(thing[:-4]+".txt", np.column_stack((data[:,0], new_col, data[:,1])))


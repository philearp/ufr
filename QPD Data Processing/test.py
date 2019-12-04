import numpy as np
raw = np.loadtxt('real test 2_19-11-28_15-36-41 (5).csv', dtype=float , delimiter=',', skiprows=2)
t = raw[:,0]
x = raw[:,1]
y = raw[:,2]
intensity = raw[:,3]
print('done')
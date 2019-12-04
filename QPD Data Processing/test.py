import numpy as np
import matplotlib.pyplot as plt
raw = np.loadtxt('real test 2_19-11-28_15-36-41 (5).csv', dtype=float , delimiter=',', skiprows=2)
t = raw[:,0]
t = t - t[0]
x = raw[:,1]
y = raw[:,2]
intensity = raw[:,3]

plt.plot(t,x,'ko-')
plt.xlabel('time (microseconds)', fontsize=15)
plt.ylabel('QPD x-value (V)', fontsize=15)
plt.xlim(200000, 200500)
plt.show()
plt.savefig('QPD Raw Data ON.png')

# plt.plot(t,x,'ko-')
# plt.xlabel('time (microseconds)', fontsize=15)
# plt.ylabel('QPD x-value (V)', fontsize=15)
# plt.xlim(900000, 900500)
# plt.savefig('QPD Raw Data OFF.png')


print('done')
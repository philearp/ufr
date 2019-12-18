import numpy as np
import matplotlib.pyplot as plt
raw = np.loadtxt('real test 2_19-11-28_15-36-41 (5).csv', dtype=float , delimiter=',', skiprows=2)
t = raw[:,0] # time [microseconds]
t = t - t[0] # elapsed time [microseconds]
t = t * 1e-6 # elapsed time [seconds]
x = raw[:,1]
y = raw[:,2]
intensity = raw[:,3]

plt.plot(t, x, 'k-')
plt.xlabel('time (s)', fontsize=15)
plt.ylabel('QPD x-value (V)', fontsize=15)
# plt.savefig('QPD Raw Data ON.png')

plt.figure()
plt.plot(t*1000, x, 'ko-')
plt.xlabel('time (ms)', fontsize=15)
plt.ylabel('QPD x-value (V)', fontsize=15)
plt.xlim(200, 200.5)
# plt.savefig('QPD Raw Data ON.png')

f_osc = 20e3 # oscillation frequency [Hz]
T_osc = 1 / f_osc # oscillation time period [s]

plt.figure()
plt.plot(t / T_osc, x, 'ko-')
plt.xlabel('cycles (n)', fontsize=15)
plt.ylabel('QPD x-value (V)', fontsize=15)
plt.xlim(0.2 / T_osc, 0.2005 / T_osc)
plt.show()



print('done')
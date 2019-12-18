import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.fftpack

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

f_samp = 100e3 # sampling frequency [Hz]
T_samp = 1 / f_samp # sampling time period [s]

# calculate fft of QPD x signal
x_fft = sp.fftpack.fft(x) # complex
# calculate power spectral density (square of abs. value)
x_psd = np.abs(x_fft) ** 2 # power spectral density
# calculate frequency values of the PSD
fftfreq = sp.fftpack.fftfreq(len(x_psd), T_samp)
# only interested in positive frequencies
i = fftfreq > 0

plt.figure()
plt.plot(fftfreq[i], 10 * np.log10(x_psd[i])) # logarithmic
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB)')

plt.figure()
plt.plot(fftfreq[i], x_psd[i])
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (arb. units)')


plt.show()
print('done')
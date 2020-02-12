import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def load_data(filepath, skiprows):
    raw = np.loadtxt(filepath, dtype=float , delimiter=',', skiprows=skiprows)

    t = raw[:,0] # time [microseconds]
    t = t - t[0] # elapsed time [microseconds]
    t = t * 1e-6 # elapsed time [seconds]
    x_raw = raw[:,1]
    y_raw = raw[:,2]
    intensity = raw[:,3]

    # normalise qpd data
    x = np.divide(x_raw, intensity)
    y = np.divide(y_raw, intensity)

    return x, y, t

# ---------------------- Code Begins Here ------------------   

## Inputs
f_osc = 20e3 # approx. oscillation frequency [Hz]
sampling_frequency = 200e3 # sampling frequency [Hz]

variable_to_calculate = 'x'

#filepath = 'data\Test with new optics_20-02-05_15-59-35 (2).csv'
filepath = 'T:/Steve Berks/First observed failure/20sec  camera on qpd static qpd analysis 0.1 big loop 38PerCent_20-02-10_15-28-03 (5).csv'
skiprows = 4
##

x, y, t = load_data(filepath, skiprows)

plt.plot(t, x, 'k-')
plt.xlabel('time (s)', fontsize=15)
plt.ylabel('QPD x-position (normalised)', fontsize=15)
plt.title('Figure 1 - Raw Data (whole dataset)')


f_freq, t_seg, Sxx = spectrogram(x, fs=sampling_frequency, nperseg=256, nfft=8192, return_onesided=True, scaling='spectrum')

plt.figure()
plt.pcolormesh(t_seg, f_freq, 10 * np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
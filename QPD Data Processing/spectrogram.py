import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from pathlib import Path

def load_data(filepath, skiprows):
    raw = np.loadtxt(filepath, dtype=float , delimiter=',', skiprows=skiprows)

    t = raw[:,0] # time [microseconds]
    t_step = np.mean(np.gradient(t))
    t_step = t_step * 1e-6 # sampling time step [seconds]
    sampling_frequency = 1 / t_step # [Hz]
    print(sampling_frequency / 1000)
    
    # t = t - t[0] # elapsed time [microseconds]
    # t = t * 1e-6 # elapsed time [seconds]
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

#filepath = 'data\Test with new optics_20-02-05_15-59-35 (2).csv'
folderpath = Path("data/concat set")
skiprows = 4
##

for num in range(1, 5):
    filename = "data(" + str(num) + ").csv"
    filepath = folderpath / filename
    x, y, t = load_data(filepath, skiprows)
    f_freq, t_seg, Sxx = spectrogram(x, fs=sampling_frequency, nperseg=256, nfft=8192, return_onesided=True, scaling='spectrum')


# plotting

plt.plot(t, x, 'k-')
plt.xlabel('time (s)', fontsize=15)
plt.ylabel('QPD x-position (normalised)', fontsize=15)
plt.title('Figure 1 - Raw Data (whole dataset)')

plt.figure()
plt.pcolormesh(t_seg, f_freq, 10 * np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
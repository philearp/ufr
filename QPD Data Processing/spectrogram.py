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
    
    # t = t - t[0] # elapsed time [microseconds]
    # t = t * 1e-6 # elapsed time [seconds]
    x_raw = raw[:,1]
    y_raw = raw[:,2]
    intensity = raw[:,3]

    # normalise qpd data
    x = np.divide(x_raw, intensity)
    y = np.divide(y_raw, intensity)

    return x, y, sampling_frequency


def load_spectrogram(num, folderpath, skiprows):
    filename = "data(" + str(num) + ").csv"
    filepath = folderpath / filename
    x, y, sampling_frequency = load_data(filepath, skiprows)
    f_freq, t_seg, Sxx = spectrogram(x, fs=sampling_frequency, nperseg=256, nfft=8192, return_onesided=True, scaling='spectrum')

    print("file " + str(num) 
    + ": Sampling frequency = {0:.2f} kHz, ".format(sampling_frequency / 1000) 
    + str(len(x)) + " time steps, "
    + str(len(f_freq)) + " frequency channels, " 
    + str(len(t_seg)) + " time steps")
    return f_freq, t_seg, Sxx

# ---------------------- Code Begins Here ------------------   

## Inputs
folderpath = Path("data/concat set")
skiprows = 4
##


# first file
num = 1
f_freq, t_seg, Sxx = load_spectrogram(num, folderpath, skiprows)
# subsequent files
for num in range(2, 10):
    f_freq_tmp, t_seg_tmp, Sxx_tmp = load_spectrogram(num, folderpath, skiprows)

    # make time stamps continuous
    t_seg_tmp = t_seg_tmp + t_seg[-1] 

    # concatanate datasets
    t_seg = np.hstack((t_seg, t_seg_tmp))
    Sxx = np.hstack((Sxx, Sxx_tmp))


# plotting

# plt.plot(t, x, 'k-')
# plt.xlabel('time (s)', fontsize=15)
# plt.ylabel('QPD x-position (normalised)', fontsize=15)
# plt.title('Figure 1 - Raw Data (whole dataset)')

plt.figure()
plt.pcolormesh(t_seg, f_freq, 10 * np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
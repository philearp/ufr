import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.signal import hanning

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

def crop_data(x, y, t, t_min, t_max):
    ii = np.logical_and(t < t_tmax, t > t_min)
    t = t[ii]
    x = x[ii]
    y = y[ii]
    intensity = intensity[ii]

    return x, y, t

def plot_raw_data(x, t, times_to_display):
    T_osc = 1 / f_osc # oscillation time period [s]

    plt.plot(t, x, 'k-')
    plt.xlabel('time (s)', fontsize=15)
    plt.ylabel('QPD x-position (normalised)', fontsize=15)
    plt.title('Figure 1 - Raw Data (whole dataset)')
    # plt.savefig('QPD Raw Data ON.png')

    plt.figure()
    plt.plot(t*1000, x, 'ko-')
    plt.xlabel('time (ms)', fontsize=15)
    plt.ylabel('QPD x-position (normalised)', fontsize=15)
    plt.xlim(times_to_display[0] * 1000, times_to_display[1] * 1000)
    plt.title('Figure 2 - Raw Data (expanded view)')
    # plt.savefig('QPD Raw Data ON.png')

    plt.figure()
    plt.plot(t / T_osc, x, 'ko-')
    plt.xlabel('cycles (n)', fontsize=15)
    plt.ylabel('QPD x-position (normalised)', fontsize=15)
    plt.xlim(times_to_display[0] / T_osc, times_to_display[1] / T_osc)
    plt.title('Figure 3 - Raw Data (expanded view, #cycles)')
    return

def calc_fft(x, sampling_frequency, w):
    sampling_time_period = 1 / sampling_frequency # sampling time period [s]

    # calculate fft of QPD x signal
    f = fft(x * w) # complex
    # calculate power spectral density (square of abs. value)
    p = np.abs(f) ** 2 # power spectral density
    # calculate frequency values of the PSD
    f_freq = fftfreq(len(p), sampling_time_period)
    return f, p, f_freq

def select_fundamental_frequency(f_freq, psd):
    # ususally a large DC component, so only look at positive frequencies
    # psd(fftfreq == 0) = 0
    is_positive_freq = f_freq > 0
    f_freq = f_freq[is_positive_freq]
    psd = psd[is_positive_freq]

    f_fund = f_freq[psd == max(psd)]

    return f_fund[0] # adding [0] returns number instead of array

def plot_frequency_spectra(f, p, f_freq):
    # apply fftshift to make 0Hz at centre of spectrum
    p = fftshift(p)
    f = fftshift(f)
    f_freq = fftshift(f_freq)

    plt.figure()
    plt.plot(f_freq, 10 * np.log10(p)) # logarithmic
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB)')
    plt.title('Figure 4 - Raw Data Frequency Spectrum (log scale)')

    plt.figure()
    plt.plot(f_freq, p)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (arb. units)')
    plt.title('Figure 5 - Raw Data Frequency Spectrum (abs scale)')
    return

def filter_frequency_spectrum(f, f_freq, filter_half_width):
    # apply fftshift to make 0Hz at centre of spectrum
    f = fftshift(f)
    f_freq = fftshift(f_freq)

    # identify cutoff frequencies around +ve peak
    upper_filter_cutoff = f_fund + filter_half_width
    lower_filter_cutoff = f_fund - filter_half_width

    # filter the frequency spectrum
    f_filt = f.copy()
    f_to_keep = np.zeros_like(f)
    f_to_keep[np.abs(f_freq - f_fund) < filter_half_width] = True # positive freqs
    f_to_keep[np.abs(f_freq - (-f_fund)) < filter_half_width] = True # negative freqs
    f_to_remove = np.logical_not(f_to_keep)
    f_filt[f_to_remove] = 0

    # calculate PSD of filtered spectrum
    p_filt = np.abs(f_filt) ** 2 # power spectral density

    # Plot filtered PSD
    plt.figure()
    plt.plot(f_freq, p_filt)
    plt.xlabel('Frequency (Hz)', fontsize=15)
    plt.ylabel('Filtered PSD (arb. units)', fontsize=15)

    # remove fftshift
    f_filt = ifftshift(f_filt)
    p_filt = ifftshift(p_filt)
    return f_filt, p_filt

def reconstruct_signal(f_filt, w, remove_window):
    # Compute IFFT to reconstruct signal from filtered FFT
    x_filt = np.real(ifft(f_filt))

    if remove_window:
        # modify window to allow division by it (remove zeros)
        w_modified = w.copy()
        w_is_zero = (w_modified == 0)
        w_modified[w_is_zero] = 1

        x_filt = (x_filt / w_modified) + np.mean(x)
        x_filt[w_is_zero] = 0
    return x_filt

def plot_filtered_signal(x, x_filt, t):
    plt.figure()
    plt.plot(t, x, 'k-')
    plt.plot(t, x_filt + np.mean(x), 'r-')
    plt.xlabel('time (s)', fontsize=15)
    plt.ylabel('QPD x-value (V)', fontsize=15)
    plt.title('Figure 6 - Fundamental Frequency +-' + str(filter_half_width) + 'Hz')
    return
# ---------------------- Code Begins Here ------------------

## Inputs
f_osc = 20e3 # approx. oscillation frequency [Hz]
sampling_frequency = 200e3 # sampling frequency [Hz]

times_to_display = [0.2, 0.20025]

filepath = 'data\Test with new optics_20-02-05_15-59-35 (2).csv'
skiprows = 4

filter_half_width = 5 # Hz
## end of inputs

# Load Data
x, y, t = load_data(filepath, skiprows)

# Plot graphs of raw data
plot_raw_data(y, t, times_to_display)

# FFT Calculation
w = hanning(len(x)) # window function
f, p, f_freq = calc_fft(x, sampling_frequency, w)

# Select fundamental frequency 
f_fund = select_fundamental_frequency(f_freq, p)
print("Resonance Frequency = {0:.3f} Hz".format(f_fund))

# Plot frequency spectra
plot_frequency_spectra(f, p, f_freq)

# Filter FFT spectrum
f_filt, p_filt = filter_frequency_spectrum(f, f_freq, filter_half_width)

# integrate spectrum
peak_intensity = np.trapz(p_filt, x=f_freq)
print("Resonant peak intensity = {0:.2e}".format(peak_intensity))

# Reconstruct signal from filtered spectrum
remove_window = False
x_filt = reconstruct_signal(f_filt, w, remove_window)

# plot filtered signal
plot_filtered_signal(x, x_filt, t)

plt.show()
print('done')
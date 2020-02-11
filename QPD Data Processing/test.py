import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.fftpack import fft, fftfreq, fftshift

##
f_osc = 20e3 # approx. oscillation frequency [Hz]
f_samp = 200e3 # sampling frequency [Hz]

times_to_display = [0.2, 0.2005]
##

raw = np.loadtxt('data/Test with new optics_20-02-05_15-59-35 (2).csv', dtype=float , delimiter=',', skiprows=4)

t = raw[:,0] # time [microseconds]
t = t - t[0] # elapsed time [microseconds]
t = t * 1e-6 # elapsed time [seconds]
x_raw = raw[:,1]
y_raw = raw[:,2]
intensity = raw[:,3]

# normalise qpd data
x = np.divide(x_raw, intensity)
y = np.divide(y_raw, intensity)

ii = t < 0.6
t = t[ii]
x = x[ii]
y = y[ii]
intensity = intensity[ii]

x=y

plt.plot(t, x, 'k-')
plt.xlabel('time (s)', fontsize=15)
plt.ylabel('QPD x-position', fontsize=15)
plt.title('Figure 1 - Raw Data (whole dataset)')
# plt.savefig('QPD Raw Data ON.png')

plt.figure()
plt.plot(t*1000, x, 'ko-')
plt.xlabel('time (ms)', fontsize=15)
plt.ylabel('QPD x-position', fontsize=15)
plt.xlim(times_to_display[0] * 1000, times_to_display[1] * 1000)
plt.title('Figure 2 - Raw Data (expanded view)')
# plt.savefig('QPD Raw Data ON.png')

T_osc = 1 / f_osc # oscillation time period [s]

plt.figure()
plt.plot(t / T_osc, x, 'ko-')
plt.xlabel('cycles (n)', fontsize=15)
plt.ylabel('QPD x-position', fontsize=15)
plt.xlim(times_to_display[0] / T_osc, times_to_display[1] / T_osc)
plt.title('Figure 3 - Raw Data (expanded view, #cycles)')

## FFT calculation

T_samp = 1 / f_samp # sampling time period [s]

# calculate fft of QPD x signal
f = fft(x) # complex
# calculate power spectral density (square of abs. value)
p = np.abs(f) ** 2 # power spectral density
# calculate frequency values of the PSD
f_freq = fftfreq(len(p), T_samp)

# apply fftshift to make 0Hz at centre of spectrum
p = fftshift(p)
f = fftshift(p)
f_freq = fftshift(f_freq)

plt.figure()
plt.plot(f_freq, 10 * np.log10(p)) # logarithmic
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB)')
plt.title('Figure 4 - Raw Data Frequency Spectrum (log scale)')
plt.show()

plt.figure()
plt.plot(f_freq, p)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (arb. units)')
plt.title('Figure 5 - Raw Data Frequency Spectrum (abs scale)')

## Select fundamental frequency 
def select_fundamental_frequency(f_freq, psd):
    # ususally a large DC component, so only look at positive frequencies
    # psd(fftfreq == 0) = 0
    is_positive_freq = f_freq > 0
    f_freq = f_freq[is_positive_freq]
    psd = psd[is_positive_freq]

    f_fund = f_freq[psd == max(psd)]

    return f_fund[0] # adding [0] returns number instead of array

f_fund = select_fundamental_frequency(f_freq, p)
print("Resonance Frequency = {0:.3f} Hz".format(f_fund))


# filtering frequency spectrum
filter_half_width = 10 # Hz
upper_filter_cutoff = f_fund + filter_half_width
lower_filter_cutoff = f_fund - filter_half_width

f_filt = f.copy()
f_filt[np.abs(f_freq > upper_filter_cutoff)] = 0
f_filt[np.abs(f_freq < lower_filter_cutoff)] = 0

x_filt = 2 * np.real(sp.fftpack.ifft(f_filt))

plt.figure()
plt.plot(t, x, 'k-')
plt.plot(t, x_filt + np.mean(x), 'r-')
plt.xlabel('time (s)', fontsize=15)
plt.ylabel('QPD x-value (V)', fontsize=15)
plt.title('Figure 6 - Fundamental Frequency +-' + str(filter_half_width) + 'Hz')

plt.show()
print('done')
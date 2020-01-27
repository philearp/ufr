import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.fftpack

##
f_osc = 20e3 # approx. oscillation frequency [Hz]
f_samp = 100e3 # sampling frequency [Hz]

times_to_display = [0.2, 0.2005]
##

raw = np.loadtxt('first good test_19-12-05_11-35-02 (5).csv', dtype=float , delimiter=',', skiprows=2)
t = raw[:,0] # time [microseconds]
t = t - t[0] # elapsed time [microseconds]
t = t * 1e-6 # elapsed time [seconds]
x = raw[:,1]
y = raw[:,2]
intensity = raw[:,3]

ii = t < 0.6
t = t[ii]
x = x[ii]
y = y[ii]
intensity = intensity[ii]

x=y

plt.plot(t, x, 'k-')
plt.xlabel('time (s)', fontsize=15)
plt.ylabel('QPD x-value (V)', fontsize=15)
# plt.savefig('QPD Raw Data ON.png')

plt.figure()
plt.plot(t*1000, x, 'ko-')
plt.xlabel('time (ms)', fontsize=15)
plt.ylabel('QPD x-value (V)', fontsize=15)
plt.xlim(times_to_display[0] * 1000, times_to_display[1] * 1000)
# plt.savefig('QPD Raw Data ON.png')

T_osc = 1 / f_osc # oscillation time period [s]

plt.figure()
plt.plot(t / T_osc, x, 'ko-')
plt.xlabel('cycles (n)', fontsize=15)
plt.ylabel('QPD x-value (V)', fontsize=15)
plt.xlim(times_to_display[0] / T_osc, times_to_display[1] / T_osc)

## FFT calculation

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

## Select fundamental frequency 
def select_fundamental_frequency(fftfreq, psd):
    # ususally a large DC component, so only look at positive frequencies
    # psd(fftfreq == 0) = 0
    is_positive_freq = fftfreq > 0
    fftfreq = fftfreq[is_positive_freq]
    psd = psd[is_positive_freq]

    f_fund = fftfreq[psd == max(psd)]

    return f_fund[0] # adding [0] returns number instead of array

f_fund = select_fundamental_frequency(fftfreq, x_psd)

filter_half_width = 10 # Hz
upper_filter_cutoff = f_fund + filter_half_width
lower_filter_cutoff = f_fund - filter_half_width

x_fft_fund = x_fft.copy()
x_fft_fund[np.abs(fftfreq > upper_filter_cutoff)] = 0
x_fft_fund[np.abs(fftfreq < lower_filter_cutoff)] = 0

x_fund = np.real(sp.fftpack.ifft(x_fft_fund))

plt.figure()
plt.plot(t, x, 'k-')
plt.plot(t, x_fund + np.mean(x), 'r-')
plt.xlabel('time (s)', fontsize=15)
plt.ylabel('QPD x-value (V)', fontsize=15)

plt.show()
print('done')
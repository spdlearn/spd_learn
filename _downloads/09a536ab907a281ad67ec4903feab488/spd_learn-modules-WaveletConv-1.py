import torch
import numpy as np
import matplotlib.pyplot as plt
from spd_learn.modules import WaveletConv
from spd_learn.functional import compute_gabor_wavelet

# Create wavelet filterbank
foi_init = [2.0, 3.0, 4.0, 5.0]  # 4, 8, 16, 32 Hz
sfreq = 250
wavelet = WaveletConv(kernel_width_s=0.5, foi_init=foi_init, sfreq=sfreq)

# Get wavelet kernels
tt = wavelet.tt.numpy()
wavelets = compute_gabor_wavelet(
    wavelet.tt, wavelet.foi, wavelet.fwhm, sfreq=sfreq
).detach().numpy()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

freq_labels = [f'{2**f:.0f} Hz' for f in foi_init]

for i, (ax, wav, label) in enumerate(zip(axes.flat, wavelets, freq_labels)):
    ax.plot(tt * 1000, wav.real, 'b-', label='Real', alpha=0.8)
    ax.plot(tt * 1000, wav.imag, 'r-', label='Imag', alpha=0.8)
    ax.fill_between(tt * 1000, -np.abs(wav), np.abs(wav),
                    color='gray', alpha=0.2, label='Envelope')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Wavelet at {label}', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(tt[0]*1000, tt[-1]*1000)

plt.suptitle('Gabor Wavelet Filterbank', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
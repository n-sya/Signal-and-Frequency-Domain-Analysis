import matplotlib.pyplot as plt
import numpy as np

from frequency_analysis.frequency_analysis import (
    discrete_fourier_transform,
    inverse_discrete_fourier_transform,
)


# Generate a signal containing two frequency components
sampling_frequency = 500.0
duration = 1.0

time = np.arange(0.0, duration, 1.0 / sampling_frequency)

signal = np.sin(2.0 * np.pi * 20.0 * time) + 0.5 * np.sin(
    2.0 * np.pi * 50.0 * time
)


# Calculate the DFT using the manual implementation
manual_transform = discrete_fourier_transform(signal)

# Compare the manual implementation against NumPy
numpy_transform = np.fft.fft(signal)

transform_error = np.max(np.abs(manual_transform - numpy_transform))


# Reconstruct the original signal using the manual IDFT
reconstructed_signal = inverse_discrete_fourier_transform(manual_transform)

reconstruction_error = np.max(np.abs(signal - reconstructed_signal.real))


# Display verification results
print(f"Maximum DFT error: {transform_error:.3e}")
print(f"Maximum reconstruction error: {reconstruction_error:.3e}")


# Plot the original and reconstructed signals
plt.figure(figsize=(10, 5))

plt.plot(time, signal, label="Original signal")
plt.plot(
    time,
    reconstructed_signal.real,
    "--",
    label="Reconstructed signal",
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("DFT and IDFT Signal Reconstruction")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    "outputs/dft_reconstruction.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
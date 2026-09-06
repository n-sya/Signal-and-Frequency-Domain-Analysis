import matplotlib.pyplot as plt
import numpy as np

from frequency_analysis.frequency_analysis import enforce_conjugate_symmetry


# Generate a real-valued signal
sampling_frequency = 500.0
duration = 1.0

time = np.arange(0.0, duration, 1.0 / sampling_frequency)

signal = np.sin(2.0 * np.pi * 20.0 * time) + 0.5 * np.sin(
    2.0 * np.pi * 50.0 * time
)


# Calculate the FFT
transform = np.fft.fft(signal)

number_of_samples = signal.size


# Retain only the non-negative frequency components
partial_transform = transform.copy()

if number_of_samples % 2 == 0:
    partial_transform[number_of_samples // 2 + 1 :] = 0.0
else:
    partial_transform[(number_of_samples + 1) // 2 :] = 0.0


# Reconstruct the negative-frequency components using conjugate symmetry
reconstructed_transform = enforce_conjugate_symmetry(partial_transform)

reconstructed_signal = np.fft.ifft(reconstructed_transform)


# Calculate reconstruction errors
spectrum_error = np.max(np.abs(transform - reconstructed_transform))

signal_error = np.max(np.abs(signal - reconstructed_signal.real))


# Display verification results
print(f"Maximum spectrum reconstruction error: {spectrum_error:.3e}")
print(f"Maximum signal reconstruction error: {signal_error:.3e}")


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
plt.title("FFT Reconstruction Using Conjugate Symmetry")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    "outputs/fft_conjugate_symmetry.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
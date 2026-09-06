import numpy as np

from frequency_analysis.frequency_analysis import (
    discrete_fourier_transform,
    inverse_discrete_fourier_transform,
    single_sided_amplitude_spectrum,
)
from signal_processing.signal_processing import (
    beat_frequency,
    hilbert_envelope,
)


# Generate a demonstration signal
sampling_frequency = 500.0
duration = 1.0

time = np.arange(0.0, duration, 1.0 / sampling_frequency)

signal = np.sin(2.0 * np.pi * 20.0 * time) + 0.5 * np.sin(
    2.0 * np.pi * 50.0 * time
)


# Demonstrate DFT and IDFT
transform = discrete_fourier_transform(signal)
reconstructed_signal = inverse_discrete_fourier_transform(transform)

reconstruction_error = np.max(
    np.abs(signal - reconstructed_signal.real)
)


# Demonstrate frequency-domain analysis
frequencies, amplitudes = single_sided_amplitude_spectrum(
    signal,
    sampling_frequency,
)

dominant_indices = np.argsort(amplitudes)[-2:]
dominant_frequencies = np.sort(frequencies[dominant_indices])


# Demonstrate Hilbert-transform analysis
beating_signal = np.sin(2.0 * np.pi * 20.0 * time) + np.sin(
    2.0 * np.pi * 25.0 * time
)

envelope = hilbert_envelope(beating_signal)
calculated_beat_frequency = beat_frequency(
    beating_signal,
    sampling_frequency,
)


# Display results
print("Signal and Frequency Domain Analysis")
print("------------------------------------")
print(f"DFT reconstruction error: {reconstruction_error:.3e}")
print(
    "Dominant frequencies: "
    f"{dominant_frequencies[0]:.2f} Hz, "
    f"{dominant_frequencies[1]:.2f} Hz"
)
print(f"Beat frequency: {calculated_beat_frequency:.2f} Hz")
print(f"Hilbert envelope samples: {envelope.size}")
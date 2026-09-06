import matplotlib.pyplot as plt
import numpy as np

from signal_processing.signal_processing import (
    beat_frequency,
    dominant_frequencies,
    hilbert_envelope,
)


# Generate a beating signal from two nearby frequencies
sampling_frequency = 1000.0
duration = 2.0

frequency_1 = 20.0
frequency_2 = 25.0

time = np.arange(0.0, duration, 1.0 / sampling_frequency)

signal = np.sin(2.0 * np.pi * frequency_1 * time) + np.sin(
    2.0 * np.pi * frequency_2 * time
)


# Calculate the Hilbert envelope
envelope = hilbert_envelope(signal)


# Identify the dominant frequencies and beat frequency
frequencies = dominant_frequencies(
    signal,
    sampling_frequency,
    number_of_peaks=2,
)

calculated_beat_frequency = beat_frequency(
    signal,
    sampling_frequency,
)


# Display calculated results
print(f"Detected frequencies: {frequencies[0]:.2f} Hz, {frequencies[1]:.2f} Hz")
print(f"Calculated beat frequency: {calculated_beat_frequency:.2f} Hz")


# Plot the signal and Hilbert envelope
plt.figure(figsize=(10, 5))

plt.plot(time, signal, label="Beating signal")
plt.plot(time, envelope, "--", label="Upper envelope")
plt.plot(time, -envelope, "--", label="Lower envelope")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Beat Frequency Detection Using the Hilbert Envelope")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    "outputs/beat_frequency_hilbert.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
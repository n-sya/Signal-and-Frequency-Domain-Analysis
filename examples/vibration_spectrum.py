import matplotlib.pyplot as plt
import numpy as np

from frequency_analysis.frequency_analysis import single_sided_amplitude_spectrum


# Load vibration displacement data
signal = np.loadtxt("data/Vibration.txt")


# Reconstruct the time axis using the original sampling interval
time_step = 0.01
sampling_frequency = 1.0 / time_step

number_of_samples = signal.size
time = np.arange(number_of_samples) * time_step


# Calculate the single-sided amplitude spectrum
frequencies, amplitudes = single_sided_amplitude_spectrum(
    signal,
    sampling_frequency,
)


# Convert frequency from Hz to angular frequency
angular_frequencies = 2.0 * np.pi * frequencies


# Ignore the zero-frequency component when identifying the dominant frequency
peak_index = np.argmax(amplitudes[1:]) + 1

dominant_frequency = frequencies[peak_index]
dominant_angular_frequency = angular_frequencies[peak_index]


# Display calculated results
print(f"Sampling frequency: {sampling_frequency:.2f} Hz")
print(f"Dominant frequency: {dominant_frequency:.2f} Hz")
print(f"Dominant angular frequency: {dominant_angular_frequency:.2f} rad/s")


# Plot the vibration signal in the time domain
plt.figure(figsize=(10, 5))

plt.plot(time, signal)

plt.xlabel("Time (s)")
plt.ylabel("Displacement")
plt.title("Vibration Signal")
plt.grid()

plt.tight_layout()
plt.savefig(
    "outputs/vibration_signal.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# Plot the vibration spectrum in the frequency domain
plt.figure(figsize=(10, 5))

plt.plot(angular_frequencies, amplitudes)

plt.xlabel("Angular Frequency (rad/s)")
plt.ylabel("Amplitude")
plt.title("Vibration Frequency Spectrum")
plt.grid()

plt.tight_layout()
plt.savefig(
    "outputs/vibration_spectrum.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
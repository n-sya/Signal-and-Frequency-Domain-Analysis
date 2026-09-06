import matplotlib.pyplot as plt
import numpy as np

from signal_processing.signal_processing import low_pass_filter_response


# Load the noisy signal
signal = np.loadtxt("data/Noisy.txt")


# Reconstruct the time axis using the original sampling interval
time_step = 0.05
sampling_frequency = 1.0 / time_step

number_of_samples = signal.size
time = np.arange(number_of_samples) * time_step


# Transform the noisy signal into the frequency domain
transform = np.fft.fft(signal)
frequencies = np.fft.fftfreq(
    number_of_samples,
    d=time_step,
)


# Apply a first-order low-pass filter
corner_frequency = 1.0

filter_response = low_pass_filter_response(
    frequencies,
    corner_frequency,
)

filtered_transform = transform * filter_response


# Reconstruct the filtered signal in the time domain
filtered_signal = np.fft.ifft(filtered_transform).real


# Calculate single-sided spectra for plotting
positive_frequencies = np.fft.rfftfreq(
    number_of_samples,
    d=time_step,
)

original_spectrum = np.abs(np.fft.rfft(signal)) / number_of_samples
filtered_spectrum = np.abs(np.fft.rfft(filtered_signal)) / number_of_samples

if number_of_samples % 2 == 0:
    original_spectrum[1:-1] *= 2.0
    filtered_spectrum[1:-1] *= 2.0
else:
    original_spectrum[1:] *= 2.0
    filtered_spectrum[1:] *= 2.0


# Display filter settings
print(f"Sampling frequency: {sampling_frequency:.2f} Hz")
print(f"Low-pass corner frequency: {corner_frequency:.2f} Hz")


# Plot the noisy and filtered time-domain signals
plt.figure(figsize=(10, 5))

plt.plot(time, signal, label="Noisy signal")
plt.plot(time, filtered_signal, label="Filtered signal")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Frequency-Domain Low-Pass Filtering")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    "outputs/frequency_domain_filtering.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# Plot the original and filtered spectra
plt.figure(figsize=(10, 5))

plt.plot(
    positive_frequencies,
    original_spectrum,
    label="Original spectrum",
)
plt.plot(
    positive_frequencies,
    filtered_spectrum,
    label="Filtered spectrum",
)

plt.axvline(
    corner_frequency,
    linestyle="--",
    label="Corner frequency",
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Effect of Low-Pass Filtering in the Frequency Domain")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    "outputs/frequency_domain_filtering_spectrum.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
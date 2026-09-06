import numpy as np


# Construct the analytic signal using the Hilbert transform
def analytic_signal(signal):
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("signal must be a one-dimensional array.")

    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")

    number_of_samples = signal.size
    transform = np.fft.fft(signal)

    hilbert_multiplier = np.zeros(number_of_samples)

    if number_of_samples % 2 == 0:
        hilbert_multiplier[0] = 1.0
        hilbert_multiplier[1 : number_of_samples // 2] = 2.0
        hilbert_multiplier[number_of_samples // 2] = 1.0
    else:
        hilbert_multiplier[0] = 1.0
        hilbert_multiplier[1 : (number_of_samples + 1) // 2] = 2.0

    return np.fft.ifft(transform * hilbert_multiplier)


# Calculate the amplitude envelope of a signal using the Hilbert transform
def hilbert_envelope(signal):
    return np.abs(analytic_signal(signal))


# Find the dominant frequencies in a signal
def dominant_frequencies(signal, sampling_frequency, number_of_peaks=2):
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("signal must be a one-dimensional array.")

    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")

    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")

    if number_of_peaks < 1:
        raise ValueError("number_of_peaks must be at least 1.")

    transform = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(
        signal.size, d=1.0 / sampling_frequency
    )
    amplitudes = np.abs(transform)

    if amplitudes.size > 1:
        amplitudes[0] = 0.0

    number_of_peaks = min(number_of_peaks, amplitudes.size)

    peak_indices = np.argsort(amplitudes)[-number_of_peaks:]
    peak_indices = peak_indices[np.argsort(frequencies[peak_indices])]

    return frequencies[peak_indices]


# Calculate the beat frequency from the two strongest frequency components
def beat_frequency(signal, sampling_frequency):
    frequencies = dominant_frequencies(
        signal,
        sampling_frequency,
        number_of_peaks=2,
    )

    if frequencies.size < 2:
        raise ValueError("At least two frequency components are required.")

    return abs(frequencies[1] - frequencies[0])

# Calculate the frequency response of a first-order low-pass filter
def low_pass_filter_response(frequencies, corner_frequency):
    frequencies = np.asarray(frequencies, dtype=float)

    if corner_frequency <= 0:
        raise ValueError("corner_frequency must be positive.")

    return 1.0 / (1.0 + 1j * frequencies / corner_frequency)

# Calculate the frequency response of a first-order high-pass filter
def high_pass_filter_response(frequencies, corner_frequency):
    frequencies = np.asarray(frequencies, dtype=float)

    if corner_frequency <= 0:
        raise ValueError("corner_frequency must be positive.")

    return 1j * frequencies / (
        corner_frequency + 1j * frequencies
    )
import numpy as np


# Discrete Fourier Transform implemented directly from the DFT definition
def discrete_fourier_transform(signal):
    signal = np.asarray(signal, dtype=complex)

    if signal.ndim != 1:
        raise ValueError("signal must be a one-dimensional array.")

    number_of_samples = signal.size

    if number_of_samples < 1:
        raise ValueError("signal must contain at least one sample.")

    transform = np.zeros(number_of_samples, dtype=complex)

    for k in range(number_of_samples):
        for n in range(number_of_samples):
            angle = -2.0 * np.pi * k * n / number_of_samples
            transform[k] += signal[n] * np.exp(1j * angle)

    return transform


# Inverse Discrete Fourier Transform implemented directly from the IDFT definition
def inverse_discrete_fourier_transform(transform):
    transform = np.asarray(transform, dtype=complex)

    if transform.ndim != 1:
        raise ValueError("transform must be a one-dimensional array.")

    number_of_samples = transform.size

    if number_of_samples < 1:
        raise ValueError("transform must contain at least one value.")

    signal = np.zeros(number_of_samples, dtype=complex)

    for n in range(number_of_samples):
        for k in range(number_of_samples):
            angle = 2.0 * np.pi * k * n / number_of_samples
            signal[n] += transform[k] * np.exp(1j * angle)

        signal[n] /= number_of_samples

    return signal


# Generate the frequency bins corresponding to a DFT or FFT
def frequency_axis(number_of_samples, sampling_frequency):
    if number_of_samples < 1:
        raise ValueError("number_of_samples must be at least 1.")

    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")

    return np.fft.fftfreq(number_of_samples, d=1.0 / sampling_frequency)


# Calculate the single-sided amplitude spectrum of a real-valued signal
def single_sided_amplitude_spectrum(signal, sampling_frequency):
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("signal must be a one-dimensional array.")

    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")

    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")

    number_of_samples = signal.size

    transform = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(
        number_of_samples, d=1.0 / sampling_frequency
    )
    amplitudes = np.abs(transform) / number_of_samples

    if number_of_samples % 2 == 0:
        amplitudes[1:-1] *= 2.0
    else:
        amplitudes[1:] *= 2.0

    return frequencies, amplitudes


# Reconstruct the negative-frequency FFT bins using conjugate symmetry
def enforce_conjugate_symmetry(transform):
    transform = np.asarray(transform, dtype=complex).copy()

    if transform.ndim != 1:
        raise ValueError("transform must be a one-dimensional array.")

    number_of_samples = transform.size

    if number_of_samples < 1:
        raise ValueError("transform must contain at least one value.")

    transform[0] = transform[0].real

    if number_of_samples % 2 == 0:
        nyquist_index = number_of_samples // 2
        transform[nyquist_index] = transform[nyquist_index].real
        positive_indices = range(1, nyquist_index)
    else:
        positive_indices = range(1, (number_of_samples + 1) // 2)

    for k in positive_indices:
        transform[-k] = np.conj(transform[k])

    return transform
import unittest

import numpy as np

from frequency_analysis.frequency_analysis import (
    discrete_fourier_transform,
    enforce_conjugate_symmetry,
    frequency_axis,
    inverse_discrete_fourier_transform,
    single_sided_amplitude_spectrum,
)


class TestDiscreteFourierTransform(unittest.TestCase):
    def test_matches_numpy_fft(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0])

        result = discrete_fourier_transform(signal)
        expected = np.fft.fft(signal)

        np.testing.assert_allclose(result, expected, atol=1.0e-10)


class TestInverseDiscreteFourierTransform(unittest.TestCase):
    def test_reconstructs_signal(self):
        signal = np.array([1.0, -2.0, 0.5, 3.0])

        transform = discrete_fourier_transform(signal)
        reconstructed = inverse_discrete_fourier_transform(transform)

        np.testing.assert_allclose(reconstructed.real, signal, atol=1.0e-10)
        np.testing.assert_allclose(reconstructed.imag, 0.0, atol=1.0e-10)


class TestFrequencyAxis(unittest.TestCase):
    def test_matches_numpy(self):
        result = frequency_axis(
            number_of_samples=8,
            sampling_frequency=1000.0,
        )

        expected = np.fft.fftfreq(
            8,
            d=1.0 / 1000.0,
        )

        np.testing.assert_allclose(result, expected)


class TestSingleSidedAmplitudeSpectrum(unittest.TestCase):
    def test_detects_known_frequency_and_amplitude(self):
        sampling_frequency = 1000.0
        duration = 1.0
        frequency = 50.0
        amplitude = 2.0

        time = np.arange(
            0.0,
            duration,
            1.0 / sampling_frequency,
        )

        signal = amplitude * np.sin(2.0 * np.pi * frequency * time)

        frequencies, amplitudes = single_sided_amplitude_spectrum(
            signal,
            sampling_frequency,
        )

        peak_index = np.argmax(amplitudes)

        self.assertAlmostEqual(
            frequencies[peak_index],
            frequency,
            places=7,
        )

        self.assertAlmostEqual(
            amplitudes[peak_index],
            amplitude,
            places=7,
        )


class TestConjugateSymmetry(unittest.TestCase):
    def test_reconstructs_even_length_real_signal(self):
        signal = np.array([1.0, 2.0, -1.0, 0.5, 3.0, -2.0])

        transform = np.fft.fft(signal)
        partial_transform = transform.copy()

        partial_transform[signal.size // 2 + 1 :] = 0.0

        reconstructed_transform = enforce_conjugate_symmetry(
            partial_transform
        )
        reconstructed_signal = np.fft.ifft(reconstructed_transform)

        np.testing.assert_allclose(
            reconstructed_signal.real,
            signal,
            atol=1.0e-10,
        )

        np.testing.assert_allclose(
            reconstructed_signal.imag,
            0.0,
            atol=1.0e-10,
        )

    def test_reconstructs_odd_length_real_signal(self):
        signal = np.array([1.0, 2.0, -1.0, 0.5, 3.0])

        transform = np.fft.fft(signal)
        partial_transform = transform.copy()

        partial_transform[(signal.size + 1) // 2 :] = 0.0

        reconstructed_transform = enforce_conjugate_symmetry(
            partial_transform
        )
        reconstructed_signal = np.fft.ifft(reconstructed_transform)

        np.testing.assert_allclose(
            reconstructed_signal.real,
            signal,
            atol=1.0e-10,
        )

        np.testing.assert_allclose(
            reconstructed_signal.imag,
            0.0,
            atol=1.0e-10,
        )


if __name__ == "__main__":
    unittest.main()
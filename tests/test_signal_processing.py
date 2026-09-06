import unittest

import numpy as np

from signal_processing.signal_processing import (
    analytic_signal,
    beat_frequency,
    dominant_frequencies,
    hilbert_envelope,
    high_pass_filter_response,
    low_pass_filter_response,
)


class TestAnalyticSignal(unittest.TestCase):
    def test_real_part_reconstructs_original_signal(self):
        sampling_frequency = 1000.0
        time = np.arange(0.0, 1.0, 1.0 / sampling_frequency)

        signal = np.sin(2.0 * np.pi * 20.0 * time)
        result = analytic_signal(signal)

        np.testing.assert_allclose(result.real, signal, atol=1.0e-10)


class TestHilbertEnvelope(unittest.TestCase):
    def test_constant_amplitude_sine_wave(self):
        sampling_frequency = 1000.0
        time = np.arange(0.0, 1.0, 1.0 / sampling_frequency)

        amplitude = 3.0
        signal = amplitude * np.sin(2.0 * np.pi * 20.0 * time)

        envelope = hilbert_envelope(signal)

        np.testing.assert_allclose(
            envelope,
            amplitude,
            atol=1.0e-10,
        )


class TestDominantFrequencies(unittest.TestCase):
    def test_detects_two_known_frequencies(self):
        sampling_frequency = 1000.0
        time = np.arange(0.0, 2.0, 1.0 / sampling_frequency)

        signal = np.sin(2.0 * np.pi * 20.0 * time) + np.sin(
            2.0 * np.pi * 25.0 * time
        )

        frequencies = dominant_frequencies(
            signal,
            sampling_frequency,
            number_of_peaks=2,
        )

        expected = np.array([20.0, 25.0])

        np.testing.assert_allclose(
            frequencies,
            expected,
            atol=1.0e-10,
        )


class TestBeatFrequency(unittest.TestCase):
    def test_five_hertz_beat_frequency(self):
        sampling_frequency = 1000.0
        time = np.arange(0.0, 2.0, 1.0 / sampling_frequency)

        signal = np.sin(2.0 * np.pi * 20.0 * time) + np.sin(
            2.0 * np.pi * 25.0 * time
        )

        result = beat_frequency(signal, sampling_frequency)

        self.assertAlmostEqual(result, 5.0, places=10)

    def test_ten_hertz_beat_frequency(self):
        sampling_frequency = 1000.0
        time = np.arange(0.0, 2.0, 1.0 / sampling_frequency)

        signal = np.sin(2.0 * np.pi * 20.0 * time) + np.sin(
            2.0 * np.pi * 30.0 * time
        )

        result = beat_frequency(signal, sampling_frequency)

        self.assertAlmostEqual(result, 10.0, places=10)


class TestLowPassFilterResponse(unittest.TestCase):
    def test_corner_frequency_magnitude(self):
        corner_frequency = 10.0

        response = low_pass_filter_response(
            np.array([corner_frequency]),
            corner_frequency,
        )

        expected_magnitude = 1.0 / np.sqrt(2.0)

        self.assertAlmostEqual(
            np.abs(response[0]),
            expected_magnitude,
            places=10,
        )

    def test_low_frequency_passes(self):
        response = low_pass_filter_response(
            np.array([0.0]),
            corner_frequency=10.0,
        )

        self.assertAlmostEqual(np.abs(response[0]), 1.0, places=10)


class TestHighPassFilterResponse(unittest.TestCase):
    def test_corner_frequency_magnitude(self):
        corner_frequency = 10.0

        response = high_pass_filter_response(
            np.array([corner_frequency]),
            corner_frequency,
        )

        expected_magnitude = 1.0 / np.sqrt(2.0)

        self.assertAlmostEqual(
            np.abs(response[0]),
            expected_magnitude,
            places=10,
        )

    def test_zero_frequency_is_blocked(self):
        response = high_pass_filter_response(
            np.array([0.0]),
            corner_frequency=10.0,
        )

        self.assertAlmostEqual(np.abs(response[0]), 0.0, places=10)

if __name__ == "__main__":
    unittest.main()
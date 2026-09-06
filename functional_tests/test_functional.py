import json
import unittest
from pathlib import Path

import numpy as np

from frequency_analysis.frequency_analysis import (
    discrete_fourier_transform,
    inverse_discrete_fourier_transform,
    single_sided_amplitude_spectrum,
)
from signal_processing.signal_processing import (
    beat_frequency,
    high_pass_filter_response,
    hilbert_envelope,
    low_pass_filter_response,
)


TEST_CASES_PATH = Path(__file__).with_name("test_cases.json")


class TestSignalAnalysisFunctional(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with TEST_CASES_PATH.open("r", encoding="utf-8") as file:
            cls.test_cases = json.load(file)

    def test_discrete_fourier_transform(self):
        case = self.test_cases["dft"]

        signal = np.array(case["signal"], dtype=float)

        result = discrete_fourier_transform(signal)

        expected = np.array(case["expected_real"]) + 1j * np.array(
            case["expected_imaginary"]
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=case["tolerance"],
        )

    def test_inverse_discrete_fourier_transform(self):
        case = self.test_cases["inverse_dft"]

        transform = np.array(case["transform_real"]) + 1j * np.array(
            case["transform_imaginary"]
        )

        result = inverse_discrete_fourier_transform(transform)

        expected = np.array(case["expected_real"]) + 1j * np.array(
            case["expected_imaginary"]
        )

        np.testing.assert_allclose(
            result,
            expected,
            atol=case["tolerance"],
        )

    def test_single_sided_amplitude_spectrum(self):
        case = self.test_cases["amplitude_spectrum"]

        time = np.arange(case["number_of_samples"]) / case["sampling_frequency"]

        signal = case["amplitude"] * np.sin(
            2.0 * np.pi * case["frequency"] * time
        )

        frequencies, amplitudes = single_sided_amplitude_spectrum(
            signal,
            case["sampling_frequency"],
        )

        dominant_index = np.argmax(amplitudes)

        self.assertAlmostEqual(
            frequencies[dominant_index],
            case["expected_frequency"],
            delta=case["tolerance"],
        )
        self.assertAlmostEqual(
            amplitudes[dominant_index],
            case["expected_amplitude"],
            delta=case["tolerance"],
        )

    def test_hilbert_envelope(self):
        case = self.test_cases["hilbert_envelope"]

        time = np.arange(case["number_of_samples"]) / case["sampling_frequency"]

        signal = case["amplitude"] * np.sin(
            2.0 * np.pi * case["frequency"] * time
        )

        envelope = hilbert_envelope(signal)

        np.testing.assert_allclose(
            envelope,
            case["expected_amplitude"],
            atol=case["tolerance"],
        )

    def test_beat_frequency(self):
        case = self.test_cases["beat_frequency"]

        time = np.arange(case["number_of_samples"]) / case["sampling_frequency"]

        signal = np.sin(2.0 * np.pi * case["frequency_1"] * time)
        signal += np.sin(2.0 * np.pi * case["frequency_2"] * time)

        result = beat_frequency(
            signal,
            case["sampling_frequency"],
        )

        self.assertAlmostEqual(
            result,
            case["expected"],
            delta=case["tolerance"],
        )

    def test_filter_responses(self):
        case = self.test_cases["filter_response"]

        frequencies = np.array(case["frequencies"], dtype=float)

        low_pass = low_pass_filter_response(
            frequencies,
            case["corner_frequency"],
        )
        high_pass = high_pass_filter_response(
            frequencies,
            case["corner_frequency"],
        )

        np.testing.assert_allclose(
            np.abs(low_pass),
            case["expected_low_pass_magnitude"],
            atol=case["tolerance"],
        )
        np.testing.assert_allclose(
            np.abs(high_pass),
            case["expected_high_pass_magnitude"],
            atol=case["tolerance"],
        )


if __name__ == "__main__":
    unittest.main()
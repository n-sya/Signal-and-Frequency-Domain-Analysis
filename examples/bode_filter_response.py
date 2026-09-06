import matplotlib.pyplot as plt
import numpy as np

from signal_processing.signal_processing import (
    high_pass_filter_response,
    low_pass_filter_response,
)


# Define logarithmically spaced frequencies
frequencies = np.logspace(-2, 3, 1000)

low_pass_corner_frequency = 10.0
high_pass_corner_frequency = 1.0


# Calculate individual filter responses
low_pass_response = low_pass_filter_response(
    frequencies,
    low_pass_corner_frequency,
)

high_pass_response = high_pass_filter_response(
    frequencies,
    high_pass_corner_frequency,
)


# Calculate the cascaded filter response
cascade_response = low_pass_response * high_pass_response


# Convert magnitude responses to decibels
low_pass_magnitude_db = 20.0 * np.log10(
    np.abs(low_pass_response)
)

high_pass_magnitude_db = 20.0 * np.log10(
    np.abs(high_pass_response)
)

cascade_magnitude_db = 20.0 * np.log10(
    np.abs(cascade_response)
)


# Calculate phase responses
low_pass_phase = np.angle(
    low_pass_response,
    deg=True,
)

high_pass_phase = np.angle(
    high_pass_response,
    deg=True,
)

cascade_phase = np.angle(
    cascade_response,
    deg=True,
)


# Plot magnitude response
plt.figure(figsize=(10, 5))

plt.semilogx(
    frequencies,
    low_pass_magnitude_db,
    label="Low-pass filter",
)

plt.semilogx(
    frequencies,
    high_pass_magnitude_db,
    label="High-pass filter",
)

plt.semilogx(
    frequencies,
    cascade_magnitude_db,
    label="Cascaded response",
)

plt.axvline(
    high_pass_corner_frequency,
    linestyle="--",
    label="High-pass corner frequency",
)

plt.axvline(
    low_pass_corner_frequency,
    linestyle="--",
    label="Low-pass corner frequency",
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Filter Bode Magnitude Response")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    "outputs/bode_magnitude_response.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# Plot phase response
plt.figure(figsize=(10, 5))

plt.semilogx(
    frequencies,
    low_pass_phase,
    label="Low-pass filter",
)

plt.semilogx(
    frequencies,
    high_pass_phase,
    label="High-pass filter",
)

plt.semilogx(
    frequencies,
    cascade_phase,
    label="Cascaded response",
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (degrees)")
plt.title("Filter Bode Phase Response")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    "outputs/bode_phase_response.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
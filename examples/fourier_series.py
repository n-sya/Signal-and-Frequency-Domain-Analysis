import matplotlib.pyplot as plt
import numpy as np


# Generate the time axis
time = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 2000)


# Calculate truncated Fourier series for a square wave
def square_wave_fourier_series(time, number_of_terms):
    series = np.zeros_like(time)

    for n in range(number_of_terms):
        harmonic = 2 * n + 1
        series += np.sin(harmonic * time) / harmonic

    return 4.0 * series / np.pi


# Compare different numbers of Fourier terms
terms_to_compare = [1, 5, 20]

plt.figure(figsize=(10, 5))

for number_of_terms in terms_to_compare:
    approximation = square_wave_fourier_series(
        time,
        number_of_terms,
    )

    plt.plot(
        time,
        approximation,
        label=f"{number_of_terms} terms",
    )

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Square Wave Fourier Series Approximation")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    "outputs/fourier_series.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
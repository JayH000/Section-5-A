import numpy as np
import matplotlib.pyplot as plt


# Constants (you can change these)
m = 1.0       # Mass of the particle
omega = 1.0   # Frequency of the oscillator

# Define the phase space volume function Ω(E)
def phase_space_volume(E):
    return (2 * np.pi**2 / (m**2 * omega**4)) * E**2

# Define energy range
E_values = np.linspace(0.1, 10, 100)  # Avoid E=0 to prevent numerical issues

# Compute Ω(E) and g(E)
Omega_values = phase_space_volume(E_values)
g_values = np.gradient(Omega_values, E_values)  # Numerical derivative

plt.figure(figsize=(8, 5))
plt.plot(E_values, Omega_values, label=r'$\Omega(E)$ (Phase Space Volume)', linestyle='dashed', color='blue')
plt.plot(E_values, g_values, label=r'$g(E)$ (Density of States)', color='red')


for i in range(0, len(E_values), 20):  # Every 20th point
    E, Omega, g = E_values[i], Omega_values[i], g_values[i]
    plt.scatter(E, Omega, color='blue', marker='o')  # Mark points on Ω(E)
    plt.scatter(E, g, color='red', marker='x')  # Mark points on g(E)
    plt.text(E, Omega, f'{Omega:.1f}', fontsize=8, color='blue', ha='right')
    plt.text(E, g, f'{g:.1f}', fontsize=8, color='red', ha='left')

# Add theoretical equation as text
equation_text = r"$g(E) = \frac{4\pi^2}{m^2 \omega^4} E$"
plt.text(1, 3, equation_text, fontsize=12, color='black', ha='left', va='top')

plt.xlabel("Energy E")
plt.ylabel("Value")
plt.legend()
plt.title("Phase Space Volume and Density of States for 2D Harmonic Oscillator")
plt.grid()
plt.show()

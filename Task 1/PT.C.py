import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def hamiltonian(px, py, x, y, m, omega, lambd):
    """Compute the Hamiltonian H for given values of px, py, x, and y."""
    kinetic = (px**2 + py**2) / (2 * m)
    potential = 0.5 * m * omega**2 * (x**2 + y**2) + lambd * (x**2 + y**2)**2
    return kinetic + potential

def density_of_states(E, m, omega, lambd, num_samples=100000):
    """Estimate the density of states g(E) using Monte Carlo integration."""
    count = 0
    vol = 0
    
    # Define integration limits (adjust for better accuracy)
    p_max = np.sqrt(2 * m * E)  # Approximate momentum cutoff
    x_max = np.sqrt(E / (m * omega**2))  # Approximate spatial cutoff
    
    for _ in range(num_samples):
        px = np.random.uniform(-p_max, p_max)
        py = np.random.uniform(-p_max, p_max)
        x = np.random.uniform(-x_max, x_max)
        y = np.random.uniform(-x_max, x_max)
        
        if hamiltonian(px, py, x, y, m, omega, lambd) <= E:
            count += 1
    
    phase_space_volume = (2 * p_max) * (2 * p_max) * (2 * x_max) * (2 * x_max)
    g_E = count / num_samples * phase_space_volume / (2 * np.pi * np.pi * (hbar**2))
    return g_E

# Constants
m = 1.0  # Mass
omega = 1.0  # Frequency
lambd = 0.1  # Quartic potential coefficient
hbar = 1.0545718e-34  # Planck's reduced constant

# Compute DOS for a range of energies
E_values = np.linspace(0.1, 10, 50)
g_values = [density_of_states(E, m, omega, lambd) for E in E_values]

# Plot the results
plt.plot(E_values, g_values, label="Density of States g(E)")
plt.xlabel("Energy E")
plt.ylabel("g(E)")
plt.title("Density of States of H")
plt.legend()
plt.show()

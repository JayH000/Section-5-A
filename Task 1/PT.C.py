import numpy as np

def density_of_states(m, hbar):
    """
    Compute the density of states g(p) for a two-dimensional system.
    
    Parameters:
        m (float): mass of the particle (in kg).
        hbar (float): reduced Planck's constant (in J·s).
        
    Returns:
        float: Density of states g(p) (in units of 1/J·m²).
    """
    # The formula for g(p) is g(p) = m / (2 * pi * hbar^2)
    g_p = m / (2 * np.pi * hbar**2)
    return g_p

# Constants
m = 1.0  # mass of the particle in kg (adjust as needed)
hbar = 1.0545718e-34  # reduced Planck's constant in J·s

# Calculate the density of states
g_p = density_of_states(m, hbar)

# Print the result
print(f"The density of states g(p) is: {g_p:.2e} 1/J·m²")

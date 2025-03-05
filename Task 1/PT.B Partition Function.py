import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def density_of_states(E, m, omega):
    """
    Calculate the corrected density of states g(E) = 4π²/(m²ω²)
    
    Parameters:
    E (float or array): Energy values
    m (float): Mass parameter
    omega (float): Angular frequency
    
    Returns:
    float or array: The density of states at energy E
    """
    return (4 * np.pi**2) / (m**2 * omega**2)

def integrand(E, beta, m, omega):
    """
    The integrand of the canonical partition function: g(E)exp(-βE)
    
    Parameters:
    E (float): Energy value
    beta (float): Inverse temperature (1/kT)
    m (float): Mass parameter
    omega (float): Angular frequency
    
    Returns:
    float: Value of the integrand at E
    """
    return density_of_states(E, m, omega) * np.exp(-beta * E)

def partition_function(beta, m, omega, E_max=1000):
    """
    Compute the canonical partition function Z(β) = ∫g(E)exp(-βE)dE
    Integration is performed from 0 to E_max
    
    Parameters:
    beta (float): Inverse temperature (1/kT)
    m (float): Mass parameter
    omega (float): Angular frequency
    E_max (float): Upper limit of integration
    
    Returns:
    float: The canonical partition function Z(β)
    """
    result, error = integrate.quad(
        lambda E: integrand(E, beta, m, omega),
        0,  # Lower limit
        E_max  # Upper limit
    )
    return result

def analytical_solution(beta, m, omega):
    """
    Analytical solution for the canonical partition function with g(E) = 4π²/(m²ω²)
    Z(β) = (4π²)/(m²ω²) * (1/β)
    
    Parameters:
    beta (float): Inverse temperature (1/kT)
    m (float): Mass parameter
    omega (float): Angular frequency
    
    Returns:
    float: The analytical solution for Z(β)
    """
    return (4 * np.pi**2) / (m**2 * omega**2 * beta)

def main():
    # Set parameters
    m = 1.0  # Mass
    omega = 1.0  # Angular frequency
    
    # Create a range of beta values (inverse temperatures)
    beta_values = np.linspace(0.1, 5.0, 100)
    
    # Compute Z(β) for each beta using numerical integration
    Z_values = [partition_function(beta, m, omega) for beta in beta_values]
    
    # Compute Z(β) for each beta using the analytical solution
    Z_analytical = [analytical_solution(beta, m, omega) for beta in beta_values]
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(beta_values, Z_values, 'b-', label='Numerical Integration')
    plt.plot(beta_values, Z_analytical, 'r--', label='Analytical Solution')
    plt.xlabel(r'$\beta$ (1/kT)')
    plt.ylabel(r'Partition Function $Z(\beta)$')
    plt.title(r'Canonical Partition Function for g(E) = $4\pi^2/(m^2\omega^2)$')
    plt.legend()
    plt.grid(True)
    
    # Calculate and print some specific values
    test_betas = [0.5, 1.0, 2.0]
    print("\nPartition Function Values:")
    print("--------------------------")
    print(f"{'β':>10} | {'Z(β) Numerical':>15} | {'Z(β) Analytical':>15} | {'Relative Error':>15}")
    print("-" * 60)
    
    for beta in test_betas:
        Z_num = partition_function(beta, m, omega)
        Z_ana = analytical_solution(beta, m, omega)
        rel_error = abs((Z_num - Z_ana) / Z_ana) * 100  # percentage
        print(f"{beta:10.2f} | {Z_num:15.8f} | {Z_ana:15.8f} | {rel_error:15.8f}%")
    
    # Plot thermodynamic quantities
    T_values = 1 / beta_values  # Convert beta to temperature
    
    # Calculate free energy F = -kT ln(Z) = -ln(Z)/beta
    # Using k=1 for simplicity
    F_values = [-np.log(Z) / beta for Z, beta in zip(Z_analytical, beta_values)]
    
    # Calculate internal energy U = -∂ln(Z)/∂β = 1/β
    U_values = [1/beta for beta in beta_values]
    
    # Calculate entropy S = k(ln(Z) + βU) = ln(Z) + 1
    S_values = [np.log(Z) + 1 for Z in Z_analytical]
    
    # Plot thermodynamic quantities
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    axs[0].plot(T_values, F_values, 'g-')
    axs[0].set_xlabel('Temperature (T)')
    axs[0].set_ylabel('Free Energy (F)')
    axs[0].set_title('Free Energy vs Temperature')
    axs[0].grid(True)
    
    axs[1].plot(T_values, U_values, 'm-')
    axs[1].set_xlabel('Temperature (T)')
    axs[1].set_ylabel('Internal Energy (U)')
    axs[1].set_title('Internal Energy vs Temperature')
    axs[1].grid(True)
    
    axs[2].plot(T_values, S_values, 'c-')
    axs[2].set_xlabel('Temperature (T)')
    axs[2].set_ylabel('Entropy (S)')
    axs[2].set_title('Entropy vs Temperature')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def double_pendulum_derivatives(t, state, m1, m2, L1, L2, g):
    theta1, theta2, p1, p2 = state
    
    # Compute derivatives
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)
    
    theta1_dot = (L2 * p1 - L1 * p2 * c) / (L1**2 * L2 * (m1 + m2 * s**2))
    theta2_dot = (L1 * (m1 + m2) * p2 - L2 * m2 * p1 * c) / (L1 * L2**2 * m2 * (m1 + m2 * s**2))
    
    p1_dot = -(m1 + m2) * g * L1 * np.sin(theta1) - s * (L1 * theta1_dot * p2 - L2 * theta2_dot * p1)
    p2_dot = -m2 * g * L2 * np.sin(theta2) + s * (L1 * theta1_dot * p2 - L2 * theta2_dot * p1)
    
    return [theta1_dot, theta2_dot, p1_dot, p2_dot]

def simulate_double_pendulum(t_max=10, dt=0.01):
    # Define parameters
    m1, m2 = 1.0, 1.0
    L1, L2 = 1.0, 1.0
    g = 9.81
    
    # Initial conditions (theta1, theta2, p1, p2)
    theta1_0, theta2_0 = np.pi / 2, np.pi / 4
    p1_0, p2_0 = 0.0, 0.0
    
    state0 = [theta1_0, theta2_0, p1_0, p2_0]
    t_eval = np.arange(0, t_max, dt)
    
    sol = solve_ivp(double_pendulum_derivatives, [0, t_max], state0, t_eval=t_eval, args=(m1, m2, L1, L2, g))
    
    return sol.t, sol.y

def plot_phase_space_and_trajectory():
    t, sol = simulate_double_pendulum()
    theta2, p2 = sol[1], sol[3]
    
    # Phase space plot (θ2, p2)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(theta2, p2, label='Phase Space (θ2, p2)')
    plt.xlabel("θ2")
    plt.ylabel("p2")
    plt.legend()
    plt.title("Phase Space Trajectory")
    
    # Convert to Cartesian coordinates
    theta1, theta2 = sol[0], sol[1]
    L1, L2 = 1.0, 1.0
    x1, y1 = L1 * np.sin(theta1), -L1 * np.cos(theta1)
    x2, y2 = x1 + L2 * np.sin(theta2), y1 - L2 * np.cos(theta2)
    
    plt.subplot(1, 2, 2)
    plt.plot(x2, y2, label='Real Space (x2, y2)')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Double Pendulum Trajectory")
    
    plt.show()

# Run the visualization
plot_phase_space_and_trajectory()

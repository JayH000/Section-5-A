import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull

def double_pendulum_derivatives(t, state, m1, m2, L1, L2, g):
    theta1, theta2, p1, p2 = state
    
    # Compute derivatives
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)
    
    theta1_dot = (L2 * p1 - L1 * p2 * c) / (L1**2 * L2 * (m1 + m2 * s**2))
    theta2_dot = (L1 * (m1 + m2) * p2 - L2 * m2 * p1 * c) / (L1 * L2**2 * m2 * (m1 + m2 * s**2))
    
    p1_dot = -(m1 + m2) * g * L1 * np.sin(theta1) - s * (L1 * theta1_dot * p2 - L2 * theta2_dot * p1)
    p2_dot = -m2 * g * L2 * np.sin(theta2) + s * (L1 * theta1_dot * p2 - L2 * theta2_dot * p1)
    
    return [theta1_dot, theta2_dot, p1_dot, p2_dot]

def simulate_double_pendulum(state0, t_max=10, dt=0.01):
    # Define parameters
    m1, m2 = 1.0, 1.0
    L1, L2 = 1.0, 1.0
    g = 9.81
    
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(double_pendulum_derivatives, [0, t_max], state0, t_eval=t_eval, args=(m1, m2, L1, L2, g))
    
    return sol.t, sol.y

def initialize_point_cloud(n_points=100, spread=0.01):
    theta1_0, theta2_0 = np.pi / 2, np.pi / 4
    p1_0, p2_0 = 0.0, 0.0
    
    # Generate small perturbations around initial condition
    perturbations = spread * np.random.randn(n_points, 4)
    point_cloud = np.array([theta1_0, theta2_0, p1_0, p2_0]) + perturbations
    
    return point_cloud

def compute_phase_space_volume(theta2, p2):
    points = np.column_stack((theta2, p2))
    hull = ConvexHull(points)
    return hull.volume

def plot_phase_space_evolution():
    point_cloud = initialize_point_cloud()
    t_max, dt = 10, 0.05
    t_eval = np.arange(0, t_max, dt)

    volumes = []
    for t_step in t_eval:
        theta2_samples = []
        p2_samples = []
    
    for i in range(len(point_cloud)):
        t, sol = simulate_double_pendulum(point_cloud[i], t_max, dt)
        theta2, p2 = sol[1], sol[3]
        volumes.append(compute_phase_space_volume(theta2, p2))
    
    plt.figure()
    plt.plot(t_eval, volumes, label="Phase Space Volume")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.legend()
    plt.title("Phase Space Volume Evolution")
    plt.show()

# Run the visualization
plot_phase_space_evolution()

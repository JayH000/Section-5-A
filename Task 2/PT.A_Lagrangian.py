import numpy as np
import sympy as sp

def derive_lagrangian():

    # Define symbols
    m1, m2, L1, L2, g = sp.symbols('m1 m2 L1 L2 g')
    theta1, theta2 = sp.symbols('theta1 theta2', real=True)
    dtheta1, dtheta2 = sp.symbols('dtheta1 dtheta2', real=True)
    ddtheta1, ddtheta2 = sp.symbols('ddtheta1 ddtheta2', real=True)
    
    # Position coordinates
    x1 = L1 * sp.sin(theta1)
    y1 = -L1 * sp.cos(theta1)
    x2 = x1 + L2 * sp.sin(theta2)
    y2 = y1 - L2 * sp.cos(theta2)
    
    # Velocities
    v1_sq = L1**2 * dtheta1**2
    v2_sq = L1**2 * dtheta1**2 + L2**2 * dtheta2**2 + 2 * L1 * L2 * dtheta1 * dtheta2 * sp.cos(theta1 - theta2)
    
    # Kinetic and potential energies
    T = (1/2) * m1 * v1_sq + (1/2) * m2 * v2_sq
    V = m1 * g * y1 + m2 * g * y2
    
    # Lagrangian
    L = T - V
    
    # Euler-Lagrange equations
    dL_dtheta1 = sp.diff(L, theta1)
    dL_dtheta2 = sp.diff(L, theta2)
    dL_ddtheta1 = sp.diff(L, dtheta1)
    dL_ddtheta2 = sp.diff(L, dtheta2)
    
    d_dt_dL_ddtheta1 = sp.diff(dL_ddtheta1, theta1) * dtheta1 + sp.diff(dL_ddtheta1, theta2) * dtheta2 + sp.diff(dL_ddtheta1, dtheta1) * ddtheta1 + sp.diff(dL_ddtheta1, dtheta2) * ddtheta2
    d_dt_dL_ddtheta2 = sp.diff(dL_ddtheta2, theta1) * dtheta1 + sp.diff(dL_ddtheta2, theta2) * dtheta2 + sp.diff(dL_ddtheta2, dtheta1) * ddtheta1 + sp.diff(dL_ddtheta2, dtheta2) * ddtheta2
    
    # Equations of motion
    eq1 = sp.simplify(d_dt_dL_ddtheta1 - dL_dtheta1)
    eq2 = sp.simplify(d_dt_dL_ddtheta2 - dL_dtheta2)
    
    # Mass matrix and force vector
    mass_matrix = sp.Matrix([[sp.diff(eq1, ddtheta1), sp.diff(eq1, ddtheta2)],
                              [sp.diff(eq2, ddtheta1), sp.diff(eq2, ddtheta2)]]).simplify()
    force_vector = sp.Matrix([eq1, eq2]).simplify()
    
    return mass_matrix, force_vector

print("Equation 1:", eq1)
print("Equation 2:", eq2)
# Compute mass matrix and force vector
mass_matrix = sp.Matrix([[sp.diff(eq1, ddtheta1), sp.diff(eq1, ddtheta2)],
                              [sp.diff(eq2, ddtheta1), sp.diff(eq2, ddtheta2)]])
force_vector = sp.Matrix([eq1, eq2])
    
print("Mass Matrix:", mass_matrix)
print("Force Vector:", force_vector)
return mass_matrix, force_vector

if __name__ == "__main__":
    try:
        import sympy
    except ImportError:
        print("SymPy is not installed. Please install it using 'pip install sympy'.")
        exit(1)

# Compute results
M, C = derive_lagrangian()
print("Mass Matrix M:")
sp.pprint(M)
print("\nForce Vector C:")
sp.pprint(C)

import sympy as sp

def compute_hamiltonian(L):
    # Define symbols
    q, dq, p, t = sp.symbols('q dq p t')
    
    # Compute generalized momentum
    p_expr = sp.diff(L, dq)
    print("Generalized Momentum p:")
    sp.pprint(p_expr)
    
    # Solve for dq in terms of p (invert equation)
    dq_solved = sp.solve(p - p_expr, dq)
    print("Solutions for dq:", dq_solved)

    if dq_solved:
        dq_expr = dq_solved[0]  # Take first solution
    else:
        raise ValueError("Could not solve for dq in terms of p.")
    

    # Compute Hamiltonian
    H = p * dq_expr - L.subs(dq, dq_expr)
    H = sp.simplify(H)
    
    return H

# Example Lagrangian 
m, k = sp.symbols('m k')
L = (1/2) * m * sp.Symbol('dq')**2 - (1/2) * k * sp.Symbol('q')**2

# Compute Hamiltonian
H = compute_hamiltonian(L)
print("Hamiltonian H:")
sp.pprint(H)
"""
Printed result:

Generalized Momentum p:

1.0⋅dq⋅m

Solutions for dq: [p/m]

Hamiltonian H:
                 2
       2   0.5⋅p 
0.5⋅k⋅q  + ──────
             m   
"""
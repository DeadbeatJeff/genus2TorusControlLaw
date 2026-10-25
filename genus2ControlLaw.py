import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.linalg import solve_continuous_are

# --- 1. Global Parameters for the HJB Controller ---
# Q: Penalty for state error [theta1, theta2, p1, p2]
Q = np.diag([10.0, 10.0, 1.0, 1.0]) 
# R_effort: Penalty for motor torque u
R_effort = np.diag([0.1, 0.1])

def dh_htm(theta1, theta2, r, d):
    """Computes the standard DH Homogeneous Transformation Matrix."""
    return sp.Matrix([
        [sp.cos(theta1), -sp.sin(theta1)*sp.cos(theta2),  sp.sin(theta1)*sp.sin(theta2), r*sp.cos(theta1)],
        [sp.sin(theta1),  sp.cos(theta1)*sp.cos(theta2), -sp.cos(theta1)*sp.sin(theta2), r*sp.sin(theta1)],
        [0,              sp.sin(theta2),               sp.cos(theta2),              d],
        [0,              0,                           0,                          1]
    ])

def compute_riemann(Gamma2nd, Theta, n):
    """Computes the contravariant Riemann Curvature Tensor R^i_{jkl}."""
    Riemann = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    # Derivative terms: diff(Gamma_ijl, Theta_k) - diff(Gamma_ijk, Theta_l)
                    term1 = sp.diff(Gamma2nd[i, j, l], Theta[k])
                    term2 = sp.diff(Gamma2nd[i, j, k], Theta[l])
                    
                    # Summation term for indices p=1 to n
                    sum_term = 0
                    for p in range(n):
                        sum_term += (Gamma2nd[i, k, p] * Gamma2nd[p, j, l] - 
                                     Gamma2nd[i, l, p] * Gamma2nd[p, j, k])
                    
                    Riemann[i, j, k, l] = term1 - term2 + sum_term
    return Riemann

def get_hjb_optimal_control(state, target_q, M_func):
    """
    Computes u* by solving a local Linear Quadratic Regulator (LQR).
    """
    q = state[:2]
    p = state[2:]
    
    # Current Metric and its Inverse
    g = M_func(q[1])
    g_inv = np.linalg.inv(g)
    
    # Linearize dynamics locally: x_dot = Ax + Bu
    A = np.zeros((4, 4))
    A[:2, 2:] = g_inv # dq/dt = g_inv * p
    
    B = np.zeros((4, 2))
    B[2:, :] = np.eye(2) # Torque maps to momentum change
    
    try:
        # Solve the Algebraic Riccati Equation for the current state
        P = solve_continuous_are(A, B, Q, R_effort)
        K = np.linalg.inv(R_effort) @ B.T @ P
        
        # Compute error (Target momentum is 0)
        error = np.concatenate([q - target_q, p])
        u_star = -K @ error
    except Exception:
        u_star = np.array([0.0, 0.0])
        
    return u_star

def optimal_dynamics(t, state, target_q, Q, R_effort, M_func, get_inv_metric):
    """
    Fixed dynamics that accepts all required positional arguments 
    via the 'args' tuple in solve_ivp.
    """
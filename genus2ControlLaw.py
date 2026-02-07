import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

# --- 1. Global Parameters for the HJB Controller ---
Q_mat = np.diag([10.0, 10.0, 1.0, 1.0]) 
R_effort = np.diag([0.1, 0.1])

def compute_riemann(Gamma2nd, Theta, n):
    Riemann = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    term1 = sp.diff(Gamma2nd[i, j, l], Theta[k])
                    term2 = sp.diff(Gamma2nd[i, j, k], Theta[l])
                    sum_term = 0
                    for p in range(n):
                        sum_term += (Gamma2nd[i, k, p] * Gamma2nd[p, j, l] - 
                                     Gamma2nd[i, l, p] * Gamma2nd[p, j, k])
                    Riemann[i, j, k, l] = term1 - term2 + sum_term
    return Riemann

def optimal_dynamics(t, state, target_q, Q, R_eff, M_func, get_inv_metric):
    theta1, theta2, p1, p2 = state
    q = state[:2]
    p = state[2:]
    
    g_inv = get_inv_metric(theta2)
    A = np.zeros((4, 4))
    A[:2, 2:] = g_inv 
    B = np.zeros((4, 2))
    B[2:, :] = np.eye(2)
    
    try:
        P = solve_continuous_are(A, B, Q, R_eff)
        K = np.linalg.inv(R_eff) @ B.T @ P
        error = np.concatenate([q - target_q, p])
        u = -K @ error
    except:
        u = np.array([0.0, 0.0])

    q_dot = g_inv @ p
    eps = 1e-6
    h_plus = 0.5 * p @ get_inv_metric(theta2 + eps) @ p
    h_minus = 0.5 * p @ get_inv_metric(theta2 - eps) @ p
    dp2 = -(h_plus - h_minus) / (2 * eps)
    
    return [q_dot[0], q_dot[1], u[0], dp2 + u[1]]

if __name__ == "__main__":
    # --- 2. Define Variables ---
    q = sp.Matrix([sp.symbols('theta1'), sp.symbols('theta4')])
    p_vars = sp.Matrix([sp.symbols('theta2'), sp.symbols('theta3')]) # Passive
    q_dot = sp.Matrix([sp.symbols('thetaDot1'), sp.symbols('thetaDot4')])
    
    Theta = q # Define Theta for differentiation in Christoffel/Riemann

    L = sp.symbols('L1:6')
    m = sp.symbols('m1:5')
    Izz = sp.symbols('Izz1:5')
    g_sym = sp.symbols('g')
    n_joints = 2

    # --- 3. Loop Closure & Jacobian ---
    f1 = L[0]*sp.cos(q[0]) + L[1]*sp.cos(p_vars[0]) - (L[4] + L[3]*sp.cos(q[1]) + L[2]*sp.cos(p_vars[1]))
    f2 = L[0]*sp.sin(q[0]) + L[1]*sp.sin(p_vars[0]) - (L[3]*sp.sin(q[1]) + L[2]*sp.sin(p_vars[1]))
    Phi = sp.Matrix([f1, f2])

    J_p = Phi.jacobian(p_vars)
    J_q = Phi.jacobian(q)
    J_vel = -J_p.inv() @ J_q
    p_dot = J_vel @ q_dot 

    # --- 4. Link Velocities & Kinetic Energy ---
    v_sq = []
    v_sq.append((L[0]/2 * q_dot[0])**2) # Link 1
    
    vB = sp.Matrix([-L[0]*sp.sin(q[0])*q_dot[0], L[0]*sp.cos(q[0])*q_dot[0]])
    v2 = vB + sp.Matrix([-L[1]/2*sp.sin(p_vars[0])*p_dot[0], L[1]/2*sp.cos(p_vars[0])*p_dot[0]])
    v_sq.append(v2.dot(v2)) # Link 2
    
    vD = sp.Matrix([-L[3]*sp.sin(q[1])*q_dot[1], L[3]*sp.cos(q[1])*q_dot[1]])
    v3 = vD + sp.Matrix([-L[2]/2*sp.sin(p_vars[1])*p_dot[1], L[2]/2*sp.cos(p_vars[1])*p_dot[1]])
    v_sq.append(v3.dot(v3)) # Link 3
    
    v_sq.append((L[3]/2 * q_dot[1])**2) # Link 4

    KE = 0
    all_omg_dots = [q_dot[0], p_dot[0], p_dot[1], q_dot[1]]
    for i in range(4):
        KE += 0.5 * m[i] * v_sq[i] + 0.5 * Izz[i] * all_omg_dots[i]**2

    # --- 5. Mass Matrix (Riemannian Metric) ---
    M = sp.Matrix([
        [sp.diff(KE, q_dot[0], q_dot[0]), sp.diff(KE, q_dot[0], q_dot[1])],
        [sp.diff(KE, q_dot[1], q_dot[0]), sp.diff(KE, q_dot[1], q_dot[1])]
    ])
    
    rob_values = {
        L[0]: 4.0, L[1]: 30.0, L[2]: 30.0, L[3]: 4.0, L[4]: 10.0,
        m[0]: 0.1, m[1]: 0.5, m[2]: 0.5, m[3]: 0.1,
        Izz[0]: 0.001, Izz[1]: 0.001, Izz[2]: 0.001, Izz[3]: 0.001,
        g_sym: 9.81
    }
    print("Mass Matrix M (Symbolic):")
    sp.pprint(M)
    
    # Substitute values and simplify
    # Note: theta2 in the HJB section refers to the second control variable (theta4)
    M_numeric_sym = M.subs(rob_values).subs({p_vars[0]: 0, p_vars[1]: 0}) # Simplified for example
    M_simplified = sp.simplify(M.subs(rob_values))

    # --- 6. Optimized Christoffel Symbols ---
    M_inv = M_simplified.inv()
    
    # Pre-compute derivatives of the metric matrix
    # dM[k] = partial M / partial Theta[k]
    dM = [M_simplified.diff(Theta[k]) for k in range(n_joints)]

    Gamma2nd = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints)
    for i in range(n_joints):
        for j in range(n_joints):
            for k in range(n_joints):
                # Using the standard formula: 1/2 * g^il * (dg_lj/dxk + dg_lk/dxj - dg_jk/dxl)
                val = 0
                for l in range(n_joints):
                    term = 0.5 * M_inv[i, l] * (dM[k][l, j] + dM[j][l, k] - dM[l][j, k])
                    val += term
                Gamma2nd[i, j, k] = val # No simplify here!

    print("Christoffel Symbols (2nd Kind) Computed.")

    # --- 7. Optimized Riemann & Curvature ---
    RiemannContra = compute_riemann(Gamma2nd, Theta, n_joints)
    
    # Directly calculate R_0101 to avoid the full 4D tensor if only K is needed
    # R_{0101} = g_{0m} * R^m_{101}
    R0101 = 0
    for m in range(n_joints):
        R0101 += M_simplified[0, m] * RiemannContra[m, 1, 0, 1]

    # Final simplification happens only ONCE here
    K_sym = sp.simplify(R0101 / M_simplified.det())

    print("\nGaussian Curvature (K) Symbolic Expression:")
    sp.pprint(K_sym)

    # ... [Keep Sections 1 through 7 as they are] ...

    # --- 8. Geodesic Control Law (Natural Geometry Path) ---
    def geodesic_dynamics(t, state, get_inv_metric):
        """
        Hamiltonian equations for the Geodesic (No-Cost Path):
        dq/dt = g^-1 * p
        dp/dt = -dH/dq
        """
        theta1, theta2, p1, p2 = state
        p = np.array([p1, p2])
        
        g_inv = get_inv_metric(theta2)
        q_dot = g_inv @ p
        
        # Numerical gradient of the kinetic energy (Hamiltonian) w.r.t theta4 (theta2 here)
        eps = 1e-6
        h_plus = 0.5 * p @ get_inv_metric(theta2 + eps) @ p
        h_minus = 0.5 * p @ get_inv_metric(theta2 - eps) @ p
        dp2 = -(h_plus - h_minus) / (2 * eps)
        
        # dp1 is 0 because the metric M doesn't depend on theta1 (axisymmetric)
        return [q_dot[0], q_dot[1], 0, dp2]

    # Initial momentum for the geodesic "push"
    geodesic_p0 = [0.01, 0.005] 
    y0_geodesic = [0, 0, geodesic_p0[0], geodesic_p0[1]]
    
    sol_geodesic = solve_ivp(
        geodesic_dynamics, 
        (0, 10), 
        y0_geodesic, 
        args=(get_inverse_metric_numeric,),
        t_eval=np.linspace(0, 10, 500)
    )
    print("Geodesic Control Law (Dido move) computed.")

    # --- 9. HJB Optimal Control Simulation ---
    target_conf = np.array([np.pi, 0.0])
    y0_hjb = [0, 0.1, 0, 0]

    sol_hjb = solve_ivp(
        optimal_dynamics, 
        (0, 20), 
        y0_hjb, 
        args=(target_conf, Q_mat, R_effort, M_func_numeric, get_inverse_metric_numeric),
        t_eval=np.linspace(0, 20, 1000)
    )

    # --- 10. Visualization ---
    fig, axs = plt.subplots(1, 4, figsize=(22, 5))

    # Plot 1: Gaussian Curvature K
    theta_vals = np.linspace(0, 2 * np.pi, 100)
    K_vals = K_func(theta_vals)
    axs[2].plot(theta_vals, K_vals, color='blue', lw=2)
    axs[2].set_title("Gaussian Curvature $K$")
    axs[2].set_xlabel(r"$\theta_4$")
    axs[2].grid(True, linestyle='--')

    # Plot 2: Geodesic Path (The "Natural" Flow)
    axs[0].plot(sol_geodesic.y[0] % (2*np.pi), sol_geodesic.y[1] % (2*np.pi), color='red', lw=2)
    axs[0].set_title("Geodesic Path (Natural Flow)")
    axs[0].set_xlabel(r'$\theta_1$')
    axs[0].set_ylabel(r'$\theta_4$')
    axs[0].grid(True)

    # Plot 3: HJB State Convergence
    axs[1].plot(sol_hjb.t, sol_hjb.y[0], label=r'$\theta_1$')
    axs[1].plot(sol_hjb.t, sol_hjb.y[1], label=r'$\theta_4$')
    axs[1].axhline(target_conf[0], color='r', linestyle='--', alpha=0.5)
    axs[1].set_title("HJB State Convergence")
    axs[1].legend()

    # Plot 4: HJB Path on C-Space
    axs[3].plot(sol_hjb.y[0] % (2*np.pi), sol_hjb.y[1] % (2*np.pi), color='green')
    axs[3].set_xlabel(r'$\theta_1$')
    axs[3].set_ylabel(r'$\theta_4$')
    axs[3].set_title("Optimal HJB Path")

    plt.tight_layout()
    plt.show()
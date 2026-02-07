import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

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

# Define the Riemann computation function before use
def compute_riemann(Gamma2nd, Theta, n):
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
    theta1, theta2, p1, p2 = state
    q = state[:2]
    p = state[2:]
    
    # 1. Local Linearization for LQR (HJB Approximation)
    g_inv = get_inv_metric(theta2)
    
    A = np.zeros((4, 4))
    A[:2, 2:] = g_inv 
    B = np.zeros((4, 2))
    B[2:, :] = np.eye(2)
    
    # Solve Riccati Equation for Optimal Control Effort u
    try:
        P = solve_continuous_are(A, B, Q, R_effort)
        K = np.linalg.inv(R_effort) @ B.T @ P
        error = np.concatenate([q - target_q, p]) # Target p is 0
        u = -K @ error
    except:
        u = np.array([0.0, 0.0])

    # 2. Hamiltonian Geodesic Dynamics
    q_dot = g_inv @ p
    
    eps = 1e-6
    h_plus = 0.5 * p @ get_inv_metric(theta2 + eps) @ p
    h_minus = 0.5 * p @ get_inv_metric(theta2 - eps) @ p
    dp2 = -(h_plus - h_minus) / (2 * eps)
    
    # Return [dq1, dq2, dp1 + u1, dp2 + u2]
    return [q_dot[0], q_dot[1], u[0], dp2 + u[1]]

if __name__ == "__main__":
    # --- 1. Define Variables ---
    # Active Motors (q) and Passive Joints (p)
    q = sp.Matrix([sp.symbols('theta1'), sp.symbols('theta4')])
    p = sp.Matrix([sp.symbols('theta2'), sp.symbols('theta3')])
    q_dot = sp.Matrix([sp.symbols('thetaDot1'), sp.symbols('thetaDot4')])

    L = sp.symbols('L1:6')
    m = sp.symbols('m1:5')
    Izz = sp.symbols('Izz1:5')
    g_sym = sp.symbols('g')
    n_joints = 2 # Crucial: The system has 2 Degrees of Freedom (DoF)

    # --- 2. Loop Closure & Jacobian ---
    # f = [xC_left - xC_right, yC_left - yC_right] = 0
    f1 = L[0]*sp.cos(q[0]) + L[1]*sp.cos(p[0]) - (L[4] + L[3]*sp.cos(q[1]) + L[2]*sp.cos(p[1]))
    f2 = L[0]*sp.sin(q[0]) + L[1]*sp.sin(p[0]) - (L[3]*sp.sin(q[1]) + L[2]*sp.sin(p[1]))
    Phi = sp.Matrix([f1, f2])

    # Implicit Differentiation: J_p * p_dot + J_q * q_dot = 0
    J_p = Phi.jacobian(p)
    J_q = Phi.jacobian(q)
    J_vel = -J_p.inv() @ J_q
    p_dot = J_vel @ q_dot # Maps motor speeds to passive link speeds

    # --- 3. Link Velocities (CoM midpoints) ---
    v_sq = []
    # Link 1 & 4 (Cranks)
    v_sq.append((L[0]/2 * q_dot[0])**2) # v1
    # Link 2 (Coupler 1): joint B velocity + link 2 relative velocity
    vB = sp.Matrix([-L[0]*sp.sin(q[0])*q_dot[0], L[0]*sp.cos(q[0])*q_dot[0]])
    v2 = vB + sp.Matrix([-L[1]/2*sp.sin(p[0])*p_dot[0], L[1]/2*sp.cos(p[0])*p_dot[0]])
    v_sq.append(v2.dot(v2)) # v2
    # Link 3 (Coupler 2): joint D velocity + link 3 relative velocity
    vD = sp.Matrix([-L[3]*sp.sin(q[1])*q_dot[1], L[3]*sp.cos(q[1])*q_dot[1]])
    v3 = vD + sp.Matrix([-L[2]/2*sp.sin(p[1])*p_dot[1], L[2]/2*sp.cos(p[1])*p_dot[1]])
    v_sq.append(v3.dot(v3)) # v3
    v_sq.append((L[3]/2 * q_dot[1])**2) # v4

    # --- 4. Total Kinetic Energy & Metric ---
    KE = 0
    all_omg_dots = [q_dot[0], p_dot[0], p_dot[1], q_dot[1]]
    for i in range(4):
        KE += 0.5 * m[i] * v_sq[i] + 0.5 * Izz[i] * all_omg_dots[i]**2

    # Extract the 2x2 Mass Matrix M
    M = sp.Matrix([
        [sp.diff(KE, q_dot[0], q_dot[0]), sp.diff(KE, q_dot[0], q_dot[1])],
        [sp.diff(KE, q_dot[1], q_dot[0]), sp.diff(KE, q_dot[1], q_dot[1])]
    ])
    M_simplified = sp.simplify(M)

    # 5. Lagrangian and Mass Matrix
    KE = 0
    all_sq_vels = [v1_sq, v2_sq, v3_sq, v4_sq]
    all_omg_dots = [q_dot[0], p_dot[0], p_dot[1], q_dot[1]]

    for i in range(4):
        KE += 0.5 * m[i] * all_sq_vels[i] + 0.5 * Izz[i] * all_omg_dots[i]**2

    # Extract 2x2 Riemannian Metric (Mass Matrix)
    M = sp.Matrix([
        [sp.diff(KE, q_dot[0], q_dot[0]), sp.diff(KE, q_dot[0], q_dot[1])],
        [sp.diff(KE, q_dot[1], q_dot[0]), sp.diff(KE, q_dot[1], q_dot[1])]
    ])

    M_simplified = sp.simplify(M)

    # 6. Energy Calculations
    KE = 0
    V_potential = 0
    for i in range(4):
        # KE = 1/2 m v^2 + 1/2 I omega^2
        KE += 0.5 * m[i] * v_sq[i] + 0.5 * Izz[i] * all_theta_dot[i]**2
        V_potential += m[i] * g * y_com[i]

    Lagrangian = sp.simplify(KE - V_potential)

    # 7. Extract 2x2 Mass Matrix M(q)
    # M_ij = d^2(KE) / (d(q_dot_i) d(q_dot_j))
    M = sp.Matrix([
        [sp.diff(KE, q_dot[0], q_dot[0]), sp.diff(KE, q_dot[0], q_dot[1])],
        [sp.diff(KE, q_dot[1], q_dot[0]), sp.diff(KE, q_dot[1], q_dot[1])]
    ])

    M = sp.simplify(M)

    # 7. Create a dictionary of numerical values
    # This replaces the MATLAB loop for assigning constants
    rob_values = {}

    for i in range(n_joints):
        # Map the symbolic variable to the numeric value
        # rob_values[m[i]] = 0.181
        # Updated for 5-bar Grashof "Double Crank" behavior
        # Updated for 5-bar where l1 and l4 are cranks
        rob_values = {
            L[0]: 4.0,   # l1 (Crank 1)
            L[1]: 30.0,  # l2 (Coupler 1)
            L[2]: 30.0,  # l3 (Coupler 2)
            L[3]: 4.0,   # l4 (Crank 2)
            L[4]: 10.0,  # l5 (Base)
            m[0]: 0.1, m[1]: 0.5, m[2]: 0.5, m[3]: 0.1, # Masses
            g: 9.81
}
        rob_values[Izz[i]] = 0.001
        # If you included the full inertia tensor in your EoM:
        # rob_values[Ixx[i]] = 0.093805
        # rob_values[Iyy[i]] = 0.001

    # 8. Add gravity if needed
    # rob_values[g] = 9.81

    # 9. Extract Mass Matrix (MJeff)
    # The Mass Matrix M is the coefficient of doubleThetaDot
    MJeff = sp.Matrix.zeros(n_joints, n_joints)
    for i in range(n_joints):
        for j in range(n_joints):
            # Collect coefficients of the acceleration terms
            MJeff[i, j] = sp.simplify(sp.diff(RHS[i], doubleThetaDot[j]))

    # 10. Substitute values into your Mass Matrix (MJeff) or Lagrangian
    # This creates a version of the matrix that only depends on Theta
    M_numeric_sym = MJeff.subs(rob_values)

    print("Mass matrix (Riemannian metric) computed:")
    sp.pprint(sp.simplify(M_numeric_sym).evalf(3))

    # 11. Compute the Inverse Mass Matrix (MJeffinv)
    # MJeff is the Riemannian metric tensor g_ij
    MJeffinv = MJeff.inv()

    # 12. Compute Christoffel Symbols of the First Kind (Gamma1st)
    # Gamma_ijk = 0.5 * ( d/dq_k(g_ij) + d/dq_j(g_ik) - d/dq_i(g_jk) )
    Gamma1st = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints)

    for i in range(n_joints):
        for j in range(n_joints):
            for k in range(n_joints):
                term = 0.5 * (
                    sp.diff(MJeff[i, j], Theta[k]) + 
                    sp.diff(MJeff[i, k], Theta[j]) - 
                    sp.diff(MJeff[j, k], Theta[i])
                )
                Gamma1st[i, j, k] = sp.simplify(term)

    # 13. Compute Christoffel Symbols of the Second Kind (Gamma2nd)
    # Gamma^i_jk = sum_l ( g^il * Gamma_ljk )
    Gamma2nd = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints)

    for i in range(n_joints):
        for j in range(n_joints):
            for k in range(n_joints):
                gamma_val = 0
                for l in range(n_joints):
                    gamma_val += MJeffinv[i, l] * Gamma1st[l, j, k]
                Gamma2nd[i, j, k] = sp.simplify(gamma_val)

    print("Christoffel symbols of the second kind computed.")

    # Note: The original MATLAB script calculates Christoffel symbols (Gamma2nd)
    # and then the Riemann tensor.
    n = n_joints
    RiemannContra = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    RiemannContra = compute_riemann(Gamma2nd, Theta, n)

    print("Contravariant Riemann Tensor structure computed.")

    # 14. Compute RiemannCovar (Fully Covariant Riemann Tensor)
    # R_{lijk} = sum_m ( g_{lm} * R^m_{ijk} )
    RiemannCovar = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints, n_joints)

    for i in range(n_joints):
        for j in range(n_joints):
            for k in range(n_joints):
                for l in range(n_joints):
                    sum_term = 0
                    for m_idx in range(n_joints):
                        # RiemannContra was computed in the previous step
                        sum_term += MJeff[l, m_idx] * RiemannContra[m_idx, i, j, k]
                    RiemannCovar[l, i, j, k] = sp.simplify(sum_term)

    print("Fully Covariant Riemann Tensor structure computed.")

    # 15. Compute Gaussian Curvature (K)
    # For a 2D surface, K = R_{1212} / det(g)
    det_g = M_numeric_sym.det()
    # Note: indices are 0-based in Python (1,2,1,2 becomes 0,1,0,1)
    K_sym = RiemannCovar[0, 1, 0, 1] / det_g
    K_sym = sp.simplify(K_sym)

    print("Gaussian Curvature (K) computed:")
    sp.pprint(K_sym.evalf(3))

    # 16. Plot K over the range [0, 2*pi]
    # Since K only depends on Theta2 in this model:
    K_func = sp.lambdify(Theta[1], K_sym.subs(rob_values), "numpy")

    theta2_vals = np.linspace(0, 2 * np.pi, 100)
    K_vals = K_func(theta2_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(theta2_vals, K_vals, lw=2, color='blue')
    plt.title("Gaussian Curvature $K$ of Robot C-Space vs. $\Theta_2$")
    plt.xlabel("$\Theta_2$ (Joint 2 Angle in Radians)")
    plt.ylabel("Curvature $K$")
    plt.grid(True, linestyle='--')
    plt.axhline(0, color='black', lw=1)
    plt.show()

    # --- 17. Prepare the Control Law (Hamiltonian Dynamics) ---

    # Convert the symbolic Mass Matrix into a fast numerical function
    # We use the version with numerical rob_values substituted
   # --- 17. Prepare the Numerical Functions ---
    M_func_numeric = sp.lambdify(Theta[1], M_numeric_sym, "numpy")

    def get_inverse_metric_numeric(theta2):
        """Retrieves the numerical g^-1 at a specific theta2."""
        # Ensure theta2 is a float for linalg
        g_num = M_func_numeric(float(theta2))
        return np.linalg.inv(g_num)

    def control_law_dynamics(t, state):
        """
        Hamiltonian equations for the Control Law:
        dq/dt = dH/dp
        dp/dt = -dH/dq
        """
        # state = [theta1, theta2, p1, p2]
        theta1, theta2, p1, p2 = state
        p = np.array([p1, p2])
        
        # 1. Compute velocity: q_dot = g^-1 * p
        g_inv = get_inverse_metric_numeric(theta2)
        q_dot = g_inv @ p
        
        # 2. Compute momentum change: dp/dt = -dH/dq
        # Since the metric only depends on theta2, p1 is conserved (dp1 = 0)
        dp1 = 0
        
        # Numerical gradient of the Hamiltonian w.r.t theta2
        eps = 1e-6
        h_plus = 0.5 * p @ get_inverse_metric_numeric(theta2 + eps) @ p
        h_minus = 0.5 * p @ get_inverse_metric_numeric(theta2 - eps) @ p
        dp2 = -(h_plus - h_minus) / (2 * eps)
        
        return [q_dot[0], q_dot[1], dp1, dp2]

    # --- 18. Simulate a Geodesic Path (The "Dido" Move) ---

    # Initial state: Starting at [0,0] with initial 'push' (momenta)
    # Varying p1 and p2 changes the "area" or displacement goal
    initial_momenta = [0.01, 0.005]
    y0 = [0, 0, initial_momenta[0], initial_momenta[1]]
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 500)

    sol = solve_ivp(control_law_dynamics, t_span, y0, t_eval=t_eval)
    print("No-Cost-Function Control Law B[u(q, \dot{q})] computed.")

    # --- 19. Plot the Control Law Path on the Torus ---

    plt.figure(figsize=(8, 6))
    # We use modulo 2*pi to wrap the path around the torus
    plt.plot(sol.y[0] % (2*np.pi), sol.y[1] % (2*np.pi), color='red', lw=2)
    plt.xlabel('$\Theta_1$ (rad)')
    plt.ylabel('$\Theta_2$ (rad)')
    plt.title('Optimal Control Law Path (Geodesic) on the C-Space Torus')
    plt.grid(True)
    plt.show()

    # --- 20. Execution: Reach the Target on the Torus ---
    target_configuration = np.array([np.pi, 0.0]) # Goal: Link 1 flipped, Link 2 straight
    y0_optimal = [0, 0.1, 0, 0] # Start near origin with zero momentum

    sol_hjb = solve_ivp(
        optimal_dynamics, 
        (0, 20), 
        y0_optimal, 
        args=(target_configuration, Q, R_effort, M_func_numeric, get_inverse_metric_numeric),
        t_eval=np.linspace(0, 20, 1000)
    )
    
    print("Optimal Control Law B[u(q, \dot{q})] computed.")

    # --- 21. Visualization ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sol_hjb.t, sol_hjb.y[0], label='Theta 1')
    plt.plot(sol_hjb.t, sol_hjb.y[1], label='Theta 2')
    plt.axhline(target_configuration[0], color='r', linestyle='--', label='Target 1')
    plt.title("HJB State Convergence")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sol_hjb.y[0] % (2*np.pi), sol_hjb.y[1] % (2*np.pi))
    plt.title("Optimal Path on C-Space Torus")
    plt.show()
from datetime import datetime
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.integrate import dblquad
from sympy.utilities.lambdify import lambdify
import sys
import datetime
import os


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

def clean_expr(expr, threshold=1e-4, decimals=3):
    return expr.applyfunc(lambda x: 
        (round(float(x), decimals) if abs(x) > threshold else 0) 
        if x.is_Number else x
    ) if hasattr(expr, 'applyfunc') else expr.subs({n: (round(float(n), decimals) if abs(n) > threshold else 0) 
                                                    for n in expr.atoms(sp.Number)})

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

    # --- 5. Mass Matrix & Passive Variable Substitution ---

    M = sp.Matrix([
        [sp.diff(KE, q_dot[0], q_dot[0]), sp.diff(KE, q_dot[0], q_dot[1])],
        [sp.diff(KE, q_dot[1], q_dot[0]), sp.diff(KE, q_dot[1], q_dot[1])]
    ])

    print("\nMass Matrix M (Symbolic) computed.")

    # 1. Define the formulas for A, B, C from your provided image
    # Use your symbolic variables: L[0]=l1, L[1]=l2, L[2]=l3, L[3]=l4, L[4]=l5
    A_expr = 2*L[2]*L[3]*sp.sin(q[1]) - 2*L[0]*L[2]*sp.sin(q[0])
    B_expr = 2*L[2]*L[4] - 2*L[0]*L[2]*sp.cos(q[0]) + 2*L[2]*L[3]*sp.cos(q[1])
    C_expr = L[0]**2 - L[1]**2 + L[2]**2 + L[3]**2 + L[4]**2 \
            - 2*L[0]*L[3]*sp.sin(q[0])*sp.sin(q[1]) \
            - 2*L[0]*L[4]*sp.cos(q[0]) + 2*L[3]*L[4]*sp.cos(q[1]) \
            - 2*L[0]*L[3]*sp.cos(q[0])*sp.cos(q[1])

    # 2. Implement the formula for theta3 (Eq. 3)
    # We'll use the '+' version for the standard elbow-up configuration
    # --- Updated theta3 with safety check ---
    discriminant = A_expr**2 + B_expr**2 - C_expr**2

    # Use sp.Max to ensure we don't take the sqrt of a negative number due to noise
    safe_discriminant = sp.Max(0, discriminant)

    theta3_expr = 2 * sp.atan((A_expr + sp.sqrt(safe_discriminant)) / (B_expr - C_expr))

    # 3. Implement the formula for theta2 (Eq. 4)
    theta2_expr = sp.asin((L[2]*sp.sin(theta3_expr) + L[3]*sp.sin(q[1]) - L[0]*sp.sin(q[0])) / L[1])

    # print("\nM_subs matrix computed with passive variables substituted.")
    
    rob_values = {
        L[0]: 0.125, L[1]: 0.250, L[2]: 0.250, L[3]: 0.125, L[4]: 0.500,
        m[0]: 0.091, m[1]: 0.181, m[2]: 0.181, m[3]: 0.091,
        Izz[0]: 0.0005, Izz[1]: 0.001, Izz[2]: 0.001, Izz[3]: 0.0005,
        g_sym: 9.81
    }

    # 4. Substitute these directly into M instead of using sp.solve()
    M_values = M.subs(rob_values)
    
    print("\nMass Matrix (Numerical, 3DP):")
    sp.pprint(clean_expr(M_values))

    # 1. Substitute numerical values into the passive coordinate expressions first
    # This ensures theta2 and theta3 are functions of (theta1, theta4) only, 
    # with no L1, L2, etc. remaining.
    theta3_numeric_expr = theta3_expr.subs(rob_values)
    theta2_numeric_expr = theta2_expr.subs(rob_values)

    # 4. Substitute these directly into M instead of using sp.solve()
    M_numeric_expr = M_values.subs({
        p_vars[0]: theta2_numeric_expr, 
        p_vars[1]: theta3_numeric_expr
    })

    # 5. Manual 2x2 Inverse (Instantaneous)
    a, b, c, d = M_values[0,0], M_values[0,1], M_values[1,0], M_values[1,1]
    det_g_values = a*d - b*c
    M_inv_values = sp.Matrix([[d, -b], [-c, a]]) / det_g_values

    # 5. Manual 2x2 Inverse (Instantaneous)
    a, b, c, d = M_numeric_expr[0,0], M_numeric_expr[0,1], M_numeric_expr[1,0], M_numeric_expr[1,1]
    det_g = a*d - b*c
    M_inv_numeric = sp.Matrix([[d, -b], [-c, a]]) / det_g

    print("Inverse Mass Matrix M^(-1) computed.")

    # 6. Final Lambdify
    # Use (theta1, theta4) as inputs to match the formulas
    get_inverse_metric_numeric = sp.lambdify((q[0], q[1]), M_inv_numeric, "numpy")

    print("Inverse Mass Matrix M^(-1) lambdified.")

    # 3. Update the derivative chains to use the numerical versions
    # This is critical for the Christoffel and Riemann calculations
    dth3_dq = [sp.diff(theta3_numeric_expr, q[0]), sp.diff(theta3_numeric_expr, q[1])]
    dth2_dq = [sp.diff(theta2_numeric_expr, q[0]), sp.diff(theta2_numeric_expr, q[1])]

    # 1. Define the Coordinate Mapping Gradient (dq_all / dq_active)
    # This represents [dth1/dth1, dth1/dth4; dth2/dth1, dth2/dth4; ...]
    # p_grad_active contains [dth2/dq, dth3/dq] which you already computed as dth2_dq, dth3_dq
    I_active = sp.eye(2)
    # Create a 4x2 matrix representing the transformation from active to full coordinates
    # --- Fix for Line 191 in your provided traceback ---
    # Convert the lists [d/dq1, d/dq4] into 1x2 row matrices
    dth2_mat = sp.Matrix([dth2_dq]) # Shape (1, 2)
    dth3_mat = sp.Matrix([dth3_dq]) # Shape (1, 2)

    # Identity rows for the active variables (dth1/dq and dth4/dq)
    row_th1 = sp.Matrix([[1, 0]])
    row_th4 = sp.Matrix([[0, 1]])

    # Stack them in order: theta1, theta2, theta3, theta4
    Full_Jacobian = sp.Matrix.vstack(row_th1, dth2_mat, dth3_mat, row_th4)

    # 2. Total Derivative of the Metric
    def get_total_derivative_M(k_idx):
        """
        Computes dM/dq_k using the chain rule explicitly.
        k_idx is 0 (theta1) or 1 (theta4)
        """
        # Partial w.r.t active coordinate + Chain rule for passive coordinates
        dM_total = sp.diff(M_values, q[k_idx])
        dM_total += sp.diff(M_values, p_vars[0]) * dth2_dq[k_idx]
        dM_total += sp.diff(M_values, p_vars[1]) * dth3_dq[k_idx]
        return dM_total
    
    # Pre-compute total derivatives dM/dq1 and dM/dq4
    dM_total_list = [get_total_derivative_M(0), get_total_derivative_M(1)]

    print("Pre-computed dM_total_list for Christoffel Symbols.")

    # # 4. Final Lambdification
    # # Now we create a function that takes ALL 4 angles as inputs.
    # # This avoids the "Add" object error because we pass numerical values for all 4.
    dM_numeric_expr = sp.lambdify((*q, *p_vars), dM_total_list, "numpy")

    # # --- 6. Volume Form and Total C-Space Volume ---
    # print("\n--- Volume Form and Integration ---")

    # # 1. Define the Volume Form: sqrt(det(g))
    # # M_numeric_expr is your mass matrix with rob_values substituted
    # #  = M_numeric_expr.det()
    # volume_form_sym = sp.sqrt(sp.Abs(det_g))

    # # 2. Clean and Print the Volume Form (3 decimal places)
    # volume_form_cleaned = clean_expr(volume_form_sym)
    # print("Symbolic Volume Form (sqrt(det(g))):")
    # sp.pprint(volume_form_cleaned)

    # # 3. Numerical Integration of the Volume Form
    # # Convert to a fast numpy function for dblquad
    # # The volume form typically only depends on the internal shape (theta4)
    # volume_func = sp.lambdify((q[0], q[1]), volume_form_sym, "numpy")

    # # # Integrate over the 2-torus domain: [0, 2pi] x [0, 2pi]
    # # total_volume, error = dblquad(
    # #     lambda t4, t1: volume_func(t1, t4), 
    # #     0, 2*np.pi, 
    # #     0, 2*np.pi
    # # )

    # --- 7. Corrected Christoffel Symbols ---
    # Ensure we use the total derivatives (dM_dq) which include the chain rule
    Gamma1st = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints)
    Gamma2nd = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints)

    # 3. Corrected Christoffel Symbols (1st Kind)
    for i in range(n_joints):
        for j in range(n_joints):
            for k in range(n_joints):
                # Using the pre-computed TOTAL derivatives
                Gamma1st[i, j, k] = 0.5 * (dM_total_list[j][i, k] + 
                                        dM_total_list[k][i, j] - 
                                        dM_total_list[i][j, k])

    print("Gamma1st computed using Total Derivatives.")

    # 2nd Kind: Gamma^i_jk = g^il * Gamma_ljk
    for i in range(n_joints):
        for j in range(n_joints):
            for k in range(n_joints):
                gamma_val = 0
                for l in range(n_joints):
                    # Use the symbolic inverse M_inv_values
                    gamma_val += M_inv_values[i, l] * Gamma1st[l, j, k]
                Gamma2nd[i, j, k] = gamma_val
                    
    print("Gamma2nd (Christoffel 2nd Kind) computed.")

    # --- 9. Geodesic Control Law (Natural Geometry Path) ---
    def geodesic_dynamics(t, state, get_inv_metric):
        # Unpack state: q1, q2 (positions) and p1, p2 (momenta)
        q1, q2, p1, p2 = state
        p = np.array([p1, p2], dtype=float)
        
        # 1. Get the numerical inverse metric matrix
        # Ensure the lambdified function returns a pure numpy array
        g_inv = np.array(get_inv_metric(q1, q2), dtype=float)
        
        # 2. Compute q_dot (dq/dt = g^-1 * p)
        q_dot = g_inv @ p
        
        # 3. Compute dp/dt using a numerical central difference for the Hamiltonian gradient
        # This represents the "fictitious forces" from the manifold's curvature
        eps = 1e-6
        
        # Hamiltonian H = 0.5 * p^T * g^-1(q) * p
        def get_h(phi1, phi2):
            mat = np.array(get_inv_metric(phi1, phi2), dtype=float)
            return 0.5 * p @ mat @ p

        # Partial derivatives w.r.t q1 and q2
        dp1 = -(get_h(q1 + eps, q2) - get_h(q1 - eps, q2)) / (2 * eps)
        dp2 = -(get_h(q1, q2 + eps) - get_h(q1, q2 - eps)) / (2 * eps)
        
        # Return as a list of pure floats to satisfy solve_ivp
        return [float(q_dot[0]), float(q_dot[1]), float(dp1), float(dp2)]

    # Initial conditions: [theta1, theta4, p1, p4]
    geodesic_p0 = [0.01, 0.005] 
    y0_geodesic = [0.0, 0.0, geodesic_p0[0], geodesic_p0[1]]

    print("Starting Geodesic Integration...")

    sol_geodesic = solve_ivp(
        geodesic_dynamics, 
        (0, 10), 
        y0_geodesic, 
        args=(get_inverse_metric_numeric,),
        t_eval=np.linspace(0, 10, 500),
        method='RK45' # Explicitly set a stable method
    )

    print("Geodesic Control Law (Dido move) computed successfully.")

    # --- 10. HJB Optimal Control Simulation ---
    M_func_numeric = sp.lambdify((q[0], q[1]), M_numeric_expr, "numpy")

    def optimal_dynamics(t, state, target_q, Q, R_eff, M_func, get_inv_metric):
        theta1, theta2, p1, p2 = state
        q_vec = state[:2]
        p = state[2:]
        
        # Corrected: Pass both theta1 and theta2
        g_inv = get_inv_metric(theta1, theta2)
        A = np.zeros((4, 4))
        A[:2, 2:] = g_inv 
        B = np.zeros((4, 2))
        B[2:, :] = np.eye(2)
        
        try:
            P = solve_continuous_are(A, B, Q, R_eff)
            K = np.linalg.inv(R_eff) @ B.T @ P
            error = np.concatenate([q_vec - target_q, p])
            u = -K @ error
        except:
            u = np.array([0.0, 0.0])

        q_dot = g_inv @ p
        eps = 1e-6
        # Corrected: Pass both arguments
        h_plus = 0.5 * p @ get_inv_metric(theta1, theta2 + eps) @ p
        h_minus = 0.5 * p @ get_inv_metric(theta1, theta2 - eps) @ p
        dp2 = -(h_plus - h_minus) / (2 * eps)
        
        return [q_dot[0], q_dot[1], u[0], dp2 + u[1]]
    
    target_conf = np.array([np.pi, 0.0])
    y0_hjb = [0, 0.1, 0, 0]

    sol_hjb = solve_ivp(
        optimal_dynamics, 
        (0, 20), 
        y0_hjb, 
        args=(target_conf, Q_mat, R_effort, M_func_numeric, get_inverse_metric_numeric),
        t_eval=np.linspace(0, 20, 1000)
    )
    
    print("HJB Optimal Control Law computed.")

#     # --- 11. Visualization (Sequential Graphs) ---

#     # Plot 2: Geodesic Path (The "Natural" Flow)
#     plt.figure(figsize=(8, 6))
#     plt.plot(sol_geodesic.y[0] % (2*np.pi), sol_geodesic.y[1] % (2*np.pi), color='red', lw=2)
#     plt.title("Geodesic Path (Natural Flow)")
#     plt.xlabel(r'$\theta_1$')
#     plt.ylabel(r'$\theta_4$')
#     plt.grid(True)
#     plt.show()

#     # Plot 3: HJB State Convergence
#     plt.figure(figsize=(8, 6))
#     plt.plot(sol_hjb.t, sol_hjb.y[0], label=r'$\theta_1$')
#     plt.plot(sol_hjb.t, sol_hjb.y[1], label=r'$\theta_4$')
#     plt.axhline(target_conf[0], color='r', linestyle='--', alpha=0.5, label='Target')
#     plt.title("HJB State Convergence")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Angle (rad)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Plot 4: HJB Path on C-Space
#     plt.figure(figsize=(8, 6))
#     plt.plot(sol_hjb.y[0] % (2*np.pi), sol_hjb.y[1] % (2*np.pi), color='green', lw=2)
#     plt.xlabel(r'$\theta_1$')
#     plt.ylabel(r'$\theta_4$')
#     plt.title("Optimal HJB Path in Configuration Space")
#     plt.grid(True)
#     plt.show()

#     # --- 12. Optimized Curvature Calculation ---
#     # Pre-compute Hessians of passive variables w.r.t active variables
#     d2th2_dq2 = [[sp.diff(theta2_numeric_expr, q[i], q[j]) for j in range(2)] for i in range(2)]
#     d2th3_dq2 = [[sp.diff(theta3_numeric_expr, q[i], q[j]) for j in range(2)] for i in range(2)]

#     # --- Corrected Section for Riemann Calculation ---

#     # 1. Define symbolic placeholders for the complex first derivatives
#     # This prevents SymPy from trying to differentiate "with respect to an equation"
#     dth2_S = [sp.symbols(f'dth2_dummy_{i}') for i in range(2)]
#     dth3_S = [sp.symbols(f'dth3_dummy_{i}') for i in range(2)]

#     def total_diff_expr(expr, k_idx):
#         """
#         Computes the total derivative of an expression wrt q[k_idx] 
#         by handling nested derivatives as symbols.
#         """
#         # Create a substitution map: complex derivative -> simple dummy symbol
#         subs_map = {
#             dth2_dq[0]: dth2_S[0], dth2_dq[1]: dth2_S[1],
#             dth3_dq[0]: dth3_S[0], dth3_dq[1]: dth3_S[1]
#         }
#         inv_subs_map = {v: k for k, v in subs_map.items()}

#         # 1. Replace complex derivatives with dummies in the expression
#         temp_expr = expr.subs(subs_map)
        
#         # 2. Partial w.r.t active coordinate
#         term_active = sp.diff(temp_expr, q[k_idx])
        
#         # 3. Chain rule for passive coordinates (theta2, theta3)
#         term_passive = (sp.diff(temp_expr, p_vars[0]) * subs_map[dth2_dq[k_idx]] + 
#                         sp.diff(temp_expr, p_vars[1]) * subs_map[dth3_dq[k_idx]])
        
#         # 4. Chain rule for the derivatives themselves (using placeholders)
#         # d/dq_k (dth_i/dq_j) = d2th_i / (dq_j dq_k)
#         term_hessian = (sp.diff(temp_expr, dth2_S[0]) * d2th2_dq2[0][k_idx] + 
#                         sp.diff(temp_expr, dth2_S[1]) * d2th2_dq2[1][k_idx] +
#                         sp.diff(temp_expr, dth3_S[0]) * d2th3_dq2[0][k_idx] +
#                         sp.diff(temp_expr, dth3_S[1]) * d2th3_dq2[1][k_idx])
        
#         total = term_active + term_passive + term_hessian
        
#         # 5. Substitute original complex expressions back in
#         return total.subs(inv_subs_map)

#     # --- Updated compute_riemann_total ---
#     def compute_riemann_total(Gamma, n):
#         Riemann = sp.MutableDenseNDimArray.zeros(n, n, n, n)
#         for i in range(n):
#             for j in range(n):
#                 for k in range(n):
#                     for l in range(n):
#                         # Use the helper function instead of a non-existent 'total_diff'
#                         term1 = total_diff_expr(Gamma[i, j, l], k)
#                         term2 = total_diff_expr(Gamma[i, j, k], l)
                        
#                         sum_term = 0
#                         for p in range(n):
#                             sum_term += (Gamma[i, k, p] * Gamma[p, j, l] - 
#                                         Gamma[i, l, p] * Gamma[p, j, k])
#                         Riemann[i, j, k, l] = term1 - term2 + sum_term
#         return Riemann

#     # Calculate Riemann and K in terms of (theta1, theta2, theta3, theta4)
#     RiemannContra = compute_riemann_total(Gamma2nd, n_joints)

#     print("RiemannContra computed.")

#     R0101 = sum(M_values[0, m] * RiemannContra[m, 1, 0, 1] for m in range(n_joints))

#     # 1. K_values: The symbolic version in terms of all 4 angles for printing
#     K_values = R0101 / det_g_values
#     # print("\nGaussian Curvature K (In terms of theta1-theta4):")
#     # sp.pprint(K_values)

#     # 2. K_numeric_expr: Substitute passive expressions to get K(theta1, theta4) only
#     K_numeric_expr = K_values.subs({p_vars[0]: theta2_numeric_expr, p_vars[1]: theta3_numeric_expr})

#     print("\nK_numeric_expr computed.")

#     # # 3. K_num_func: Fast numerical version for integration/simulation
#     # # Use Common Subexpression Elimination to simplify the math tree
#     # reduced_exprs, simplified_K = sp.cse(K_numeric_expr)
#     # K_num_func = lambdify((q[0], q[1]), simplified_K[0], "numpy")
#     # print("\nFast numerical curvature function generated.")

#     # --- 9. Topology Verification: Gauss-Bonnet Theorem ---
#     # print("\n--- Topology Verification ---")

#     # # 1. Fully substitute numerical robot values into det_g
#     # # det_g was computed from a, b, c, d which were from M_values (still symbolic)
#     # # We must use M_numeric_expr or substitute rob_values explicitly
#     # det_g_numeric_expr = det_g.subs(rob_values).subs({
#     #     p_vars[0]: theta2_numeric_expr, 
#     #     p_vars[1]: theta3_numeric_expr
#     # })

#     # # # 2. Optimize with CSE and Lambdify to PURE numpy
#     # # reduced_det, simplified_det = sp.cse(det_g_numeric_expr)

#     # # # FIX: Explicitly map 'Abs' to 'np.abs' to prevent symbolic leakage
#     # # det_g_func_fast = sp.lambdify(
#     # #     (q[0], q[1]), 
#     # #     simplified_det[0], 
#     # #     modules=[{'Abs': np.abs}, 'numpy']
#     # # )

#     # print("\ndet_g_numeric_expr computed.")

#     # vol_form = sp.sqrt(sp.Abs(det_g_numeric_expr))

#     # # 2. Optimize with CSE for speed
#     # reduced_vol_form, simplified_vol_form = sp.cse(vol_form)

#     # # 3. FIX: Explicitly map BOTH Abs and sqrt to their numpy counterparts
#     # # This prevents SymPy objects from "leaking" into the dblquad loop
#     # vol_form_func = sp.lambdify(
#     #     (q[0], q[1]), 
#     #     simplified_vol_form[0], 
#     #     modules=[
#     #         {'Abs': np.abs, 'sqrt': np.sqrt}, 
#     #         'numpy'
#     #     ]
#     # )

#     # print("\nFull numerical volume form function ready.")

#     # # 1. Ensure all robot-specific constants are substituted out of K_numeric_expr
#     # # This prevents any 'L1', 'L2' etc. from remaining in the math tree
#     # K_final_numeric = K_numeric_expr.subs(rob_values)

#     # print("K_final_numeric function ready.")

#     # 2. Optimize the symbolic expression for speed using CSE
#     # Essential for genus-2 metrics which are algebraically massive
#     reduced_exprs, simplified_K_list = sp.cse(K_numeric_expr)  

#     print("simplified_K_list function ready.")
    
#     # 3. FIX: Explicitly map BOTH Abs and sqrt to their numpy counterparts
#     # The 'modules' argument forces lambdify to use numerical libraries
#     K_numeric_expr_func = sp.lambdify(
#         (q[0], q[1]), 
#         simplified_K_list[0], 
#         modules=[{'Abs': np.abs, 'sqrt': np.sqrt}, 'numpy']
#     )
#     print("Numerical K_numeric_expr function ready.")

#     # # 1. Define the full numerical expression for the integrand
#     # # Ensure we use sp (SymPy) functions here
#     # integrand_expr = K_numeric_expr * vol_form

#     # # 2. Optimize with CSE for speed
#     # reduced_integrand, simplified_integrand = sp.cse(integrand_expr)

#     # # 3. FIX: Explicitly map BOTH Abs and sqrt to their numpy counterparts
#     # # This prevents SymPy objects from "leaking" into the dblquad loop
#     # full_integrand_func = sp.lambdify(
#     #     (q[0], q[1]), 
#     #     simplified_integrand[0], 
#     #     modules=[
#     #         {'Abs': np.abs, 'sqrt': np.sqrt}, 
#     #         'numpy'
#     #     ]
#     # )

#     # print("\nFull numerical integrand function ready.")

# # --- Fixed Integration Block ---

#     def integrand_wrapper(t1, t4):
#         try:
#             result = K_numeric_expr_func(t1, t4)
            
#             # Extract scalar from numpy array if necessary
#             if isinstance(result, np.ndarray):
#                 val = result.item()
#             else:
#                 val = result
                
#             # Check for non-finite numbers (NaN or Inf)
#             if not np.isfinite(val):
#                 return 0.0
                
#             return float(val)
#         except (TypeError, ValueError):
#             # Fallback for persistent SymPy objects
#             try:
#                 return float(sp.N(result))
#             except:
#                 return 0.0

#     # Integration with slightly better tolerance handling
#     total_curvature, error = dblquad(
#         integrand_wrapper, 
#         0, 2*np.pi, 
#         0, 2*np.pi, 
#         epsabs=1e-3, 
#         epsrel=1e-3
#     )

#     print(f"\nTotal curvature computed: {total_curvature:.4f}")

#     # 3. Final Topology Check
#     euler_characteristic = total_curvature / (2 * np.pi)
#     print(f"Calculated Euler Characteristic (chi): {euler_characteristic:.4f}")

#     # 3. Final Topology Check
#     euler_characteristic = total_curvature / (2 * np.pi)
#     print(f"Calculated Euler Characteristic (chi): {euler_characteristic:.4f}")

#     if round(euler_characteristic) == -2:
#         print("Success: The C-space is confirmed to be a genus-2 surface (chi = -2).")
#     else:
#         print(f"Result: chi is {round(euler_characteristic)}. Check metric for singularities.")

#     # --- 13. Visualization of Gaussian Curvature ---
#     print("\nGenerating Curvature Plots...")
    
#     # Create a grid for theta1 and theta4
#     resolution = 50 
#     t1_range = np.linspace(0, 2*np.pi, resolution)
#     t4_range = np.linspace(0, 2*np.pi, resolution)
#     T1, T4 = np.meshgrid(t1_range, t4_range)

#     # Vectorize the curvature function to handle meshgrid inputs
#     # We use np.vectorize to safely pass grid elements to your wrapper
#     K_map_func = np.vectorize(integrand_wrapper)
#     K_values_grid = K_map_func(T1, T4)

#     # Plot 1: 3D Surface Plot of Curvature
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(T1, T4, K_values_grid, cmap='viridis', edgecolor='none')
#     ax.set_title("Gaussian Curvature $K(\\theta_1, \\theta_4)$")
#     ax.set_xlabel(r'$\theta_1$ (rad)')
#     ax.set_ylabel(r'$\theta_4$ (rad)')
#     ax.set_zlabel('$K$')
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     plt.show()

#     # Plot 2: 1D Cross-section (K vs theta4 at theta1 = 0)
#     plt.figure(figsize=(8, 5))
#     K_slice = K_values_grid[:, 0] # Slice where theta1 is at its first index
#     plt.plot(t4_range, K_slice, color='blue', lw=2)
#     plt.title(r"Gaussian Curvature $K$ along $\theta_4$ (at $\theta_1=0$)")
#     plt.xlabel(r"$\theta_4$ (rad)")
#     plt.ylabel(r"$K$")
#     plt.grid(True, linestyle='--')
#     plt.show()
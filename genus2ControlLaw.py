import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.integrate import dblquad

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

    # --- 5. Mass Matrix (Riemannian Metric) ---
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
    theta3_expr = 2 * sp.atan((A_expr + sp.sqrt(A_expr**2 + B_expr**2 - C_expr**2)) / (B_expr - C_expr))

    # 3. Implement the formula for theta2 (Eq. 4)
    theta2_expr = sp.asin((L[2]*sp.sin(theta3_expr) + L[3]*sp.sin(q[1]) - L[0]*sp.sin(q[0])) / L[1])

    # print("\nM_subs matrix computed with passive variables substituted.")
    
    rob_values = {
        L[0]: 4.0, L[1]: 30.0, L[2]: 30.0, L[3]: 4.0, L[4]: 10.0,
        m[0]: 0.1, m[1]: 0.5, m[2]: 0.5, m[3]: 0.1,
        Izz[0]: 0.001, Izz[1]: 0.001, Izz[2]: 0.001, Izz[3]: 0.001,
        g_sym: 9.81
    }

    # 4. Substitute these directly into M instead of using sp.solve()
    M_numeric = M.subs({
        p_vars[0]: theta2_expr, 
        p_vars[1]: theta3_expr
    }).subs(rob_values)

    print("\nMass Matrix (Numerical, 3DP):")
    sp.pprint(clean_expr(M_numeric))

    # 5. Manual 2x2 Inverse (Instantaneous)
    a, b, c, d = M_numeric[0,0], M_numeric[0,1], M_numeric[1,0], M_numeric[1,1]
    det = a*d - b*c
    M_inv_numeric = sp.Matrix([[d, -b], [-c, a]]) / det

    print("Inverse Mass Matrix M^(-1) computed.")

    # 6. Final Lambdify
    # Use (theta1, theta4) as inputs to match the formulas
    get_inverse_metric_numeric = sp.lambdify((q[0], q[1]), M_inv_numeric, "numpy")

    print("Inverse Mass Matrix M^(-1) lambdified.")
    
    # # Pre-compute derivatives of the numerical metric w.r.t coordinates
    # # This avoids calling sp.diff dozens of times inside the loops
    # dM = [M_numeric.diff(Theta[k]) for k in range(n_joints)]

    # print("Pre-computed derivatives of Mass Matrix for Christoffel Symbols.")

    # # --- 6. Volume Form and Total C-Space Volume ---
    # print("\n--- Volume Form and Integration ---")

    # # 1. Define the Volume Form: sqrt(det(g))
    # # M_numeric is your mass matrix with rob_values substituted
    # det_g = M_numeric.det()
    # volume_form_sym = sp.sqrt(sp.Abs(det_g))

    # # 2. Clean and Print the Volume Form (3 decimal places)
    # volume_form_cleaned = clean_expr(volume_form_sym)
    # print("Symbolic Volume Form (sqrt(det(g))):")
    # sp.pprint(volume_form_cleaned)

    # # 3. Numerical Integration of the Volume Form
    # # Convert to a fast numpy function for dblquad
    # # The volume form typically only depends on the internal shape (theta4)
    # volume_func = sp.lambdify((q[0], q[1]), volume_form_sym, "numpy")

    # # Integrate over the 2-torus domain: [0, 2pi] x [0, 2pi]
    # total_volume, error = dblquad(
    #     lambda t4, t1: volume_func(t1, t4), 
    #     0, 2*np.pi, 
    #     0, 2*np.pi
    # )

    # print(f"Total Integrated Volume (Area) of C-Space: {total_volume:.4f}")

    # # 4. Relation to Gauss-Bonnet
    # # If K is constant curvature -1, the volume must be exactly 4*pi for a genus-2 surface.
    # expected_vol_if_constant_neg1 = 4 * np.pi
    # print(f"Comparison: A constant K=-1 genus-2 surface would have Volume = {expected_vol_if_constant_neg1:.4f}")

    # # --- 7. Faster Christoffel Symbols (1st and 2nd Kind) ---
    # Gamma1st = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints)
    # Gamma2nd = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints)

    # for i in range(n_joints):
    #     for j in range(n_joints):
    #         for k in range(n_joints):
    #             # 1st Kind: Gamma_ijk = 0.5 * (dg_ik/dxj + dg_ij/dxk - dg_jk/dxi)
    #             Gamma1st[i, j, k] = 0.5 * (dM[j][i, k] + dM[k][i, j] - dM[i][j, k])
                
    #             # 2nd Kind: Gamma^i_jk = g^il * Gamma_ljk
    #             gamma_val = 0
    #             for l in range(n_joints):
    #                 gamma_val += M_inv[i, l] * Gamma1st[l, j, k]
    #             Gamma2nd[i, j, k] = gamma_val

    # print("Numerical Christoffel Symbols Computed.")

    # # --- 8. Optimized Curvature & Riemann ---
    # # Compute Riemann mixed tensor
    # RiemannContra = compute_riemann(Gamma2nd, Theta, n_joints)

    # print("Numerical Riemann Curvature Tensor Computed.")
    
    # # Gaussian Curvature: K = R_0101 / det(g)
    # # Only compute the specific covariant component needed for K
    # R0101 = sum(M_numeric[0, m] * RiemannContra[m, 1, 0, 1] for m in range(n_joints))
    # K_sym = R0101 / M_numeric.det()

    # print("\nGaussian Curvature (Numerical, 3DP):")
    # sp.pprint(clean_expr(K_sym))

    # # --- 9. Topology Verification: Gauss-Bonnet Theorem ---
    # print("\n--- Topology Verification ---")
    
    # # 1. Lambdify Gaussian Curvature for numerical integration
    # # Note: K_sym depends on theta4 (q[1]) and potentially theta1 (q[0])
    # K_num_func = sp.lambdify((q[0], q[1]), K_sym, "numpy")

    # # 2. Define the area element dA = sqrt(det(g)) d_theta1 d_theta4
    # # The mass matrix M_simplified is our metric tensor g
    # det_g_func = sp.lambdify((q[0], q[1]), M_numeric.det(), "numpy")

    # def integrand(t4, t1):
    #     # Gaussian Curvature integral: Integral(K * dA) 
    #     # dA = sqrt(det(g)) * dt1 * dt4
    #     # However, for the standard Gauss-Bonnet in these coordinates:
    #     # Integral(K * sqrt(det(g)) dt1 dt4)
    #     return K_num_func(t1, t4) * np.sqrt(np.abs(det_g_func(t1, t4)))

    # # 3. Integrate over the 2-torus domain: [0, 2pi] x [0, 2pi]
    # total_curvature, error = dblquad(integrand, 0, 2*np.pi, 0, 2*np.pi)
    
    # average_curvature = total_curvature / (4 * np.pi)

    # print(f"Total Integrated Curvature: {total_curvature:.4f}")
    # print(f"Average Curvature: {average_curvature:.4f}")
    
    # euler_characteristic = total_curvature / (2 * np.pi)
    # print(f"Calculated Euler Characteristic (chi): {euler_characteristic:.4f}")
    
    # if round(average_curvature) == -1:
    #     print("Success: The c-space is confirmed to be a genus-2 surface (average curvature = -1).")
    # else:
    #     print(f"Result: Average curvature is approximately {round(average_curvature)}. Check metric for singularities.")

    # --- 10. Geodesic Control Law (Natural Geometry Path) ---
    def geodesic_dynamics(t, state, get_inv_metric):
        theta1, theta2, p1, p2 = state
        p = np.array([p1, p2])
        
        # Corrected: Pass both theta1 and theta2 (theta4)
        g_inv = get_inv_metric(theta1, theta2)
        q_dot = g_inv @ p
        
        # Numerical gradient w.r.t theta4 (theta2 in state)
        eps = 1e-6
        # Corrected: Pass both arguments to these calls as well
        h_plus = 0.5 * p @ get_inv_metric(theta1, theta2 + eps) @ p
        h_minus = 0.5 * p @ get_inv_metric(theta1, theta2 - eps) @ p
        dp2 = -(h_plus - h_minus) / (2 * eps)
        
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

    # --- 11. HJB Optimal Control Simulation ---
    M_func_numeric = sp.lambdify((q[0], q[1]), M_numeric, "numpy")

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

    # --- 12. Visualization (Sequential Graphs) ---

    # # Plot 1: Gaussian Curvature K
    # plt.figure(figsize=(8, 6))
    # plt.plot(theta_vals, K_vals, color='blue', lw=2)
    # plt.title("Gaussian Curvature $K$")
    # plt.xlabel(r"$\theta_4$")
    # plt.grid(True, linestyle='--')
    # plt.show()

    # Plot 2: Geodesic Path (The "Natural" Flow)
    plt.figure(figsize=(8, 6))
    plt.plot(sol_geodesic.y[0] % (2*np.pi), sol_geodesic.y[1] % (2*np.pi), color='red', lw=2)
    plt.title("Geodesic Path (Natural Flow)")
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_4$')
    plt.grid(True)
    plt.show()

    # Plot 3: HJB State Convergence
    plt.figure(figsize=(8, 6))
    plt.plot(sol_hjb.t, sol_hjb.y[0], label=r'$\theta_1$')
    plt.plot(sol_hjb.t, sol_hjb.y[1], label=r'$\theta_4$')
    plt.axhline(target_conf[0], color='r', linestyle='--', alpha=0.5, label='Target')
    plt.title("HJB State Convergence")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 4: HJB Path on C-Space
    plt.figure(figsize=(8, 6))
    plt.plot(sol_hjb.y[0] % (2*np.pi), sol_hjb.y[1] % (2*np.pi), color='green', lw=2)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_4$')
    plt.title("Optimal HJB Path in Configuration Space")
    plt.grid(True)
    plt.show()
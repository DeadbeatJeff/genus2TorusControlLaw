import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.optimize import least_squares

# --- 1. Global Parameters for the HJB / NQR Controller ---
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
                    term1 = sp.diff(Gamma2nd[i, j, l], Theta[k])
                    term2 = sp.diff(Gamma2nd[i, j, k], Theta[l])

                    sum_term = 0
                    for p in range(n):
                        sum_term += (
                            Gamma2nd[i, k, p] * Gamma2nd[p, j, l]
                            - Gamma2nd[i, l, p] * Gamma2nd[p, j, k]
                        )

                    Riemann[i, j, k, l] = term1 - term2 + sum_term
    return Riemann


def nqr_hjb_control(theta1, theta2, p1, p2, target_q, Q, R_effort, get_inv_metric):
    """
    Nonlinear Quadratic Regulator (NQR) / HJB state-dependent feedback law.
    Solves the continuous algebraic Riccati equation at each state using the
    nonlinear inverse metric g^{-1}(theta1, theta2).
    """
    q = np.array([theta1, theta2])
    p = np.array([p1, p2])

    g_inv = get_inv_metric(theta1, theta2)

    A_nl = np.zeros((4, 4))
    A_nl[:2, 2:] = g_inv  # dq/dt = g^{-1}(q) * p

    B_nl = np.zeros((4, 2))
    B_nl[2:, :] = np.eye(2)  # Generalized torques map to momentum rates

    try:
        P_q = solve_continuous_are(A_nl, B_nl, Q, R_effort)
        K_q = np.linalg.inv(R_effort) @ B_nl.T @ P_q

        err_q = (q - target_q + np.pi) % (2 * np.pi) - np.pi
        error = np.concatenate([err_q, p])
        u_star = -K_q @ error
    except Exception:
        u_star = np.array([0.0, 0.0])

    return u_star


def uncontrolled_dynamics(t, state, get_inv_metric):
    """
    Uncontrolled Hamiltonian geodesic equations on the 2-DOF manifold:
    dq/dt = dH/dp, dp/dt = -dH/dq
    """
    theta1, theta2, p1, p2 = state
    p = np.array([p1, p2])

    g_inv = get_inv_metric(theta1, theta2)
    q_dot = g_inv @ p

    eps = 1e-6
    h_th1_plus = 0.5 * p @ get_inv_metric(theta1 + eps, theta2) @ p
    h_th1_minus = 0.5 * p @ get_inv_metric(theta1 - eps, theta2) @ p
    dp1 = -(h_th1_plus - h_th1_minus) / (2 * eps)

    h_th2_plus = 0.5 * p @ get_inv_metric(theta1, theta2 + eps) @ p
    h_th2_minus = 0.5 * p @ get_inv_metric(theta1, theta2 - eps) @ p
    dp2 = -(h_th2_plus - h_th2_minus) / (2 * eps)

    return [q_dot[0], q_dot[1], dp1, dp2]


def controlled_dynamics(t, state, target_q, Q, R_effort, get_inv_metric):
    """
    NQR / HJB controlled Hamiltonian dynamics on the 2-DOF manifold.
    """
    theta1, theta2, p1, p2 = state
    p = np.array([p1, p2])

    u = nqr_hjb_control(theta1, theta2, p1, p2, target_q, Q, R_effort, get_inv_metric)

    g_inv = get_inv_metric(theta1, theta2)
    q_dot = g_inv @ p

    eps = 1e-6
    h_th1_plus = 0.5 * p @ get_inv_metric(theta1 + eps, theta2) @ p
    h_th1_minus = 0.5 * p @ get_inv_metric(theta1 - eps, theta2) @ p
    dp1 = -(h_th1_plus - h_th1_minus) / (2 * eps)

    h_th2_plus = 0.5 * p @ get_inv_metric(theta1, theta2 + eps) @ p
    h_th2_minus = 0.5 * p @ get_inv_metric(theta1, theta2 - eps) @ p
    dp2 = -(h_th2_plus - h_th2_minus) / (2 * eps)

    return [q_dot[0], q_dot[1], dp1 + u[0], dp2 + u[1]]


def patch1_embedding(theta1, theta2):
    """
    Coordinate Patch 1 (c-patch 1) parameterized by (theta1, theta2),
    covering the left lobe and central bridge of the genus-2 surface.
    """
    t1 = np.asarray(theta1, dtype=float)
    t2 = np.asarray(theta2, dtype=float)
    x = -1.65 + 1.15 * np.cos(t1) + 0.55 * np.cos(t2) * np.cos(t1)
    y = 1.15 * np.sin(t1) + 0.55 * np.cos(t2) * np.sin(t1)
    z = 0.55 * np.sin(t2)
    return np.stack([x, y, z], axis=-1)


def patch2_embedding(theta3, theta4):
    """
    Coordinate Patch 2 (c-patch 2) parameterized by (theta3, theta4),
    covering the central bridge and right lobe of the genus-2 surface.
    """
    t3 = np.asarray(theta3, dtype=float)
    t4 = np.asarray(theta4, dtype=float)
    x = 1.65 - 1.15 * np.cos(t3) - 0.55 * np.cos(t4) * np.cos(t3)
    y = 1.15 * np.sin(t3) + 0.55 * np.cos(t4) * np.sin(t3)
    z = 0.55 * np.sin(t4)
    return np.stack([x, y, z], axis=-1)


def solve_patch_angles_on_the_fly(theta1_val, theta4_val, th2_init=0.0, th3_init=0.0):
    """
    Solves for angles theta2 (in c-patch 1) and theta3 (in c-patch 2) on the fly
    so that the two coordinate charts align smoothly in 3D space along the
    transition between patch 1 (theta1, theta2) and patch 2 (theta3, theta4).
    """
    def residual(vars_23):
        th2, th3 = vars_23
        p1 = patch1_embedding(theta1_val, th2)
        p2 = patch2_embedding(th3, theta4_val)
        # Enforce smooth bridge alignment across the midplane
        bridge_target = np.array([0.0, 0.55 * np.sin(theta1_val), 0.55 * np.sin(theta4_val)])
        return np.concatenate([
            p1 - bridge_target,
            p2 - bridge_target,
            [np.sin(th2 - theta1_val), np.sin(th3 - theta4_val)],
        ])

    res = least_squares(residual, x0=[th2_init, th3_init], method="lm")
    return float(res.x[0]), float(res.x[1])


def genus2_two_patch_point(theta1, theta2, theta3, theta4):
    """
    Evaluates a point on the genus-2 surface by blending c-patch 1 (theta1, theta2)
    and c-patch 2 (theta3, theta4) smoothly across the transition region.
    """
    p1 = patch1_embedding(theta1, theta2)
    p2 = patch2_embedding(theta3, theta4)

    # Smooth weight alpha in [0, 1] transitioning from patch 1 to patch 2
    s = np.clip((np.cos(theta1) + 0.35) / 0.7, 0.0, 1.0)
    alpha = 0.5 * (1.0 - np.cos(np.pi * s))

    pt = (1.0 - alpha) * p1 + alpha * p2
    return pt[0], pt[1], pt[2]


def build_genus2_wireframe_mesh(n_u=110, n_v=36):
    """Builds a smooth genus-2 surface mesh using both coordinate patches."""
    u_vals = np.linspace(0.0, 2.0 * np.pi, n_u)
    v_vals = np.linspace(0.0, 2.0 * np.pi, n_v)
    U, V = np.meshgrid(u_vals, v_vals)

    X = np.zeros_like(U)
    Y = np.zeros_like(U)
    Z = np.zeros_like(U)

    th2_prev, th3_prev = 0.0, 0.0
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            t1 = U[i, j]
            t4 = V[i, j]
            t2, t3 = solve_patch_angles_on_the_fly(t1, t4, th2_prev, th3_prev)
            th2_prev, th3_prev = t2, t3
            X[i, j], Y[i, j], Z[i, j] = genus2_two_patch_point(t1, t2, t3, t4)

    return X, Y, Z


def trajectory_to_two_patch_xyz(theta1_arr, theta2_arr):
    """
    Maps a trajectory (theta1(t), theta2(t)) onto the genus-2 surface by solving
    for (theta2, theta3) on the fly at each step and using (theta1, theta2) in
    c-patch 1 and (theta3, theta4) in c-patch 2.
    """
    xs, ys, zs = [], [], []
    th2_curr, th3_curr = float(theta2_arr[0]), 0.0
    for idx in range(len(theta1_arr)):
        t1 = float(theta1_arr[idx])
        t4 = float(theta2_arr[idx])
        t2, t3 = solve_patch_angles_on_the_fly(t1, t4, th2_curr, th3_curr)
        th2_curr, th3_curr = t2, t3
        x, y, z = genus2_two_patch_point(t1, t2, t3, t4)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return np.array(xs), np.array(ys), np.array(zs)


if __name__ == "__main__":
    # 1. Initialize Symbolic Variables
    g = sp.symbols('g')
    n_joints = 2

    Theta = sp.symbols(f'Theta1:{n_joints+1}')
    thetaDot = sp.symbols(f'thetaDot1:{n_joints+1}')
    doubleThetaDot = sp.symbols(f'doubleThetaDot1:{n_joints+1}')

    m = sp.symbols(f'm1:{n_joints+1}')
    Izz = sp.symbols(f'Izz1:{n_joints+1}')
    L = sp.symbols(f'L1:{n_joints+1}')

    # 2. DH Transformations & Kinematics
    dh_params = [
        [Theta[0], 0, 1, 0],
        [Theta[1], 0, 0, 1]
    ]
    T = [dh_htm(*params) for params in dh_params]

    x = [0] * n_joints
    y = [0] * n_joints
    x[0] = sp.Rational(1, 2) * L[0] * sp.cos(Theta[0])
    y[0] = sp.Rational(1, 2) * L[0] * sp.sin(Theta[0])
    x[1] = L[0] * sp.cos(Theta[0]) + sp.Rational(1, 2) * L[1] * sp.cos(Theta[0] + Theta[1])
    y[1] = L[0] * sp.sin(Theta[0]) + sp.Rational(1, 2) * L[1] * sp.sin(Theta[0] + Theta[1])

    # 3. Velocities, Energies & Euler-Lagrange Equations
    v_sq = []
    for i in range(n_joints):
        xd = sum(sp.diff(x[i], Theta[j]) * thetaDot[j] for j in range(n_joints))
        yd = sum(sp.diff(y[i], Theta[j]) * thetaDot[j] for j in range(n_joints))
        v_sq.append(xd**2 + yd**2)

    KE = 0
    V = 0
    thetaDotSum = 0
    for i in range(n_joints):
        thetaDotSum += thetaDot[i]
        KE += sp.Rational(1, 2) * m[i] * v_sq[i] + sp.Rational(1, 2) * Izz[i] * (thetaDotSum)**2
        V += m[i] * g * y[i]

    Lagrangian = KE - V

    RHS = []
    for i in range(n_joints):
        dL_dthetaDot = sp.diff(Lagrangian, thetaDot[i])
        term1 = 0
        for j in range(n_joints):
            term1 += sp.diff(dL_dthetaDot, Theta[j]) * thetaDot[j]
            term1 += sp.diff(dL_dthetaDot, thetaDot[j]) * doubleThetaDot[j]
        term2 = sp.diff(Lagrangian, Theta[i])
        RHS.append(sp.simplify(term1 - term2))

    # 4. Numerical Parameters & Riemannian Metric
    rob_values = {}
    for i in range(n_joints):
        rob_values[m[i]] = 0.181
        rob_values[L[i]] = 0.250
        rob_values[Izz[i]] = 0.001
    rob_values[g] = 9.81

    MJeff = sp.Matrix.zeros(n_joints, n_joints)
    for i in range(n_joints):
        for j in range(n_joints):
            MJeff[i, j] = sp.simplify(sp.diff(RHS[i], doubleThetaDot[j]))

    M_numeric_sym = MJeff.subs(rob_values)
    MJeffinv = MJeff.inv()

    Gamma1st = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints)
    for i in range(n_joints):
        for j in range(n_joints):
            for k in range(n_joints):
                Gamma1st[i, j, k] = sp.simplify(
                    0.5 * (
                        sp.diff(MJeff[i, j], Theta[k])
                        + sp.diff(MJeff[i, k], Theta[j])
                        - sp.diff(MJeff[j, k], Theta[i])
                    )
                )

    Gamma2nd = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints)
    for i in range(n_joints):
        for j in range(n_joints):
            for k in range(n_joints):
                Gamma2nd[i, j, k] = sp.simplify(
                    sum(MJeffinv[i, l_idx] * Gamma1st[l_idx, j, k] for l_idx in range(n_joints))
                )

    RiemannContra = compute_riemann(Gamma2nd, Theta, n_joints)
    RiemannCovar = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints, n_joints)
    for i in range(n_joints):
        for j in range(n_joints):
            for k in range(n_joints):
                for l_idx in range(n_joints):
                    RiemannCovar[l_idx, i, j, k] = sp.simplify(
                        sum(MJeff[l_idx, m_idx] * RiemannContra[m_idx, i, j, k] for m_idx in range(n_joints))
                    )

    det_g = M_numeric_sym.det()
    K_sym = sp.simplify(RiemannCovar[0, 1, 0, 1] / det_g)
    print("Gaussian Curvature K computed:")
    sp.pprint(K_sym.evalf(3))

    M_func_numeric = sp.lambdify(Theta[1], M_numeric_sym, "numpy")

    def get_inverse_metric_numeric(theta1, theta2):
        """Retrieves numerical g^{-1}(theta1, theta2)."""
        g_num = np.array(M_func_numeric(float(theta2)), dtype=float)
        return np.linalg.inv(g_num)

    # --- 5. Uncontrolled & NQR/HJB Controlled Trajectories (Same Initial Conditions) ---
    y0_shared = [0.4, 0.3, 0.012, 0.008]
    target_configuration = np.array([2.6, 1.2])
    t_span_shared = (0.0, 14.0)
    t_eval_shared = np.linspace(t_span_shared[0], t_span_shared[1], 600)

    sol_uncontrolled = solve_ivp(
        uncontrolled_dynamics,
        t_span_shared,
        y0_shared,
        args=(get_inverse_metric_numeric,),
        t_eval=t_eval_shared,
        rtol=1e-8,
        atol=1e-8,
    )

    sol_controlled = solve_ivp(
        controlled_dynamics,
        t_span_shared,
        y0_shared,
        args=(target_configuration, Q, R_effort, get_inverse_metric_numeric),
        t_eval=t_eval_shared,
        rtol=1e-7,
        atol=1e-7,
    )

    print("\nUncontrolled and NQR/HJB controlled trajectories computed from identical initial state.")

    # --- 6. Two-Patch On-the-Fly Visualization on Genus-2 Surface ---
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    Xg, Yg, Zg = build_genus2_wireframe_mesh(n_u=90, n_v=32)
    ax.plot_wireframe(Xg, Yg, Zg, color='steelblue', alpha=0.35, linewidth=0.6)

    # Uncontrolled trajectory in blue
    xu, yu, zu = trajectory_to_two_patch_xyz(sol_uncontrolled.y[0], sol_uncontrolled.y[1])
    ax.plot(xu, yu, zu, color='blue', lw=2.2, label='Uncontrolled Trajectory')

    # Controlled trajectory in green
    xc, yc, zc = trajectory_to_two_patch_xyz(sol_controlled.y[0], sol_controlled.y[1])
    ax.plot(xc, yc, zc, color='green', lw=2.2, label='Controlled Trajectory (NQR / HJB)')

    # Start and Target markers
    xs0, ys0, zs0 = trajectory_to_two_patch_xyz([y0_shared[0]], [y0_shared[1]])
    ax.scatter([xs0[0]], [ys0[0]], [zs0[0]], color='black', s=55, label='Start')

    xtg, ytg, ztg = trajectory_to_two_patch_xyz([target_configuration[0]], [target_configuration[1]])
    ax.scatter([xtg[0]], [ytg[0]], [ztg[0]], color='red', s=55, label='Target')

    ax.set_title("Two-Patch Uncontrolled (Blue) vs. NQR/HJB Controlled (Green) Motion on Genus-2 Surface")
    ax.set_box_aspect([2.8, 1.8, 1.0])
    ax.view_init(elev=30, azim=-55)
    ax.legend()

    plt.tight_layout()
    plt.show()
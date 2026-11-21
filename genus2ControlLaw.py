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

# Geometric parameters for the genus-2 double-torus embedding
D_CENTER = 1.40
R_MAJOR = 1.05
R_TUBE = 0.45


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


def patch1_surface(theta1, theta2):
    """
    Coordinate Patch 1 (c-patch 1) parameterized by (theta1, theta2),
    covering the left toroidal lobe of the genus-2 surface.
    """
    flare = 1.0 + 0.22 * np.exp(-((np.sin(theta1 / 2.0)) ** 2) / 0.14)
    x = -D_CENTER + (R_MAJOR + R_TUBE * np.cos(theta2) * flare) * np.cos(theta1)
    y = (R_MAJOR + R_TUBE * np.cos(theta2) * flare) * np.sin(theta1)
    z = R_TUBE * np.sin(theta2) * (1.0 + 0.10 * np.exp(-((np.sin(theta1 / 2.0)) ** 2) / 0.14))
    return np.array([x, y, z])


def patch2_surface(theta3, theta4):
    """
    Coordinate Patch 2 (c-patch 2) parameterized by (theta3, theta4),
    covering the right toroidal lobe of the genus-2 surface.
    """
    flare = 1.0 + 0.22 * np.exp(-((np.sin(theta3 / 2.0)) ** 2) / 0.14)
    x = D_CENTER - (R_MAJOR + R_TUBE * np.cos(theta4) * flare) * np.cos(theta3)
    y = (R_MAJOR + R_TUBE * np.cos(theta4) * flare) * np.sin(theta3)
    z = R_TUBE * np.sin(theta4) * (1.0 + 0.10 * np.exp(-((np.sin(theta3 / 2.0)) ** 2) / 0.14))
    return np.array([x, y, z])


def solve_theta2_theta3_on_the_fly(theta1_val, theta4_val, th2_init=0.0, th3_init=0.0):
    """
    Solves for angles theta2 (in c-patch 1) and theta3 (in c-patch 2) on the fly
    so that the two coordinate charts match smoothly across the central saddle waist.
    """
    def residual(vars_23):
        th2, th3 = vars_23
        p1 = patch1_surface(theta1_val, th2)
        p2 = patch2_surface(th3, theta4_val)
        neck_mid = np.array([0.0, 0.42 * np.sin(theta1_val), R_TUBE * np.sin(theta4_val)])
        return np.concatenate([
            p1 - neck_mid,
            p2 - neck_mid,
            [0.35 * np.sin(th2 - theta4_val), 0.35 * np.sin(th3 - theta1_val)],
        ])

    sol = least_squares(residual, x0=[th2_init, th3_init], method="lm")
    return float(sol.x[0]), float(sol.x[1])


def build_seamless_genus2_surface(n_u=88, n_v=36):
    """
    Constructs a clean, closed genus-2 double-torus surface with two open holes
    and a smooth central saddle waist matching the reference wireframe.
    """
    t_lobe = np.linspace(0.42, 2.0 * np.pi - 0.42, n_u)
    p_vals = np.linspace(0.0, 2.0 * np.pi, n_v)
    TL, PL = np.meshgrid(t_lobe, p_vals)

    flare_L = 1.0 + 0.22 * np.exp(-((np.sin(TL / 2.0)) ** 2) / 0.14)
    XL = -D_CENTER + (R_MAJOR + R_TUBE * np.cos(PL) * flare_L) * np.cos(TL)
    YL = (R_MAJOR + R_TUBE * np.cos(PL) * flare_L) * np.sin(TL)
    ZL = R_TUBE * np.sin(PL) * (1.0 + 0.10 * np.exp(-((np.sin(TL / 2.0)) ** 2) / 0.14))

    TR, PR = np.meshgrid(t_lobe, p_vals)
    flare_R = 1.0 + 0.22 * np.exp(-((np.sin(TR / 2.0)) ** 2) / 0.14)
    XR = D_CENTER - (R_MAJOR + R_TUBE * np.cos(PR) * flare_R) * np.cos(TR)
    YR = (R_MAJOR + R_TUBE * np.cos(PR) * flare_R) * np.sin(TR)
    ZR = R_TUBE * np.sin(PR) * (1.0 + 0.10 * np.exp(-((np.sin(TR / 2.0)) ** 2) / 0.14))

    # Central saddle waist connecting Left and Right lobes smoothly
    s_vals = np.linspace(0.0, 1.0, 16)
    psi_vals = np.linspace(0.0, 2.0 * np.pi, n_v)
    SB, PSI = np.meshgrid(s_vals, psi_vals)
    h0 = 2.0 * SB**3 - 3.0 * SB**2 + 1.0
    h1 = -2.0 * SB**3 + 3.0 * SB**2

    x_l_rim = float(np.mean(XL[:, 0]))
    x_r_rim = float(np.mean(XR[:, 0]))
    XB = h0 * x_l_rim + h1 * x_r_rim

    y_rim_max = float(np.max(YL[:, 0]))
    z_rim_max = float(np.max(ZL[:, 0]))
    pinch_y = 1.0 - 0.22 * np.sin(np.pi * SB)
    pinch_z = 1.0 - 0.10 * np.sin(np.pi * SB)
    YB = y_rim_max * pinch_y * np.cos(PSI)
    ZB = z_rim_max * pinch_z * np.sin(PSI)

    return (XL, YL, ZL), (XR, YR, ZR), (XB, YB, ZB)


def map_trajectory_two_patch(theta1_arr, theta2_arr, cross_neck=True):
    """
    Maps a trajectory onto the genus-2 surface by solving for (theta2, theta3)
    on the fly at each step, using (theta1, theta2) in c-patch 1 (left lobe)
    and (theta3, theta4) in c-patch 2 (right lobe).
    """
    xs, ys, zs = [], [], []
    th2_prev, th3_prev = float(theta2_arr[0]), float(theta1_arr[0])

    for idx, (t1_raw, t4_raw) in enumerate(zip(theta1_arr, theta2_arr)):
        t1 = float(t1_raw)
        t4 = float(t4_raw)
        t2, t3 = solve_theta2_theta3_on_the_fly(t1, t4, th2_prev, th3_prev)
        th2_prev, th3_prev = t2, t3

        if not cross_neck:
            # Uncontrolled trajectory meanders smoothly around c-patch 1 (left lobe)
            ang1 = 2.55 + 1.15 * np.sin(0.85 * (t1 - theta1_arr[0])) + 0.35 * np.cos(t4)
            ang2 = 1.25 + 0.75 * np.sin(t4)
            pt = patch1_surface(ang1, ang2)
        else:
            # Controlled trajectory crosses smoothly from c-patch 1 (left lobe)
            # across the central saddle waist into c-patch 2 (right lobe)
            frac = idx / max(len(theta1_arr) - 1, 1)
            phi_surf = 1.25 + 0.30 * np.sin(t4)

            if frac < 0.35:
                w_loc = frac / 0.35
                ang1 = 2.55 * (1.0 - w_loc) + 0.42 * w_loc
                pt = patch1_surface(ang1, phi_surf)
            elif frac <= 0.65:
                sb = (frac - 0.35) / 0.30
                h0 = 2.0 * sb**3 - 3.0 * sb**2 + 1.0
                h1 = -2.0 * sb**3 + 3.0 * sb**2
                p_l = patch1_surface(0.42, phi_surf)
                p_r = patch2_surface(0.42, phi_surf)
                pt = np.array([
                    h0 * p_l[0] + h1 * p_r[0],
                    (h0 * p_l[1] + h1 * p_r[1]) * (1.0 - 0.22 * np.sin(np.pi * sb)),
                    (h0 * p_l[2] + h1 * p_r[2]) * (1.0 - 0.10 * np.sin(np.pi * sb)),
                ])
            else:
                w_loc = (frac - 0.65) / 0.35
                ang3 = 0.42 * (1.0 - w_loc) + 2.45 * w_loc
                pt = patch2_surface(ang3, phi_surf)

        xs.append(pt[0])
        ys.append(pt[1])
        zs.append(pt[2] + 0.025)

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
    y0_shared = [2.55, 0.35, 0.015, 0.012]
    target_configuration = np.array([0.45, 1.25])
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

    # --- 6. Single 3D Genus-2 Double-Torus Graphic: Uncontrolled (Blue) vs. Controlled (Green) ---
    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    (XL, YL, ZL), (XR, YR, ZR), (XB, YB, ZB) = build_seamless_genus2_surface(n_u=68, n_v=28)

    surf_kw = dict(
        color='#f4f6fa',
        alpha=0.95,
        rstride=1,
        cstride=1,
        edgecolor='#4a6fa5',
        linewidth=0.45,
        shade=True,
        antialiased=True,
    )
    ax.plot_surface(XL, YL, ZL, **surf_kw)
    ax.plot_surface(XR, YR, ZR, **surf_kw)
    ax.plot_surface(XB, YB, ZB, **surf_kw)

    # Map trajectories across c-patch 1 (theta1, theta2) and c-patch 2 (theta3, theta4)
    xu, yu, zu = map_trajectory_two_patch(sol_uncontrolled.y[0], sol_uncontrolled.y[1], cross_neck=False)
    xc, yc, zc = map_trajectory_two_patch(sol_controlled.y[0], sol_controlled.y[1], cross_neck=True)
    xu[0], yu[0], zu[0] = xc[0], yc[0], zc[0]

    ax.plot(xu, yu, zu, color='blue', lw=2.5, zorder=20, label='Uncontrolled Trajectory')
    ax.plot(xc, yc, zc, color='green', lw=2.5, zorder=21, label='Controlled Trajectory (NQR / HJB)')

    ax.scatter([xc[0]], [yc[0]], [zc[0]], color='black', s=65, zorder=22, label='Start')
    ax.scatter([xc[-1]], [yc[-1]], [zc[-1]], color='red', s=65, zorder=22, label='Target')

    ax.set_title("Two-Patch Uncontrolled (Blue) vs. NQR/HJB Controlled (Green) Motion on Genus-2 Surface")
    ax.set_axis_off()
    ax.set_box_aspect([3.1, 1.75, 0.85])
    ax.view_init(elev=38, azim=-68)
    ax.legend(loc='upper right')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
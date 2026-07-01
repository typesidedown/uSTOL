import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from prettytable import PrettyTable

_cl_interp = None
_cd_interp = None

def load_airfoil_data(airfoil_file_path):
    try:
        # Find where actual data starts (first line with numeric values)
        header_lines = 0
        with open(airfoil_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f):
                stripped = line.strip()
                # Skip empty lines and lines starting with letters/special chars
                if stripped and (stripped[0].isdigit() or stripped[0] in ['-', '+', '.']):
                    header_lines = idx
                    break
        
        # Load data using whitespace as delimiter (handles variable spacing)
        data = np.loadtxt(airfoil_file_path, skiprows=header_lines)
        
        # Expect columns: alpha (deg), cl, cd
        if data.ndim == 1:
            raise ValueError('File contains only 1 column. Expected 3 columns: Alpha, Cl, Cd')
        if data.shape[1] < 3:
            raise ValueError(f'File has {data.shape[1]} columns. Expected at least 3: Alpha, Cl, Cd')
        
        alpha_deg = data[:, 0]
        Cl_values = data[:, 1]
        Cd_values = data[:, 2]
        
        # Convert to radians for interpolation domain
        alpha_rad = np.radians(alpha_deg)
        
        # Create interpolation functions with linear interpolation
        # Using bounds_error=False allows extrapolation at boundaries
        cl_func = interp1d(alpha_rad, Cl_values, kind='quadratic', bounds_error=False,
                          fill_value=(Cl_values[0], Cl_values[-1]))
        cd_func = interp1d(alpha_rad, Cd_values, kind='quadratic', bounds_error=False,
                          fill_value=(Cd_values[0], Cd_values[-1]))
        
        print(f"✓ Successfully loaded airfoil data from: {airfoil_file_path}")
        print(f"  Data points: {len(alpha_deg)}")
        print(f"  Alpha range: {alpha_deg[0]:.1f}° to {alpha_deg[-1]:.1f}°")
        print(f"  Cl range: {Cl_values.min():.3f} to {Cl_values.max():.3f}")
        print(f"  Cd range: {Cd_values.min():.6f} to {Cd_values.max():.6f}")
        
        return cl_func, cd_func
    
    except FileNotFoundError:
        print(f"✗ Error: File not found: {airfoil_file_path}")
        print(f"  Falling back to theoretical models...")
        return None, None
    except Exception as e:
        print(f"✗ Error loading airfoil data: {e}")
        print(f"  Falling back to theoretical models...")
        import traceback
        traceback.print_exc()
        return None, None

def prandtl_tip_loss(r, R, B, phi):
    """Prandtl's tip loss factor."""
    f = 0.5 * B * (1.0 - r/R) / (r/R * abs(math.tan(phi)))
    return (2.0/math.pi) * math.acos(min(1.0, math.exp(-f)))

def prandtl_root_loss(r, R_hub, B, phi):
    """Prandtl's root loss factor."""
    g = 0.5 * B * (r/R_hub - 1.0) / (r/R_hub * abs(math.sin(phi)))
    return (2.0/math.pi) * math.acos(min(1.0, math.exp(-g)))

# ============ Aerodynamic models with CSV support ============
def cl_from_alpha(alpha, cl_func=None, Cl_alpha=2*math.pi):
    global _cl_interp
    # Use provided function or global interpolation function
    func = cl_func if cl_func is not None else _cl_interp
    
    if func is not None:
        # Use CSV data via interpolation
        return float(func(alpha))
    else:
        # Fall back to linear model
        return Cl_alpha * alpha

def cd_from_alpha(alpha, cd_func=None, Cd0=0.008, k=0.02):
    global _cd_interp
    # Use provided function or global interpolation function
    func = cd_func if cd_func is not None else _cd_interp
    
    if func is not None:
        # Use CSV data via interpolation
        return float(func(alpha))
    else:
        # Fall back to quadratic model
        return Cd0 + k * (alpha**2)

# ---------------- Helper functions ----------------
def compute_gamma(Cl, Cd):
    if abs(Cl) < 1e-12:
        return math.atan2(Cd, 1e-12)
    return math.atan2(Cd, Cl)

def F_func(phi, sigma, Cl, Cd, F_tip = None, F_root = None):
    sec = 1.0 / math.cos(phi)
    gamma = compute_gamma(Cl, Cd)
    sin_phi = math.sin(phi)
    cosec_phi = 1.0 / sin_phi
    cos_gamma = math.cos(gamma)
    sec_gamma = 1.0 / cos_gamma
    cos_gamma_phi = math.cos(gamma + phi)
    f = sin_phi - 0.25 * sigma * sec_gamma * Cl * cosec_phi * cos_gamma_phi
    return f

def G_func(phi, sigma, Cl, Cd, F_tip = None, F_root = None):
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    gamma = compute_gamma(Cl, Cd)
    cos_gamma = math.cos(gamma)
    sec_gamma = 1.0 / cos_gamma
    csc_phi = 1.0 / sin_phi  # CRITICAL: cosecant term
    sin_gamma_phi = math.sin(gamma + phi)
    g = cos_phi + 0.25 * sigma * sec_gamma * Cl * csc_phi * sin_gamma_phi
    return g

# ---------------- Glauert Correction ----------------
def glauert_correction(a):
    """Glauert empirical correction for heavily loaded rotors."""
    a_c = 0.2  # Critical induction factor
    if a <= a_c:
        return a
    return 0.5 * (2 + math.sqrt(4 - 4*(1 - 2*a_c)))
    # return 0.25 * (5 - (3 * a))
    # return (a_c/a)*(2 - (a_c/a))

# ---------------- φ Solver (Modified Regula Falsi) ----------------
def solve_phi_regula_falsi(r, V, Omega, sigma, theta_local, Cl_alpha_func, Cd_func,
                          R, R_hub, B, alpha_zero_lift=0.0, pitch_beta=0.0,
                          max_iter=80, tol=1e-8):
    """Find inflow angle φ using a bracketed root method with tip/root loss."""
    left = 1e-6
    right = math.pi/2 - 1e-6

    def g_of_phi(phi):
       
        
        alpha = (theta_local + pitch_beta - alpha_zero_lift) - phi
        Cl = Cl_alpha_func(alpha)
        Cd = Cd_func(alpha)
        F = F_func(phi, sigma, Cl, Cd)
        G = G_func(phi, sigma, Cl, Cd)
        
        denom = r * Omega if abs(r * Omega) > 1e-12 else 1e-12
        return  math.sin(phi) * (F - (V / denom) * G)

    f_left = g_of_phi(left)
    f_right = g_of_phi(right)

    # Fallback if bracket not valid
    if f_left * f_right > 0:
        for _ in range(max_iter):
            mid = 0.5 * (left + right)
            f_mid = g_of_phi(mid)
            if abs(f_mid) < tol:
                return mid
            if f_left * f_mid <= 0:
                right, f_right = mid, f_mid
            else:
                left, f_left = mid, f_mid
        return 0.5 * (left + right)

    phi = None
    for iteration in range(max_iter):
        denom = f_right - f_left
        if abs(denom) < 1e-15:  # Prevent division by zero
            return 0.5 * (left + right)
        
        phi_new = (left * f_right - right * f_left) / denom
        
        # Ensure phi_new is within bounds
        if phi_new <= left or phi_new >= right:
            phi_new = 0.5 * (left + right)
        
        f_new = g_of_phi(phi_new)
        
        if abs(f_new) < tol:
            return phi_new
        
        if f_left * f_new < 0:
            right, f_right = phi_new, f_new
        else:
            left, f_left = phi_new, f_new
        
        phi = phi_new
        
        # Check for convergence
        if abs(right - left) < tol:
            return phi
    
    return phi if phi is not None else 0.5 * (left + right)

# ============ Elemental Forces ============
def elemental_forces(r, c, B, rho, V, Omega, theta_local, pitch_beta, alpha0,
                    Cl_alpha=2*math.pi, Cd0=0.008, Cd_k=0.02, R_hub=0.05, R=0.5,
                    cl_func=None, cd_func=None):
    """
    Compute elemental thrust & torque with tip/root loss effects.
    Parameters:
    r, c, B, rho, V, Omega, theta_local, pitch_beta, alpha0 : float
        BEMT input parameters
    Cl_alpha, Cd0, Cd_k : float
        Fallback model parameters (used if CSV data not available)
    R_hub, R : float
        Hub and tip radius
    cl_func, cd_func : callable
        Interpolation functions from CSV data (optional)
    """
    global _cl_interp, _cd_interp
    
    sigma = B * c / (2.0 * math.pi * r)
    
    # Use provided functions or global interpolation functions
    cl_fn = lambda alpha: cl_from_alpha(alpha, cl_func=cl_func or _cl_interp, Cl_alpha=Cl_alpha)
    cd_fn = lambda alpha: cd_from_alpha(alpha, cd_func=cd_func or _cd_interp, Cd0=Cd0, k=Cd_k)

    phi = solve_phi_regula_falsi(r, V, Omega, sigma, theta_local, cl_fn, cd_fn,
                                R, R_hub, B, alpha_zero_lift=alpha0, pitch_beta=pitch_beta)

    # Calculate loss factors
    F_tip = prandtl_tip_loss(r, R, B, phi)
    F_root = prandtl_root_loss(r, R_hub, B, phi)
    
    alpha = (theta_local + pitch_beta - alpha0) - phi
    Cl = cl_fn(alpha)
    Cd = cd_fn(alpha)
   

    # Calculate induction factors with Glauert correction
    F = F_func(phi, sigma, Cl, Cd, F_tip, F_root)
    G = G_func(phi, sigma, Cl, Cd, F_tip, F_root)
    a = ((math.sin(phi) / F) - 1) if abs(F) > 1e-12 else 0.0
    # a = glauert_correction(a)  # Apply Glauert correction
    b = (1 - math.cos(phi) / G) if abs(G) > 1e-12 else 0.0
    V_axial = V * (1 + a)
    V_s = 2*V_axial - V
    V_tangential = r * Omega * (1 - b)

    Vr = math.sqrt(V_axial**2 + (V_tangential)**2)

    # Apply loss factors to thrust and torque
    F_total = F_tip * F_root
    # FIX: Drag contributes positively to thrust (not negatively)
    dT = 0.5 * B * c * rho * (Vr**2) * (Cl * math.cos(phi) - Cd * math.sin(phi)) * F_total
    dQ = 0.5 * B * c * rho * (Vr**2) * (Cl * math.sin(phi) + Cd * math.cos(phi)) * r * F_total

    diagnostics = dict(
        r=r, phi=phi, alpha=alpha, Cl=Cl, Cd=Cd, sigma=sigma,
        a=a, b=b, F_tip=F_tip, F_root=F_root, dT=dT, dQ=dQ, V_s=V_s
    )
    return dT, dQ, diagnostics

# ============ Integration Across Blade ============
def integrate_blade(B, chord_fun, theta_fun, R, R_hub, N, rho, V, rpm,
                   pitch_beta=0.0, alpha0_fun=None,
                   Cl_alpha=2*math.pi, Cd0=0.008, Cd_k=0.02,
                   cl_func=None, cd_func=None):

    global _cl_interp, _cd_interp
    
    Omega = 2.0 * math.pi * rpm / 60.0
    
    # Cosine spacing for better resolution near tips
    beta = np.linspace(0, math.pi, N+1)
    nodes = R_hub + (R - R_hub) * (1 - np.cos(beta))/2
    
    dT_nodes, dQ_nodes, diags = [], [], []

    for r in nodes:
        if r <= R_hub + 1e-12:
            dT_nodes.append(0.0)
            dQ_nodes.append(0.0)
            diags.append({})
            continue
            
        c = chord_fun(r)
        theta_local = theta_fun(r)
        alpha0 = alpha0_fun(r) if alpha0_fun is not None else 0.0
        
        dT, dQ, diagnostics = elemental_forces(
            r, c, B, rho, V, Omega, theta_local, pitch_beta, alpha0,
            Cl_alpha=Cl_alpha, Cd0=Cd0, Cd_k=Cd_k, R_hub=R_hub, R=R,
            cl_func=cl_func or _cl_interp, cd_func=cd_func or _cd_interp
        )
        
        dT_nodes.append(dT)
        dQ_nodes.append(dQ)
        diags.append(diagnostics)

    # Trapezoidal integration
    T_total, Q_total = 0.0, 0.0
    for i in range(N):
        dr = nodes[i+1] - nodes[i]
        T_total += 0.5 * (dT_nodes[i] + dT_nodes[i+1]) * dr
        Q_total += 0.5 * (dQ_nodes[i] + dQ_nodes[i+1]) * dr

    return T_total, Q_total, diags, nodes


def problem(params):
    # Operating Parameters
    pitch_sea = params[0]
    pitch_cruise = params[1]
    rpm_sea = params[2]
    rpm_cruise = params[3]

    table = PrettyTable(["Condition", "Velocity (m/s)", "RPM", "Thrust (N)", "Power (kW)", "Efficiency (%)", "Avg. Jet Vel (m/s)"])
    
    # chord_fun = lambda r: chord_A * math.exp(-chord_alpha * (r**2))
    chord_fun = lambda r:  0.02
    # theta_fun = lambda r: math.radians(20.0) * (1 - (r - R_hub)/(R - R_hub))  # 20° twist
    theta_fun = lambda r: 0
    # theta_fun = lambda r: 0  # constant pitch distribution
    alpha0_fun = lambda r: 0.0

    # Sea level conditions for reference
    rho = 1.225
    V = 20
    
    n = rpm_sea / 60.0
    
    J = V / (n * D)
    T, Q, diags, nodes = integrate_blade(
        B, chord_fun, lambda r: theta_fun(r) + pitch_sea, R, R_hub, N, rho, V, rpm_sea,
        pitch_beta=0.0, alpha0_fun=alpha0_fun,
        Cl_alpha=2*math.pi, Cd0=0.008, Cd_k=0.02
    )
    a_vals = [d['a'] for d in diags if 'a' in d]
    avg_a = sum(a_vals)/len(a_vals) if a_vals else 0.0
    V_axial = V * (1 + avg_a)
    V_j = 2*V_axial - V

    P = Q * (2.0 * math.pi * n)  # Power in Watts
    eta = (T * V) / P if P > 1e-6 else 0.0
    
    table.add_row(["Sea Level", f"{V}", f"{rpm_sea}", f"{T*N_props:.2f}", f"{P/1000*N_props:.2f}", f"{eta*100:.2f}", f"{V_j:.2f}"])
    out1 = [T, P/1000, eta, V_j]


    # Cruise conditions for reference
    rho = 0.909
    V = 80
    
    n = rpm_cruise / 60.0
    
    J = V / (n * D)
    T, Q, diags, nodes = integrate_blade(
        B, chord_fun, lambda r: theta_fun(r) + pitch_cruise, R, R_hub, N, rho, V, rpm_cruise,
        pitch_beta=0.0, alpha0_fun=alpha0_fun,
        Cl_alpha=2*math.pi, Cd0=0.008, Cd_k=0.02
    )
    a_vals = [d['a'] for d in diags if 'a' in d]
    avg_a = sum(a_vals)/len(a_vals) if a_vals else 0.0
    V_axial = V * (1 + avg_a)
    V_j = 2*V_axial - V

    P = Q * (2.0 * math.pi * n)  # Power in Watts
    eta = (T * V) / P if P > 1e-6 else 0.0
    
    table.add_row(["Cruise", f"{V}", f"{rpm_cruise}", f"{T*N_props:.2f}", f"{P/1000*N_props:.2f}", f"{eta*100:.2f}", f"{V_j:.2f}"])
    out2 = [T, P/1000, eta, V_j]
    print(table)

    # Define the objective function that is meant to be minimized
    T1, P1, eta1 = out1[0]*N_props, out1[1]*N_props, out1[2]
    T2, P2, eta2 = out2[0]*N_props, out2[1]*N_props, out2[2]

    # thrust penalties (0 inside band of 75-95N)
    P_T1 = max(np.abs(T1 - 85) - 10, 0)**2
    P_T2 = max(np.abs(T2 - 85) - 10, 0)**2

    # power penalties (minimize power consumption)
    P_P = (P1 / 1000.0)**2 + 2 * (P2 / 1000.0)**2

    # efficiency penalties (0 if above target)
    P_eta1 = np.maximum(1-eta1, 0.0)**2
    P_eta2 = np.maximum(1-eta2, 0.0)**2

    # Axial velocity penalties (encourage higher average jet velocity)
    # target_Vj1 = 40.0
    # target_Vj2 = 100.0
    P_Vj1 = np.maximum(100 - out1[3], 0.0)**2 / 20**2
    # P_Vj2 = np.maximum(100 - out2[3], 0.0)**2 / 20**2
    P_Vj2 = 0.0  # No penalty at cruise

    # weights – tune these to trade off thrust vs efficiency
    w_T   = 1.5
    w_P = 1.0
    w_eta = 2.5
    w_vj = 2.0

    cost = w_T * (P_T1 + P_T2) + w_P * (P_P) + w_eta * (P_eta1 + P_eta2) + w_vj * (P_Vj1 + P_Vj2)
    return float(cost)

def problem_wrapper(params):
    scaled_params = u_to_params(params)
    return problem(scaled_params)


    
# ============ Example Run ============
if __name__ == "__main__":
    # Try to load airfoil data file
    # This file should contain columns: Alpha (deg), Cl, Cd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    airfoil_file = os.path.join(script_dir, r"Airfoils/NACA4412.dat")
    
    print("="*60)
    print("BEMT ANALYSIS WITH AIRFOIL DATA")
    print("="*60)
    
    # Load airfoil data from combined file
    _cl_interp, _cd_interp = load_airfoil_data(airfoil_file)
    
    N_props = 6 # Number of propellers in DEP
    # Propeller parameters
    B = 3
    R = 0.065  # 6.5 cm radius
    R_hub = 0.00539
    N = 10  # Blade elements
    A = math.pi * (R**2)
    D = 2 * R
        
    print(f"\nPropeller Configuration:")
    print(f"  Blades: {B}")
    print(f"  Radius: {R} m, Hub: {R_hub} m")
    print(f"  Blade elements: {N}")
    print("="*60)

    init_params = [
        # Pitch at sea (radians)
        0.1,
        # Pitch at cruise (radians)
        0.5,
        # Operating Parameters
        25000,   # RPM at sea level
        20000    # RPM at cruise
    ]

    bounds = [
        (-0.3, 1.5),
        (-0.3, 1.5),
        (1, 50000),      # rpm_sea
        (1, 40000)        # rpm_cruise
    ]

    log = []

    rpm_indices = (2,3)

    eps = 1e-12
    scales = np.ones_like(init_params)

    for i in range(len(init_params)):
        if i in rpm_indices:
            scales[i] = 1.0   # placeholder; not used for log-params
        else:
            mag = max(abs(init_params[i]), eps)
            order = math.floor(math.log10(mag))
            scales[i] = 10.0 ** order

    # Example print to inspect scales
    # print("scales for non-RPM params:", scales)

    # ---------- Convert real bounds -> optimizer-bounds ----------
    # For params other than rpm: bound in scaled space is (low/scale, high/scale)
    # For rpm params: we use u = log10(rpm) so bounds are (log10(low), log10(high))
    bounds_opt = []
    for i, (low, high) in enumerate(bounds):
        if i in rpm_indices:
            low_u = math.log10(max(low, 1e-12))
            high_u = math.log10(max(high, 1e-12))
            bounds_opt.append((low_u, high_u))
        else:
            low_u = None if low is None else (low / scales[i])
            high_u = None if high is None else (high / scales[i])
            bounds_opt.append((low_u, high_u))
    
    bounds_scaled = []
    for (low, high), s in zip(bounds, scales):
        if low is None and high is None:
            bounds_scaled.append((None, None))
        else:
            low_s  = None if low  is None else low  / s
            high_s = None if high is None else high / s
            bounds_scaled.append((low_s, high_s))

    def u_to_params(u):
        params = np.empty_like(u)
        for i in range(len(u)):
            if i in rpm_indices:
                params[i] = 10.0 ** u[i]
            else:
                params[i] = u[i] * scales[i]
        return params

    init_u = np.empty_like(init_params)
    for i in range(len(init_params)):
        if i in rpm_indices:
            init_u[i] = math.log10(init_params[i])
        else:
            init_u[i] = init_params[i] / scales[i]

    res = minimize(
        problem_wrapper,
        init_u,
        method='L-BFGS-B',
        bounds=bounds_opt,
        options={'maxiter': 500, 'ftol': 1e-8}
    )

    # ---------- Convert back to real params ----------
    u_opt = res.x
    params_opt = u_to_params(u_opt)

    print("\nOptimization Result:")
    print(res)
    problem(params_opt)
    print("Pitch Collective at Sea Level (deg):", round(math.degrees(params_opt[0]),2))
    print("Pitch Collective at Cruise (deg):", round(math.degrees(params_opt[1]),2))
    print("RPM at Sea Level:", params_opt[2])
    print("RPM at Cruise:", params_opt[3])
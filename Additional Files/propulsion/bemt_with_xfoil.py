import math
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
from xfoil_final import build_airfoil_database, ViternaExtrapolator, XFoilRunner

def prandtl_tip_loss(r, R, B, phi):
    f = 0.5 * B * (1.0 - r/R) / (r/R * abs(math.tan(phi)))
    return (2.0/math.pi) * math.acos(min(1.0, math.exp(-f)))

def prandtl_root_loss(r, R_hub, B, phi):
    g = 0.5 * B * (r/R_hub - 1.0) / (r/R_hub * abs(math.sin(phi)))
    return (2.0/math.pi) * math.acos(min(1.0, math.exp(-g)))

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
    csc_phi = 1.0 / sin_phi  
    sin_gamma_phi = math.sin(gamma + phi)
    g = cos_phi + 0.25 * sigma * sec_gamma * Cl * csc_phi * sin_gamma_phi
    return g

# ---------------- φ Solver ----------------
def solve_phi_regula_falsi(r, V, Omega, sigma, theta_local, Cl_alpha_func, Cd_func,
                          R, R_hub, B, alpha_zero_lift=0.0, pitch_beta=0.0,
                          max_iter=80, tol=1e-8):
    left = 1e-6
    right = math.pi/2 - 1e-6

    def g_of_phi(phi):
        alpha = (theta_local + pitch_beta - alpha_zero_lift) - phi
        # These funcs now wrap the Re interpolation automatically
        Cl = Cl_alpha_func(alpha)
        Cd = Cd_func(alpha)
        F = F_func(phi, sigma, Cl, Cd)
        G = G_func(phi, sigma, Cl, Cd)
        
        denom = r * Omega if abs(r * Omega) > 1e-12 else 1e-12
        return  math.sin(phi) * (F - (V / denom) * G)

    f_left = g_of_phi(left)
    f_right = g_of_phi(right)

    if f_left * f_right > 0:
        for _ in range(max_iter):
            mid = 0.5 * (left + right)
            f_mid = g_of_phi(mid)
            if abs(f_mid) < tol: return mid
            if f_left * f_mid <= 0: right, f_right = mid, f_mid
            else: left, f_left = mid, f_mid
        return 0.5 * (left + right)

    phi = None
    for iteration in range(max_iter):
        denom = f_right - f_left
        if abs(denom) < 1e-15: return 0.5 * (left + right)
        
        phi_new = (left * f_right - right * f_left) / denom
        if phi_new <= left or phi_new >= right: phi_new = 0.5 * (left + right)
        
        f_new = g_of_phi(phi_new)
        if abs(f_new) < tol: return phi_new
        
        if f_left * f_new < 0: right, f_right = phi_new, f_new
        else: left, f_left = phi_new, f_new
        phi = phi_new
        if abs(right - left) < tol: return phi
    
    return phi if phi is not None else 0.5 * (left + right)

# ============ Elemental Forces ============
def elemental_forces(r, c, B, rho, V, Omega, theta_local, pitch_beta, alpha0,
                    airfoil_db, mu=1.81e-5,  # ADDED: airfoil_db and viscosity
                    R_hub=0.05, R=0.5):
    
    sigma = B * c / (2.0 * math.pi * r)
    
    # 1. CALCULATE APPROXIMATE REYNOLDS NUMBER FOR LOOKUP
    # We estimate local velocity using geometric sum for the Re lookup
    # (Iterative Re refinement is possible but usually overkill for this step)
    V_res_approx = math.sqrt(V**2 + (r * Omega)**2)
    Re_local = rho * V_res_approx * c / mu

    # 2. DEFINE LAMBDAS THAT QUERY THE DB WITH THE CALCULATED Re
    # This allows us to pass a simple f(alpha) to the solver, while maintaining Re dependence
    cl_fn = lambda alpha_rad: airfoil_db.get_coefficients(math.degrees(alpha_rad), Re_local)[0]
    cd_fn = lambda alpha_rad: airfoil_db.get_coefficients(math.degrees(alpha_rad), Re_local)[1]

    # 3. Solve Phi
    phi = solve_phi_regula_falsi(r, V, Omega, sigma, theta_local, cl_fn, cd_fn,
                                R, R_hub, B, alpha_zero_lift=alpha0, pitch_beta=pitch_beta)

    # Calculate loss factors
    F_tip = prandtl_tip_loss(r, R, B, phi)
    F_root = prandtl_root_loss(r, R_hub, B, phi)
    
    alpha = (theta_local + pitch_beta - alpha0) - phi
    
    # Get Final Coefficients
    Cl = cl_fn(alpha)
    Cd = cd_fn(alpha)

    F = F_func(phi, sigma, Cl, Cd, F_tip, F_root)
    G = G_func(phi, sigma, Cl, Cd, F_tip, F_root)
    
    a = ((math.sin(phi) / F) - 1) if abs(F) > 1e-12 else 0.0
    b = (1 - math.cos(phi) / G) if abs(G) > 1e-12 else 0.0
    
    V_axial = V * (1 + a)
    V_s = 2*V_axial - V
    V_tangential = r * Omega * (1 - b)
    Vr = math.sqrt(V_axial**2 + (V_tangential)**2)

    F_total = F_tip * F_root
    dT = 0.5 * B * c * rho * (Vr**2) * (Cl * math.cos(phi) - Cd * math.sin(phi)) * F_total
    dQ = 0.5 * B * c * rho * (Vr**2) * (Cl * math.sin(phi) + Cd * math.cos(phi)) * r * F_total

    diagnostics = dict(
        r=r, phi=phi, alpha=alpha, Cl=Cl, Cd=Cd, sigma=sigma,
        a=a, b=b, F_tip=F_tip, F_root=F_root, dT=dT, dQ=dQ, V_s=V_s, Re=Re_local
    )
    return dT, dQ, diagnostics

# ============ Integration Across Blade ============
def integrate_blade(B, chord_fun, theta_fun, R, R_hub, N, rho, V, rpm, airfoil_db,
                   pitch_beta=0.0, alpha0_fun=None):

    Omega = 2.0 * math.pi * rpm / 60.0
    beta = np.linspace(0, math.pi, N+1)
    nodes = R_hub + (R - R_hub) * (1 - np.cos(beta))/2
    
    dT_nodes, dQ_nodes, diags = [], [], []

    for r in nodes:
        if r <= R_hub + 1e-12:
            dT_nodes.append(0.0); dQ_nodes.append(0.0); diags.append({})
            continue
            
        c = chord_fun(r)
        theta_local = theta_fun(r)
        alpha0 = alpha0_fun(r) if alpha0_fun is not None else 0.0
        
        # Pass airfoil_db here instead of global funcs
        dT, dQ, diagnostics = elemental_forces(
            r, c, B, rho, V, Omega, theta_local, pitch_beta, alpha0,
            airfoil_db=airfoil_db, R_hub=R_hub, R=R
        )
        
        dT_nodes.append(dT)
        dQ_nodes.append(dQ)
        diags.append(diagnostics)

    T_total, Q_total = 0.0, 0.0
    for i in range(N):
        dr = nodes[i+1] - nodes[i]
        T_total += 0.5 * (dT_nodes[i] + dT_nodes[i+1]) * dr
        Q_total += 0.5 * (dQ_nodes[i] + dQ_nodes[i+1]) * dr

    return T_total, Q_total, diags, nodes

# ============ Example Run ============
if __name__ == "__main__":
    
    # 1. SETUP AIRFOIL DATABASE (Runs XFOIL automatically)
    # You can change "0012" to any NACA 4-digit or a filename like "sg6043.dat"
    AIRFOIL_TARGET = "2412" 
    # Create a list of Re to cover your operating range
    RE_RANGE = [50000, 100000, 200000, 500000, 1000000] 
    
    try:
        airfoil_db = build_airfoil_database(AIRFOIL_TARGET, RE_RANGE)
        print("Airfoil database successfully built.")
    except Exception as e:
        print(f"Error building database: {e}")
        exit()

    # 2. SETUP PROPELLER
    print("\n" + "="*60)
    print("BEMT ANALYSIS WITH AUTOMATED XFOIL DATA")
    print("="*60)

    # Propeller parameters
    B = 3
    R = 0.4
    R_hub = 0.02
    rho = 1.225 # Ground
    # rho = 0.909 # Cruise Altitude
    rpm = 5000
    V_start, V_end, step = 0.0, 85.0, 1.0
    N = 10
    A = math.pi * (R**2 )
    n = rpm / 60.0
    D = 2 * R

    chord_fun = lambda r: 0.2
    theta_fun = lambda r: 12 * math.pi / 180
    alpha0_fun = lambda r: 0.0

    print(f"\nPropeller Configuration: R={R}m, RPM={rpm}")
    
    df = pd.DataFrame(columns=['Velocity (m/s)', 'Thrust (N)', 'Torque (Nm)', 'Power (W)',
                               'Efficiency', 'CT', 'CP', 'J'])

    for V in np.arange(V_start, V_end, step):
        J = V / (n * D)
        
        # Call integration with the new airfoil_db
        T, Q, diags, nodes = integrate_blade(
            B, chord_fun, theta_fun, R, R_hub, N, rho, V, rpm,
            airfoil_db=airfoil_db, pitch_beta=0.0, alpha0_fun=alpha0_fun
        )
        
        P = Q * (2.0 * math.pi * n)
        eta = (T * V) / P if P > 1e-6 else 0.0
        CT = T / (rho * n**2 * D**4)
        CP = P / (rho * n**3 * D**5)
        
        # Use concat instead of append (Pandas deprecation fix)
        new_row = pd.DataFrame([{
            'Velocity (m/s)': V, 'Thrust (N)': T, 'Torque (Nm)': Q,
            'Power (W)': P, 'Efficiency': eta, 'CT': CT, 'CP': CP, 'J': J
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        
        print(f"V={V:.1f} m/s | T={T:.2f} N | P={P/1000:.2f} kW")

    # 3. PLOTTING
    # Load comparison data if exists
    exp_path = os.path.join(os.path.dirname(__file__), 'propeller_data.csv')
    has_exp_data = os.path.exists(exp_path)
    
    ax = df.plot(x='J', y='CT', legend=None, style='b-o', label='BEMT (XFOIL)')
    
    if has_exp_data:
        exp_data = pd.read_csv(exp_path, delimiter=' ', header=0)
        exp_data.plot(x='J', y='CT', style='r-s', ax=ax, label='Exp CT')
        exp_data.plot(x='J', y='CP', ax=ax, style='r--', label='Exp CP')
        exp_data.plot(x='J', y='eta', style='m-s', ax=ax.twinx(), label='Exp Eta')

    df.plot(x='J', y='CP', ax=ax, style='g-o', label='BEMT CP')
    
    plt.ylabel('$C_T$ , $C_P$')
    plt.title('BEMT Performance with XFOIL Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    
    df['Thrust_per_unit_Area'] = df['Thrust (N)'] / A
    df.to_csv('bemt_output.csv', index=False)

    
import os
import subprocess
import numpy as np
from scipy.interpolate import interp1d
import math
import shutil

class ViternaExtrapolator:
    """
    Implements the Viterna-Corrigan method to extrapolate airfoil polars
    to a full 360-degree range.
    """
    def __init__(self, alpha, cl, cd, cr75=1.0):
        """
        :param alpha: Array of angles of attack (degrees)
        :param cl: Array of lift coefficients
        :param cd: Array of drag coefficients
        :param cr75: Chord/Radius ratio at 75% span (aspect ratio correction), default 1.0 for 2D
        """
        self.alpha = np.radians(alpha)
        self.cl = cl
        self.cd = cd
        self.cr75 = cr75
        self.AR = 1e6 # Assumed infinite for 2D XFOIL data initially

    def extrapolate(self):
        # 1. Identify Stall (Max Cl and Min Cl)
        idx_max = np.argmax(self.cl)
        idx_min = np.argmin(self.cl)
        
        alpha_stall_pos = self.alpha[idx_max]
        cl_stall_pos = self.cl[idx_max]
        cd_stall_pos = self.cd[idx_max]
        
        alpha_stall_neg = self.alpha[idx_min]
        cl_stall_neg = self.cl[idx_min]
        cd_stall_neg = self.cd[idx_min]

        # 2. Define Flat Plate parameters (approximate for typical airfoils)
        # Cd_max approx 1.11 -> 2.0. Using 1.3 as a generic mean for airfoils
        cd_max = 1.11 + 0.018 * self.AR if self.AR < 50 else 2.01 
        
        # 3. Create full range (-180 to 180)
        alpha_full = np.linspace(-np.pi, np.pi, 361)
        cl_full = np.zeros_like(alpha_full)
        cd_full = np.zeros_like(alpha_full)

        for i, a in enumerate(alpha_full):
            # Check if within original XFOIL range
            if alpha_stall_neg <= a <= alpha_stall_pos:
                # Interpolate from existing data
                cl_full[i] = np.interp(a, self.alpha, self.cl)
                cd_full[i] = np.interp(a, self.alpha, self.cd)
            else:
                # Apply Viterna method
                # Coefficients depend on positive or negative side
                if a > alpha_stall_pos:
                    # Positive Stall Side
                    A1 = cd_max / 2
                    B1 = cd_max
                    
                    # Solve for A2 and B2 to ensure continuity at stall
                    # Cl = A1 sin(2a) + A2 * (cos(a)^2 / sin(a))
                    # Cd = B1 sin(a)^2 + B2 cos(a)
                    
                    A2 = (cl_stall_pos - A1 * math.sin(2*alpha_stall_pos)) * math.sin(alpha_stall_pos) / (math.cos(alpha_stall_pos)**2)
                    B2 = (cd_stall_pos - B1 * math.sin(alpha_stall_pos)**2) / math.cos(alpha_stall_pos)
                    
                    cl_full[i] = A1 * math.sin(2*a) + A2 * (math.cos(a)**2) / math.sin(a)
                    cd_full[i] = B1 * (math.sin(a)**2) + B2 * math.cos(a)
                    
                elif a < alpha_stall_neg:
                    # Negative Stall Side (Symmetric logic)
                    A1 = -cd_max / 2
                    B1 = cd_max
                    
                    A2 = (cl_stall_neg - A1 * math.sin(2*alpha_stall_neg)) * math.sin(alpha_stall_neg) / (math.cos(alpha_stall_neg)**2)
                    B2 = (cd_stall_neg - B1 * math.sin(alpha_stall_neg)**2) / math.cos(alpha_stall_neg)
                    
                    cl_full[i] = A1 * math.sin(2*a) + A2 * (math.cos(a)**2) / math.sin(a)
                    cd_full[i] = B1 * (math.sin(a)**2) + B2 * math.cos(a)

        # Smooth clamp Cl near +/- 180 to 0
        return np.degrees(alpha_full), cl_full, cd_full

class XFoilRunner:
    def __init__(self, xfoil_path="xfoil"):
        self.xfoil_path = xfoil_path
    
    def run(self, airfoil_name, Re, Mach=0.0, alpha_start=-10, alpha_end=15, step=1.0, n_iter=100):
        """
        Runs XFOIL and returns parsed polar data.
        """
        # Determine if airfoil is a file or a NACA code
        is_file = os.path.exists(airfoil_name)
        output_file = f"{airfoil_name}_Re{Re:.1e}.txt".replace(".dat", "")
        
        # Remove previous output if exists
        if os.path.exists(output_file):
            os.remove(output_file)

        # Construct Input Commands
        cmds = []
        if is_file:
            cmds.append(f"LOAD {airfoil_name}")
        else:
            cmds.append(f"NACA {airfoil_name}")
        
        cmds.append("OPER")
        cmds.append(f"Visc {Re}")
        cmds.append(f"M {Mach}")
        cmds.append(f"ITER {n_iter}")
        cmds.append("PACC")
        cmds.append(output_file) # Output filename
        cmds.append("")          # No dump file
        
        # Sequence
        cmds.append(f"ASEQ {alpha_start} {alpha_end} {step}")
        cmds.append("") # Break
        cmds.append("QUIT")
        
        # Run XFOIL
        try:
            input_str = "\n".join(cmds)
            process = subprocess.Popen(
                self.xfoil_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(input=input_str)
        except FileNotFoundError:
            raise FileNotFoundError("XFOIL executable not found. Please add to path or set 'xfoil_path'.")

        # Parse Output
        polar = self._parse_polar(output_file) 
        
        # Remove previous output if exists
        if os.path.exists(output_file):
            os.remove(output_file)
            
        return polar

    def _parse_polar(self, filepath):
        if not os.path.exists(filepath):
            print(f"Warning: No output file generated for {filepath}. XFOIL might have diverged.")
            return None, None, None

        alphas, cls, cds = [], [], []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            start_reading = False
            for line in lines:
                if "------" in line:
                    start_reading = True
                    continue
                if start_reading:
                    # XFOIL format usually: alpha CL CD CDp CM Top_Xer ...
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            alphas.append(float(parts[0]))
                            cls.append(float(parts[1]))
                            cds.append(float(parts[2]))
                        except ValueError:
                            continue
        return np.array(alphas), np.array(cls), np.array(cds)

class AirfoilDatabase:
    """
    Stores interpolated functions for different Reynolds numbers.
    """
    def __init__(self, airfoil_name):
        self.name = airfoil_name
        # structure: { re_num: { 'cl_func': interp1d, 'cd_func': interp1d } }
        self.data = {} 
        self.re_keys = []

    def add_polar(self, re, alpha, cl, cd):
        # Store the raw interpolation functions for this specific Reynolds
        # We assume alpha is already 360 extrapolated here
        self.data[re] = {
            'cl_func': interp1d(alpha, cl, kind='linear', fill_value="extrapolate"),
            'cd_func': interp1d(alpha, cd, kind='linear', fill_value="extrapolate")
        }
        self.re_keys = sorted(self.data.keys())

    def get_coefficients(self, aoa_deg, re_target):
        """
        BEMT SOLVER INTERFACE
        1. Interpolates Cl/Cd for the specific AoA at available Reynolds numbers.
        2. Interpolates between Reynolds numbers to get final Cl/Cd.
        """
        if not self.re_keys:
            raise ValueError("Database is empty.")

        # Clamp AoA to -180 to 180 just in case
        aoa_deg = ((aoa_deg + 180) % 360) - 180

        # Case 1: Re_target is outside our range (Clamp to nearest)
        if re_target <= self.re_keys[0]:
            cl = self.data[self.re_keys[0]]['cl_func'](aoa_deg)
            cd = self.data[self.re_keys[0]]['cd_func'](aoa_deg)
            return float(cl), float(cd)
        
        if re_target >= self.re_keys[-1]:
            cl = self.data[self.re_keys[-1]]['cl_func'](aoa_deg)
            cd = self.data[self.re_keys[-1]]['cd_func'](aoa_deg)
            return float(cl), float(cd)

        # Case 2: Interpolate between two Reynolds numbers
        # Find Re_lower and Re_upper
        idx = np.searchsorted(self.re_keys, re_target)
        re_lower = self.re_keys[idx - 1]
        re_upper = self.re_keys[idx]

        # Get values at both Re
        cl_low = self.data[re_lower]['cl_func'](aoa_deg)
        cd_low = self.data[re_lower]['cd_func'](aoa_deg)
        
        cl_high = self.data[re_upper]['cl_func'](aoa_deg)
        cd_high = self.data[re_upper]['cd_func'](aoa_deg)

        # Linear interpolation based on log(Re) is often physically better, 
        # but linear Re is acceptable for close steps. Let's use linear Re.
        fraction = (re_target - re_lower) / (re_upper - re_lower)
        
        cl_final = cl_low + fraction * (cl_high - cl_low)
        cd_final = cd_low + fraction * (cd_high - cd_low)

        return float(cl_final), float(cd_final)

# ==========================================
# Main Workflow / Builder
# ==========================================

def build_airfoil_database(airfoil_identifier, re_list, mach=0.0):
    """
    Orchestrates the XFOIL run -> Extrapolation -> Storage
    """
    xfoil_path = "./xfoil.exe"
    print(f"--- Processing Airfoil: {airfoil_identifier} ---")
    db = AirfoilDatabase(airfoil_identifier)
    runner = XFoilRunner(xfoil_path) # Ensure 'xfoil' is in PATH or provide path here

    for re in re_list:
        print(f"Running XFOIL for Re={re}...")
        
        # 1. Run XFOIL (Standard range, XFOIL usually converges -10 to 15 best)
        # We scan a bit wider to catch stall
        alpha_raw, cl_raw, cd_raw = runner.run(
            airfoil_identifier, re, Mach=mach, 
            alpha_start=-10, alpha_end=20, step=0.5, n_iter=200
        )

        if alpha_raw is None or len(alpha_raw) < 5:
            print(f"  -> Failed to converge or generate data for Re={re}")
            continue

        # 2. Extrapolate to 360 degrees
        viterna = ViternaExtrapolator(alpha_raw, cl_raw, cd_raw)
        alpha_360, cl_360, cd_360 = viterna.extrapolate()

        # 3. Add to Database
        db.add_polar(re, alpha_360, cl_360, cd_360)
        print(f"  -> Processed and Extrapolated.")

    return db

# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":
    # 1. Define parameters
    # Can be "0012" (NACA) or "my_airfoil.dat"
    airfoil_name = "2412" 
    # airfoil_name = "clarky_airfoil.dat"  # Example file, ensure it exists in the same directory
    reynolds_numbers = [1000, 100000, 500000, 1000000]
    
    # 2. Build the database (Runs XFOIL automatically)
    try:
        my_airfoil_db = build_airfoil_database(airfoil_name, reynolds_numbers)
        
        # 3. Simulate BEMT Solver calls
        print("\n--- Testing BEMT Solver Interface ---")
        
        test_conditions = [
            (5.0, 100000),   # Exact Re match
            (12.5, 750000),  # Re Interpolation required (stall region)
            (95.0, 100000),  # Post-stall (Extrapolated region)
            (-170.0, 200000) # Deep post-stall negative
        ]

        print(f"{'AoA (deg)':<10} | {'Re':<10} | {'Cl':<10} | {'Cd':<10}")
        print("-" * 50)
        
        for aoa, re in test_conditions:
            cl, cd = my_airfoil_db.get_coefficients(aoa, re)
            print(f"{aoa:<10.1f} | {re:<10.0f} | {cl:<10.4f} | {cd:<10.4f}")

    except Exception as e:
        print(f"\nError: {e}")
        print("Ensure 'xfoil' is installed and in your system PATH.")
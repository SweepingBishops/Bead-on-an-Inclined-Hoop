import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter

def process_and_plot(t_data, theta_data, omega):
    """
    Phase 4: Smooth the raw trajectory data and compute/plot kinematic variables.
    """
    if len(t_data) < 10:
        print("\nError: Not enough data points to perform meaningful analysis.")
        return

    print("\n--- Phase 4.1: Data Smoothing ---")
    # Savitzky-Golay filter requires an odd window length.
    # We dynamically size it based on how many frames we captured.
    window_length = min(len(theta_data) // 4 * 2 + 1, 51) 
    if window_length < 5:
        window_length = 5
    poly_order = 3

    print(f"Applying Savitzky-Golay filter (Window: {window_length}, Polynomial Order: {poly_order})")
    
    # Smooth the theta data
    theta_smoothed = savgol_filter(theta_data, window_length, poly_order)

    print("--- Phase 4.2: Kinematic Computation ---")
    # Calculate angular velocity (d_theta / dt) using numpy's gradient function.
    # Note: theta is currently in degrees, so velocity is degrees/second.
    # Let's convert theta to radians for standard physics units.
    theta_rad = np.deg2rad(theta_data)
    theta_smoothed_rad = np.deg2rad(theta_smoothed)
    
    # d_theta / dt in radians per second (Velocity relative to the hoop)
    omega_rel = np.gradient(theta_smoothed_rad, t_data)

    print("--- Phase 4.3: Graphical Output ---")
    print("Generating kinematic plots...")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Angular Position (Theta) vs Time
    ax1.plot(t_data, theta_rad, '.', color='black', label='Raw Data', alpha=0.9)
    ax1.plot(t_data, theta_smoothed_rad, '-', color='blue', label='Smoothed Data', linewidth=2)
    ax1.set_ylabel(r"Angular Position $\theta$ (rad)", fontsize=12)
    ax1.set_title("Bead Kinematics on a Rotating Hoop", fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Angular Velocity (dTheta/dt) vs Time
    ax2.plot(t_data, omega_rel, '-', color='red', label=r'Velocity $\dot{\theta}$', linewidth=2)
    

    ax2.set_xlabel("Time $t$ (seconds)", fontsize=12)
    ax2.set_ylabel(r"Angular Velocity $\dot{\theta}$ (rad/s)", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    print("\n==========================================")
    print("         FULL PIPELINE COMPLETE           ")
    print("==========================================\n")
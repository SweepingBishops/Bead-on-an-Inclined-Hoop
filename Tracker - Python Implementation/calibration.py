import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import optimize
import os
import json
import glob
from datetime import datetime

def get_fps(cap):
    """
    Phase 1.1: Extract the precise frame rate from the video metadata.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def get_spatial_calibration(cap):
    """
    Phase 1.2: Ask the user to manually mark the 5-degree red marks, 
    associating each click with a specific angle.
    """
    # Ensure we are reading the very first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, original_frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame for calibration.")
        return None

    calibration_data = {}  # Dictionary to store {angle: (x, y)}
    
    # State variables
    current_angle = 0
    phase = 1  # Phase 1 for positive angles, Phase 2 for negative angles

    def update_display():
        """Helper function to redraw the frame with instructions and points."""
        display_frame = original_frame.copy()
        
        # Draw all the points we have collected so far
        for ang, pt in calibration_data.items():
            cv2.circle(display_frame, pt, 4, (0, 0, 255), -1)
            # Add small text next to the point showing its angle
            cv2.putText(display_frame, str(ang), (pt[0]+5, pt[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw the main instruction prompt at the top of the screen
        if phase == 1:
            msg = f"Phase 1: Click mark for {current_angle} deg. (Press 'q' to switch to negative)"
        else:
            msg = f"Phase 2: Click mark for {current_angle} deg. (Press 'q' to finish)"
            
        cv2.putText(display_frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Spatial Calibration", display_frame)

    def click_event(event, x, y, flags, param):
        nonlocal current_angle, phase # Allows us to modify the outer state variables
        
        if event == cv2.EVENT_LBUTTONDOWN:
            calibration_data[current_angle] = (x, y)
            print(f"Recorded {current_angle} degrees at pixel: ({x}, {y})")
            
            # Auto-increment or decrement for the next click
            if phase == 1:
                current_angle += 5
            elif phase == 2:
                current_angle -= 5
                
            update_display()

    # Initial setup for the window and callback
    cv2.namedWindow("Spatial Calibration")
    cv2.setMouseCallback("Spatial Calibration", click_event)
    update_display()

    print("\n--- Spatial Calibration ---")
    
    # Wait loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if phase == 1:
                # Switch to Phase 2 (negative angles)
                phase = 2
                current_angle = -5
                print("\n--- Switching to negative angles ---")
                update_display()
            else:
                # End calibration
                print("\n--- Calibration finished ---")
                break

    cv2.destroyAllWindows()
    
    # Sort the dictionary by angle so the final output is in logical order
    sorted_calibration = dict(sorted(calibration_data.items()))
    return sorted_calibration



def create_interpolation_map(calibration_data):
    """
    Phase 1.3: Creates a function mapping raw pixel coordinates to true angular positions.
    """
    if not calibration_data:
        print("Error: No calibration data provided.")
        return None

    # 1. Extract data (already sorted by true angle in step 1.2)
    true_angles = np.array(list(calibration_data.keys()))
    points = np.array(list(calibration_data.values()))
    x = points[:, 0]
    y = points[:, 1]

    # 2. Fit a circle to find the hoop's center (xc, yc)
    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    # Initial guess for the center: the mean of x and y
    center_estimate = (np.mean(x), np.mean(y))
    center_fit, _ = optimize.leastsq(f_2, center_estimate)
    xc, yc = center_fit
    print(f"Calculated Hoop Center at pixel: ({xc:.1f}, {yc:.1f})")

    # 3. Calculate observed angles (in radians) for each calibration point
    # We use unwrap to prevent the angle from jumping from pi to -pi
    observed_angles = np.unwrap(np.arctan2(y - yc, x - xc))

    # CubicSpline requires the input (observed_angles) to be strictly increasing.
    # Depending on camera orientation, they might be strictly decreasing.
    sort_idx = np.argsort(observed_angles)
    observed_angles = observed_angles[sort_idx]
    true_angles = true_angles[sort_idx]

    # 4. Create the interpolation spline
    spline = CubicSpline(observed_angles, true_angles)

    # 5. Define a helper function to convert ANY future (x, y) into a true angle
    def get_true_angle(px, py):
        # Calculate the raw angle of the tracked pixel relative to the hoop center
        obs_angle = np.arctan2(py - yc, px - xc)
        
        # Handle the same pi/-pi wrapping for the tracked point
        # We align it with the mean of our calibration observed angles
        mean_obs = np.mean(observed_angles)
        while obs_angle - mean_obs > np.pi:
            obs_angle -= 2 * np.pi
        while obs_angle - mean_obs < -np.pi:
            obs_angle += 2 * np.pi
            
        return float(spline(obs_angle))

    return get_true_angle



def save_calibration(calibration_data):
    """Saves the calibration dictionary to a timestamped JSON file."""
    folder = "calibration_data"
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    # Generate a filename with the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"calib_{timestamp}.json"
    filepath = os.path.join(folder, filename)
    
    with open(filepath, 'w') as f:
        json.dump(calibration_data, f, indent=4)
        
    print(f"\nSaved new calibration data to: {filepath}")

def load_calibration(filepath):
    """Loads a calibration dictionary from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # JSON converts integer keys to strings, so we convert them back to integers
    # and lists back to tuples for compatibility with our code
    calibration_data = {int(k): tuple(v) for k, v in data.items()}
    return calibration_data

def select_calibration_file():
    """Provides a terminal UI to select an existing calibration file."""
    folder = "calibration_data"
    if not os.path.exists(folder):
        print("No 'calibration_data' folder found.")
        return None
        
    files = glob.glob(os.path.join(folder, "*.json"))
    if not files:
        print("No calibration files found in the folder.")
        return None
        
    print("\n--- Available Calibration Files ---")
    for i, f in enumerate(files):
        print(f"[{i + 1}] {os.path.basename(f)}")
        
    while True:
        try:
            choice = input(f"Select a file number (1-{len(files)}) or '0' to cancel: ")
            choice_idx = int(choice)
            
            if choice_idx == 0:
                return None
            if 1 <= choice_idx <= len(files):
                filepath = files[choice_idx - 1]
                print(f"Loading {os.path.basename(filepath)}...")
                return load_calibration(filepath)
            print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
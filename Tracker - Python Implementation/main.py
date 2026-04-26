import argparse
import cv2
import random
import calibration
import viz
import hoop_tracking
import os
import bead_tracking
import analysis
import numpy as np

# Creating an output directory for results if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    parser = argparse.ArgumentParser(description="Analyze kinematics of a bead on a rotating hoop.")
    parser.add_argument("video_file", help="Path to the input video file (.mp4)")
    args = parser.parse_args()

    video_path = args.video_file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # --- Phase 1: Calibration ---
    fps = calibration.get_fps(cap)
    print(f"Video FPS: {fps}")

    print("\n--- Calibration Setup ---")
    print("[1] Create new calibration (manual clicking)")
    print("[2] Load existing calibration file")
    
    user_choice = input("Enter your choice (1 or 2): ").strip()
    
    calibration_data = None
    
    # Try loading an existing file if the user chose 2
    if user_choice == '2':
        calibration_data = calibration.select_calibration_file()
        
    # If the user chose 1, OR if they chose 2 but canceled/no files existed
    if not calibration_data:
        if user_choice == '2':
            print("\nFalling back to manual calibration...")
        calibration_data = calibration.get_spatial_calibration(cap)
        # Save the newly created data automatically
        if calibration_data:
            calibration.save_calibration(calibration_data)
    
    # Proceed with interpolation mapping and visualization if we successfully got data
    if calibration_data:
        angle_calculator = calibration.create_interpolation_map(calibration_data)
        
        # --- Phase 1.4: Visualization Sanity Check ---
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            random_frame_idx = random.randint(0, total_frames - 1)
            print(f"\nJumping to random frame {random_frame_idx} for visual check...")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
            ret, random_frame = cap.read()
            
            if ret:
                viz.verify_interpolation_curve(random_frame, calibration_data, angle_calculator)
            else:
                print("Error: Could not read the random frame.")
        

        # --- Phase 2: Hoop Velocity ---
        hoop_tracking_out = hoop_tracking.get_angular_velocity(cap, fps)
        omega = hoop_tracking_out['mean_omega']

        save_inp = input("Do you want to save the angular velocity results to a text file? (y/n): ").strip().lower()
        if save_inp == 'y' and hoop_tracking_out['mean_omega'] is not None:
            filename = os.path.join(output_dir, "angular_velocity_" + os.path.basename(video_path).split('.')[0] + ".txt")
            with open(filename, "w") as f:
                f.write(f"Angular Velocity : {hoop_tracking_out['mean_omega']:.3f} \n")
                f.write(f"Error : {hoop_tracking_out['error_omega']:.3f} \n")
            print(f"Angular velocity saved to {filename}")


        t_data, theta_data = bead_tracking.track_bead(cap, fps, angle_calculator, calibration_data)
        if t_data is not None and len(t_data) > 0:
                raw_data = np.column_stack((t_data, theta_data))
                filename = os.path.join(output_dir, "raw_bead_data_" + os.path.basename(video_path).split('.')[0] + ".csv")
                np.savetxt(filename, raw_data, delimiter=",", header="Time_s,Raw_Theta_deg", comments="")
                print(f"\nSaved {len(t_data)} raw data points to '{filename}'")
        
        # --- Phase 4: Data Processing (Placeholder) ---
        analysis.process_and_plot(t_data, theta_data, omega)

    cap.release()
    print("\nAnalysis complete.")
        

if __name__ == "__main__":
    main()
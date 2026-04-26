import cv2
import numpy as np

def get_angular_velocity(cap, fps):
    """
    Phase 2: Digital Tripwire ROI tracking for periodicity measurement.
    """
    print("\n--- Phase 2.1: Video Player & Frame Selection ---")
    print("Scrub to a frame where your unique colored object is clearly visible.")
    print("Press 's' to select this frame.")

    window_name = "Hoop Tracking Player"
    cv2.namedWindow(window_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def on_trackbar(val): pass
    cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackbar)
    
    # --- Step 1: Scrubbing ---
    selected_frame = None
    start_frame_idx = 0
    while True:
        current_pos = cv2.getTrackbarPos("Frame", window_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        ret, frame = cap.read()
        if not ret: break
            
        display_frame = frame.copy()
        cv2.putText(display_frame, "Scrub to target frame. Press 's' to select.", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(30) & 0xFF == ord('s'):
            selected_frame = frame.copy()
            start_frame_idx = current_pos
            break

    if selected_frame is None:
        print("Selection cancelled.")
        cv2.destroyAllWindows()
        return None

    # --- Step 2: ROI & Color Selection ---
    print("\n--- Phase 2.2: ROI Selection ---")
    print("Draw a box around the colored object. Make sure the object fills most of the box.")
    print("Press SPACE or ENTER to confirm the ROI, or 'c' to cancel.")
    
    roi = cv2.selectROI(window_name, selected_frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    
    if w == 0 or h == 0:
        print("Invalid ROI selected.")
        cv2.destroyAllWindows()
        return None

    # Extract the target color properties (using median HSV of the ROI)
    roi_patch = selected_frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2HSV)
    median_hsv = np.median(hsv_roi, axis=(0, 1))
    
    # Extract the Hue value as a standard integer
    h_val = int(median_hsv[0])
    
    # Create an HSV color range, explicitly forcing the uint8 data type
    lower_color = np.array([max(0, h_val - 15), 50, 50], dtype=np.uint8)
    upper_color = np.array([min(179, h_val + 15), 255, 255], dtype=np.uint8)
    
    total_pixels = w * h

    # --- Step 3: Tracking Loop ---
    print("\n--- Phase 2.3: Tracking Laps ---")
    print("Tracking active... Press 'q' to stop early and calculate.")

    lap_times = []
    last_lap_frame = start_frame_idx
    current_frame_idx = start_frame_idx
    
    # State machine: True = Object is inside ROI (Green), False = Object is gone (Red)
    object_present = True  
    box_color = (0, 255, 0) # Start Green
    delay = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        current_frame_idx += 1
        display_frame = frame.copy()
        
        # Analyze the specific ROI in the current frame
        current_roi = frame[y:y+h, x:x+w]
        hsv_current = cv2.cvtColor(current_roi, cv2.COLOR_BGR2HSV)
        
        # Create a mask of pixels that match our target color
        mask = cv2.inRange(hsv_current, lower_color, upper_color)
        matching_pixels = cv2.countNonZero(mask)
        fill_percentage = matching_pixels / total_pixels
        
        # State Machine Logic
        if object_present:
            # If the object leaves (drops below 20% presence)
            if fill_percentage < 0.20:
                object_present = False
                box_color = (0, 0, 255) # Turn Red
        else:
            # If the object returns (goes above 50% presence)
            if fill_percentage >= 0.50:
                object_present = True
                box_color = (0, 255, 0) # Turn Green
                
                # Mark the lap
                lap_frames = current_frame_idx - last_lap_frame
                lap_time = lap_frames / fps
                lap_times.append(lap_time)
                
                print(f"Lap marked! Time: {lap_time:.3f} sec")
                last_lap_frame = current_frame_idx

        # Draw the visual feedback
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, 2)
        status_text = "Tracking..." if object_present else "Waiting for return..."
        cv2.putText(display_frame, status_text, (x, max(10, y-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # --- Step 4: Statistical Output ---
    if not lap_times:
        print("\nWarning: No complete laps detected.")
        return None

    lap_times_arr = np.array(lap_times)
    omegas_arr = (2 * np.pi) / lap_times_arr

    # Subsection 1: Means and Std Devs
    mean_T = np.mean(lap_times_arr)
    std_T = np.std(lap_times_arr)
    mean_omega = np.mean(omegas_arr)
    std_omega = np.std(omegas_arr)

    # Subsection 2: Median Statistics
    median_T = round(float(np.median(lap_times_arr)), 3)
    rounded_laps = np.round(lap_times_arr, 3)
    median_count = np.sum(rounded_laps == median_T)
    median_omega = (2 * np.pi) / median_T

    print("\n==========================================")
    print("           TRACKING RESULTS               ")
    print("==========================================")
    print("--- Subsection 1: Averages ---")
    print(f"Mean Time Period (T):     {mean_T:.3f} s  (Std Dev: {std_T:.3f} s)")
    print(f"Mean Angular Vel (Omega): {mean_omega:.3f} rad/s  (Std Dev: {std_omega:.3f} rad/s)")
    
    print("\n--- Subsection 2: Medians ---")
    print(f"Median Time Period (T):   {median_T:.3f} s")
    print(f"Occurrences of Median T:  {median_count}/{len(lap_times_arr)} laps")
    print(f"Median Angular Vel:       {median_omega:.3f} rad/s")
    print("==========================================\n")
    
    # We return the mean omega for the final physics calculations, 
    # but you could easily change this to return the median_omega if you prefer!
    return {"mean_omega": mean_omega, "error_omega": std_omega}
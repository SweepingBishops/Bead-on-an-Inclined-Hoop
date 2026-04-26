import cv2
import numpy as np
from scipy import optimize

def track_bead(cap, fps, angle_calculator, calibration_data):
    """
    Phase 3: Isolate the bead using an Adjustable Donut Mask + Color Thresholding.
    """
    # --- Step 0: Calculate Base Hoop Geometry ---
    points = np.array(list(calibration_data.values()))
    px = points[:, 0]
    py = points[:, 1]

    def calc_R(xc, yc):
        return np.sqrt((px - xc)**2 + (py - yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = (np.mean(px), np.mean(py))
    center_fit, _ = optimize.leastsq(f_2, center_estimate)
    base_xc, base_yc = int(center_fit[0]), int(center_fit[1])
    base_R = int(calc_R(base_xc, base_yc).mean())

    # --- Step 1: Scrubbing to a good frame ---
    print("\n--- Phase 3.1: Frame Selection ---")
    print("Scrub to a frame where the bead is clearly visible.")
    print("Press 's' to select this frame.")

    window_name = "Bead Tracking"
    cv2.namedWindow(window_name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def on_trackbar(val): pass
    cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, on_trackbar)
    
    selected_frame = None
    start_frame_idx = 0
    while True:
        current_pos = cv2.getTrackbarPos("Frame", window_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        ret, frame = cap.read()
        if not ret: break
            
        display_frame = frame.copy()
        cv2.putText(display_frame, "Scrub to bead. Press 's' to select.", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(30) & 0xFF == ord('s'):
            selected_frame = frame.copy()
            start_frame_idx = current_pos
            break

    if selected_frame is None:
        cv2.destroyAllWindows()
        return None, None

    # --- Step 2: Adjust the Donut Mask ---
    print("\n--- Phase 3.2: Donut Mask Adjustment ---")
    print("Use the trackbars to align the green donut mask perfectly over the hoop track.")
    print("Press 'a' (Accept) when you are happy with the mask.")

    cv2.destroyWindow(window_name) # Refresh window to clean up trackbars
    adjust_window = "Adjust Donut Mask"
    cv2.namedWindow(adjust_window)

    # Trackbars (Start at 50 so we can have negative and positive offsets)
    cv2.createTrackbar("X Offset", adjust_window, 50, 100, on_trackbar)
    cv2.createTrackbar("Y Offset", adjust_window, 50, 100, on_trackbar)
    cv2.createTrackbar("Radius Adj", adjust_window, 50, 100, on_trackbar)
    cv2.createTrackbar("Thickness", adjust_window, 60, 150, on_trackbar)

    final_xc, final_yc, final_R, final_thickness = base_xc, base_yc, base_R, 60

    while True:
        # Calculate dynamic values based on trackbars
        dx = cv2.getTrackbarPos("X Offset", adjust_window) - 50
        dy = cv2.getTrackbarPos("Y Offset", adjust_window) - 50
        dr = cv2.getTrackbarPos("Radius Adj", adjust_window) - 50
        final_thickness = max(1, cv2.getTrackbarPos("Thickness", adjust_window))

        final_xc = base_xc + dx
        final_yc = base_yc + dy
        final_R = base_R + dr

        display_frame = selected_frame.copy()
        
        # Create a blank overlay to draw the green donut
        overlay = np.zeros_like(display_frame)
        cv2.circle(overlay, (final_xc, final_yc), final_R, (0, 255, 0), thickness=final_thickness)
        
        # Blend the green donut onto the frame with 40% transparency
        cv2.addWeighted(overlay, 0.4, display_frame, 1.0, 0, display_frame)
        
        cv2.putText(display_frame, "Adjust mask. Press 'a' to accept.", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(adjust_window, display_frame)
        if cv2.waitKey(30) & 0xFF == ord('a'):
            break

    cv2.destroyWindow(adjust_window)

    # Pre-calculate the final Donut Mask image (black & white for the computer to use)
    frame_h, frame_w = selected_frame.shape[:2]
    donut_mask_img = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.circle(donut_mask_img, (final_xc, final_yc), final_R, 255, thickness=final_thickness)

    # --- Step 3: Sampling the Bead's Color ---
    print("\n--- Phase 3.3: Color Calibration ---")
    print("Click on the bead several times to sample its color profile. Press 'c' to confirm.")
    
    cv2.namedWindow(window_name)
    pristine_frame = selected_frame.copy() 
    clicks = []
    
    def click_bead(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(selected_frame, (x, y), 2, (0, 255, 255), -1)
            cv2.imshow(window_name, selected_frame)
            
    cv2.setMouseCallback(window_name, click_bead)
    cv2.imshow(window_name, selected_frame)
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('c') and len(clicks) > 0:
            break

    hsv_frame = cv2.cvtColor(pristine_frame, cv2.COLOR_BGR2HSV)
    hsv_samples = [hsv_frame[y, x] for x, y in clicks]
    
    h_vals = [int(s[0]) for s in hsv_samples]
    s_vals = [int(s[1]) for s in hsv_samples]
    v_vals = [int(s[2]) for s in hsv_samples]
    
    lower_color = np.array([max(0, min(h_vals)-15), max(30, min(s_vals)-50), max(30, min(v_vals)-50)], dtype=np.uint8)
    upper_color = np.array([min(179, max(h_vals)+15), 255, 255], dtype=np.uint8)

    # --- Step 4: Tracking Loop ---
    print("\n--- Phase 3.4: Tracking Trajectory ---")
    print("Tracking bead... Press 'q' to stop early.")
    
    time_data = []
    theta_data = []
    current_frame_idx = start_frame_idx
    delay = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        current_frame_idx += 1
        display_frame = frame.copy()
        
        # Apply the user-adjusted Donut Mask FIRST
        masked_frame = cv2.bitwise_and(frame, frame, mask=donut_mask_img)
        
        # Color Masking 
        hsv_current = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_current, lower_color, upper_color)
        
        kernel = np.ones((3,3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                try:
                    theta = angle_calculator(cx, cy)
                    t = (current_frame_idx - start_frame_idx) / fps
                    
                    time_data.append(t)
                    theta_data.append(theta)
                    
                    cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(display_frame, f"{theta:.1f} deg", (cx + 10, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except ValueError:
                    pass
                    
        cv2.imshow(window_name, display_frame)
        cv2.imshow("Mask Debug", cv2.resize(color_mask, (0,0), fx=0.3, fy=0.3))
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
    print(f"\n--- Phase 3 Results ---")
    print(f"Successfully tracked bead for {len(time_data)} frames.")
    
    return np.array(time_data), np.array(theta_data)
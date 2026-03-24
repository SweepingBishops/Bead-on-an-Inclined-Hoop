import cv2
import numpy as np
from scipy import optimize

def verify_interpolation_curve(frame, calibration_data, angle_calculator):
    """
    Visualizes the interpolated curve on a random frame and shows the true angle on hover.
    """
    # 1. Re-calculate the center and radius to draw the perfect geometric curve
    points = np.array(list(calibration_data.values()))
    x = points[:, 0]
    y = points[:, 1]

    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = (np.mean(x), np.mean(y))
    center_fit, _ = optimize.leastsq(f_2, center_estimate)
    xc, yc = center_fit
    R = calc_R(xc, yc).mean()

    # 2. Draw the continuous curve
    display_frame = frame.copy()
    
    # Calculate pixel angles to know where to draw the arc
    angles_rad = np.arctan2(y - yc, x - xc)
    sort_idx = np.argsort(angles_rad)
    sorted_angles = angles_rad[sort_idx]
    
    # Generate points along the arc to draw a smooth yellow curve
    curve_points = []
    for ang in np.linspace(sorted_angles[0], sorted_angles[-1], 200):
        px = int(xc + R * np.cos(ang))
        py = int(yc + R * np.sin(ang))
        curve_points.append((px, py))
        
    for i in range(len(curve_points) - 1):
        cv2.line(display_frame, curve_points[i], curve_points[i+1], (0, 255, 255), 2)
        
    # Draw the original marked calibration points in red
    for pt in points:
        cv2.circle(display_frame, tuple(pt), 4, (0, 0, 255), -1)

    print("\n--- Phase 1.4: Visualization Check ---")
    print("Hover over the yellow curve to see the interpolated angle.")
    print("Press 'q' to close this window and continue to the next phase.")

    # 3. Mouse Callback for Hover
    window_name = "Interpolation Check (Hover to see angle)"
    cv2.namedWindow(window_name)

    # We use a base frame so the text doesn't smear infinitely as we move the mouse
    base_frame = display_frame.copy()

    def hover_event(event, mx, my, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            temp_frame = base_frame.copy()
            
            # Check if mouse is relatively close to the hoop's radius
            dist = np.sqrt((mx - xc)**2 + (my - yc)**2)
            if abs(dist - R) < 30:  # 30 pixels of tolerance for hovering
                try:
                    # Feed the cursor's (x, y) into our Phase 1.3 interpolator
                    angle = angle_calculator(mx, my)
                    text = f"{angle:.1f} deg"
                    
                    # Draw the angle text and a green dot near the cursor
                    cv2.putText(temp_frame, text, (mx + 15, my - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.circle(temp_frame, (mx, my), 4, (0, 255, 0), -1)
                except ValueError:
                    # Handles cases where the cursor is outside the spline's data range
                    pass
            
            cv2.imshow(window_name, temp_frame)

    cv2.setMouseCallback(window_name, hover_event)
    cv2.imshow(window_name, base_frame)

    # Wait loop until 'q' is pressed
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
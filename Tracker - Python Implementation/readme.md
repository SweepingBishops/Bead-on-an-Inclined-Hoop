# Rotating Hoop Kinematics Analyzer

A PyQt6-based computer vision application designed to track, analyze, and visualize the kinematics of a bead on a rotating hoop. Built with OpenCV, NumPy, SciPy, and Matplotlib, this tool processes standard video files to extract highly accurate position (theta), angular velocity (omega), and phase space data.

## Features
* **Phase 1: Spatial Calibration**: Click-to-map coordinate system with a built-in magnifying glass for pixel-perfect accuracy. Save and load camera profiles via JSON.
* **Phase 2: Hoop Velocity Tracking**: Calculate the hoop's driving angular velocity using a robust, hysteresis-based color tracking state machine.
* **Phase 3: Bead Tracking**: Automated donut-masking (via least-squares geometry fitting) and dynamic multi-point HSV color sampling to isolate and track the bead.
* **Phase 4: Analysis Dashboard**: Interactive Matplotlib dashboard featuring Savitzky-Golay filtering, Time-Series plots, and Phase Space (velocity vs position) visualizations.

## Prerequisites & Installation

Ensure you have Python 3.8+ installed. You will need the following libraries to run the software:

    pip install opencv-python PyQt6 numpy scipy matplotlib

## How to Run
Navigate to the project folder in your terminal and run the main GUI file:

    python gui_main.py

---

## Step-by-Step User Guide

### Step 1: Load a Video
Click the **Load Video File** button and select your experiment footage (`.mp4`, `.avi`, or `.mov`). Use the playback controls (Play, Pause, Step <, Step >) or the timeline scrubber to navigate the video.

### Phase 1: Spatial Calibration
*Maps the 2D video pixels to real-world angles.*
1. Scrub to a frame where your hoop's angle markings are clearly visible.
2. Set your desired **Step Size** (e.g., 5°).
3. Click **Start Calibration**.
4. Use the pop-up **Magnifying Glass** to click precisely on the 0° mark. Continue clicking as the system automatically increments the angle.
5. Once you finish the positive angles, click **Switch to Negatives (-)** and map the remaining points.
6. Click **Finish Calibration**.
7. *(Optional)* Click **Save Profile** to save this geometry. You can use **Load Profile** in the future to skip this entire phase if the camera hasn't moved!
8. Click **Test Calibration (Hover)** and move your mouse around the hoop to verify the mathematical interpolation live on the video.

### Phase 2: Hoop Velocity (Tripwire)
*Calculates the driving angular velocity of the rotating hoop.*
1. Find a reference mark outside of the rotating hoop (assuming your camera is in the frame of the rotating hoop).
2. Click **Draw Tripwire Box**. Click and drag a bounding box on the video frame over the path where the reference mark will pass.
3. Play the video. The software uses a hysteresis loop (20% to leave, 50% to enter) to count laps. The box will turn Green when the object is **Tracked** and Red when **Waiting**.
4. Pause the video and click **Calculate Angular Velocity**. The mean angular velocity will be saved for the final analysis.

### Phase 3: Bead Tracking
*Isolates and tracks the bead's position over time.*
1. **Adjust Donut Mask**: The software automatically guesses the hoop's center and radius based on your Phase 1 calibration. Use the X/Y Shift, Radius Adj, and Thickness spinboxes to perfectly align the green mask over the bead's track.
2. **Sample Bead Color**. Using the magnifying glass, click on the bead in a few different lighting spots to sample its color. Click the button again to stop.
3. **Verify Mask**: Click **Toggle Mask Preview (B&W)**. You should see a pure black screen with the bead showing up as a solid white dot.
4. **Track**: Click **START TRACKING**. The video will auto-play, and a red reticle will lock onto the bead, recording its time and angle. You can use **Pause Tracking** at any point.

### Phase 4: Data Analysis
*Visualizes the collected kinematics data.*
1. Once you have tracked enough bead data, click **Launch Analysis Dashboard**.
2. A new window will appear featuring three interactive plots:
   * **Top Left:** Position vs. Time
   * **Bottom Left:** Angular Velocity vs. Time (with the Phase 2 Hoop Velocity overlaid as dashed green lines).
   * **Right:** Phase Space (Velocity vs Position)
3. Use the top controls to toggle the **Savitzky-Golay Filter** or adjust its Window Size and Polynomial Order in real-time.
4. Click **Export Data to CSV** to save the raw time, raw position, smoothed position, and smoothed velocity to a spreadsheet for external use.

---



## Troubleshooting
* **Video freezes while scrubbing**: This is normal for highly compressed MP4s without keyframes. The scrubber uses a 100ms throttle to prevent crashing. 
* **Bead tracking loses the bead**: Clear your color samples, toggle the B&W Mask Preview, and sample the bead again in the lighting conditions where it was lost. Adjust the Donut Mask thickness if the bead is moving out of the tracking bounds.
import sys
import cv2
import json
import numpy as np
from scipy import optimize
from scipy.interpolate import CubicSpline
import calibration 
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QSlider, 
                             QFrame, QSpinBox, QListWidget, QScrollArea, QCheckBox)

# --- MATPLOTLIB IMPORTS FOR DASHBOARD ---
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.signal import savgol_filter
import csv

class HoopTrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Variables ---
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30
        self.raw_frame = None 
        
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)
        self.scrub_timer = QTimer()
        self.scrub_timer.setSingleShot(True)
        self.scrub_timer.timeout.connect(self.update_scrub_frame)
        self.target_frame = 0

        # --- Phase 1: Calibration Variables ---
        self.calibration_mode = False
        self.step_size = 5
        self.current_angle = 0
        self.calibration_data = {} 
        self.angle_history = [] 
        self.hover_pos = None 
        self.testing_calibration = False
        self.angle_calculator = None
        self.hover_angle = None

        # --- Phase 2: Tripwire Variables ---
        self.tripwire_mode = False
        self.drawing_box = False
        self.box_start = None
        self.box_end = None
        self.tripwire_box = None 
        self.lower_color = None
        self.upper_color = None
        self.total_pixels = 0
        self.object_present = True
        self.last_lap_frame = 0
        self.lap_times = []
        self.results_omega = None

        # --- Phase 3: Bead Tracking Variables ---
        self.base_xc = 0
        self.base_yc = 0
        self.base_R = 0
        self.donut_x_off = 0
        self.donut_y_off = 0
        self.donut_r_off = 0
        self.donut_thickness = 60
        
        self.picking_color = False
        self.color_samples = []
        self.bead_lower_color = None
        self.bead_upper_color = None
        
        self.preview_mask = False
        self.bead_tracking_active = False
        self.bead_data = [] # Stores (time_s, angle_deg, cx, cy) with NaNs for lost frames
        self.current_bead_pos = None

        # --- 1. Setup the Main Window ---
        self.setWindowTitle("Rotating Hoop Kinematics Analyzer")
        self.setGeometry(100, 100, 1300, 800) 

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # ==========================================
        #             LEFT PANEL (VIDEO)
        # ==========================================
        self.left_panel = QWidget()
        self.video_layout = QVBoxLayout(self.left_panel)
        
        self.title_label = QLabel("Step 1: Load a Video")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        self.video_layout.addWidget(self.title_label)

        self.video_screen = QLabel("Please load a video.")
        self.video_screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_screen.setStyleSheet("background-color: #000000; color: #787878; font-size: 18px; border: 1px solid #444;")
        self.video_screen.setMinimumSize(640, 480)
        self.video_screen.setMouseTracking(True)
        self.video_screen.mousePressEvent = self.video_clicked 
        self.video_screen.mouseMoveEvent = self.video_mouse_move
        self.video_screen.leaveEvent = self.video_leave
        self.video_screen.mouseReleaseEvent = self.video_mouse_release
        self.video_layout.addWidget(self.video_screen, stretch=1)

        self.controls_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Step")
        self.prev_btn.clicked.connect(self.step_backward)
        self.controls_layout.addWidget(self.prev_btn)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.controls_layout.addWidget(self.play_btn)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.scrub_video)
        self.controls_layout.addWidget(self.slider)

        self.next_btn = QPushButton("Step >")
        self.next_btn.clicked.connect(self.step_forward)
        self.controls_layout.addWidget(self.next_btn)
        self.video_layout.addLayout(self.controls_layout)

        self.load_button = QPushButton("Load Video File")
        self.load_button.setStyleSheet("font-size: 16px; padding: 10px; font-weight: bold; background-color: #005A9E; color: white;")
        self.load_button.clicked.connect(self.load_video)
        self.video_layout.addWidget(self.load_button)

        self.main_layout.addWidget(self.left_panel, stretch=3) 

        # ==========================================
        #             RIGHT PANEL (SCROLLABLE)
        # ==========================================
        self.right_panel_container = QFrame()
        self.right_panel_container.setStyleSheet("QFrame { background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; }")
        self.right_layout_main = QVBoxLayout(self.right_panel_container)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        self.tools_widget = QWidget()
        self.tools_layout = QVBoxLayout(self.tools_widget)
        self.scroll_area.setWidget(self.tools_widget)
        self.right_layout_main.addWidget(self.scroll_area)
        
        self.main_layout.addWidget(self.right_panel_container, stretch=1) 

        # --- PHASE 1 UI ---
        self.tools_title = QLabel("Phase 1: Spatial Calibration")
        self.tools_title.setStyleSheet("font-size: 18px; font-weight: bold; border: none;")
        self.tools_layout.addWidget(self.tools_title)

        self.step_layout = QHBoxLayout()
        self.step_label = QLabel("Step Size (°):")
        self.step_label.setStyleSheet("border: none;")
        self.step_spinbox = QSpinBox()
        self.step_spinbox.setRange(1, 90)
        self.step_spinbox.setValue(5)
        self.step_layout.addWidget(self.step_label)
        self.step_layout.addWidget(self.step_spinbox)
        self.tools_layout.addLayout(self.step_layout)

        self.current_angle_label = QLabel("Waiting to start...")
        self.current_angle_label.setStyleSheet("font-size: 16px; color: #4DA8DA; font-weight: bold; border: none;")
        self.tools_layout.addWidget(self.current_angle_label)

        self.start_calib_btn = QPushButton("Start Calibration")
        self.start_calib_btn.clicked.connect(self.start_calibration)
        self.tools_layout.addWidget(self.start_calib_btn)

        self.switch_neg_btn = QPushButton("Switch to Negatives (-)")
        self.switch_neg_btn.clicked.connect(self.switch_to_negatives)
        self.tools_layout.addWidget(self.switch_neg_btn)

        self.finish_calib_btn = QPushButton("Finish Calibration")
        self.finish_calib_btn.clicked.connect(self.finish_calibration)
        self.tools_layout.addWidget(self.finish_calib_btn)

        self.undo_btn = QPushButton("Undo Last Click")
        self.undo_btn.clicked.connect(self.undo_calibration)
        self.tools_layout.addWidget(self.undo_btn)

        self.save_load_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Profile")
        self.save_btn.clicked.connect(self.save_calibration)
        self.save_btn.setEnabled(False) 
        self.load_profile_btn = QPushButton("Load Profile")
        self.load_profile_btn.clicked.connect(self.load_calibration)
        self.load_profile_btn.setEnabled(False) 
        self.save_load_layout.addWidget(self.save_btn)
        self.save_load_layout.addWidget(self.load_profile_btn)
        self.tools_layout.addLayout(self.save_load_layout)

        self.test_calib_btn = QPushButton("Test Calibration (Hover)")
        self.test_calib_btn.setCheckable(True)
        self.test_calib_btn.clicked.connect(self.toggle_test_calibration)
        self.test_calib_btn.setEnabled(False) 
        self.tools_layout.addWidget(self.test_calib_btn)

        checklist_lbl = QLabel("Recorded Angles:")
        checklist_lbl.setStyleSheet("border: none; margin-top: 10px;")
        self.tools_layout.addWidget(checklist_lbl)
        
        self.calib_list = QListWidget()
        self.calib_list.setStyleSheet("background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; font-size: 14px;")
        self.calib_list.setMinimumHeight(120) 
        self.tools_layout.addWidget(self.calib_list)

        # --- PHASE 2 UI ---
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        line1.setStyleSheet("background-color: #444; margin-top: 10px; margin-bottom: 10px;")
        self.tools_layout.addWidget(line1)

        self.phase2_title = QLabel("Phase 2: Hoop Velocity (Tripwire)")
        self.phase2_title.setStyleSheet("font-size: 18px; font-weight: bold; border: none;")
        self.tools_layout.addWidget(self.phase2_title)

        self.draw_box_btn = QPushButton("Draw Tripwire Box")
        self.draw_box_btn.clicked.connect(self.toggle_draw_box)
        self.tools_layout.addWidget(self.draw_box_btn)

        self.tripwire_status_lbl = QLabel("Tripwire: Not Set")
        self.tripwire_status_lbl.setStyleSheet("color: #787878; font-weight: bold; border: none;")
        self.tools_layout.addWidget(self.tripwire_status_lbl)

        self.lap_count_lbl = QLabel("Laps: 0")
        self.lap_count_lbl.setStyleSheet("font-size: 16px; border: none;")
        self.tools_layout.addWidget(self.lap_count_lbl)

        self.calc_omega_btn = QPushButton("Calculate Angular Velocity")
        self.calc_omega_btn.clicked.connect(self.calculate_omega)
        self.calc_omega_btn.setEnabled(False)
        self.tools_layout.addWidget(self.calc_omega_btn)

        # --- PHASE 3 UI ---
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet("background-color: #444; margin-top: 10px; margin-bottom: 10px;")
        self.tools_layout.addWidget(line2)

        self.phase3_title = QLabel("Phase 3: Bead Tracking")
        self.phase3_title.setStyleSheet("font-size: 18px; font-weight: bold; border: none;")
        self.tools_layout.addWidget(self.phase3_title)

        self.tools_layout.addWidget(QLabel("1. Adjust Donut Mask:"))
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("X Shift:"))
        self.x_off_spin = QSpinBox()
        self.x_off_spin.setRange(-500, 500)
        self.x_off_spin.valueChanged.connect(self.update_donut_radii)
        row1.addWidget(self.x_off_spin)
        
        row1.addWidget(QLabel("Y Shift:"))
        self.y_off_spin = QSpinBox()
        self.y_off_spin.setRange(-500, 500)
        self.y_off_spin.valueChanged.connect(self.update_donut_radii)
        row1.addWidget(self.y_off_spin)
        self.tools_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Radius Adj:"))
        self.r_off_spin = QSpinBox()
        self.r_off_spin.setRange(-500, 500)
        self.r_off_spin.valueChanged.connect(self.update_donut_radii)
        row2.addWidget(self.r_off_spin)
        
        row2.addWidget(QLabel("Thickness:"))
        self.thick_spin = QSpinBox()
        self.thick_spin.setRange(1, 500)
        self.thick_spin.setValue(self.donut_thickness)
        self.thick_spin.valueChanged.connect(self.update_donut_radii)
        row2.addWidget(self.thick_spin)
        self.tools_layout.addLayout(row2)

        self.sample_color_btn = QPushButton("2. Sample Bead Color")
        self.sample_color_btn.setCheckable(True)
        self.sample_color_btn.clicked.connect(self.toggle_pick_color)
        self.tools_layout.addWidget(self.sample_color_btn)

        self.clear_color_btn = QPushButton("Clear Color Samples")
        self.clear_color_btn.clicked.connect(self.clear_color_samples)
        self.tools_layout.addWidget(self.clear_color_btn)

        self.preview_mask_btn = QPushButton("Toggle Mask Preview (B&W)")
        self.preview_mask_btn.setCheckable(True)
        self.preview_mask_btn.clicked.connect(self.toggle_mask_preview)
        self.tools_layout.addWidget(self.preview_mask_btn)

        # ADDED: Max Speed Gating UI
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Max Speed (°/s):"))
        self.max_speed_spin = QSpinBox()
        self.max_speed_spin.setRange(10, 10000)
        self.max_speed_spin.setValue(1000)
        row3.addWidget(self.max_speed_spin)
        self.tools_layout.addLayout(row3)

        track_btn_layout = QHBoxLayout()
        self.start_tracking_btn = QPushButton("START TRACKING")
        self.start_tracking_btn.setStyleSheet("background-color: #005A9E; color: white; font-weight: bold;")
        self.start_tracking_btn.setCheckable(True)
        self.start_tracking_btn.clicked.connect(self.toggle_bead_tracking)
        
        self.pause_tracking_btn = QPushButton("Pause Tracking")
        self.pause_tracking_btn.setCheckable(True)
        self.pause_tracking_btn.setEnabled(False)
        self.pause_tracking_btn.clicked.connect(self.pause_bead_tracking)

        self.fast_track_btn = QPushButton("Fast Track (No Video)")
        self.fast_track_btn.setStyleSheet("background-color: #8E44AD; color: white; font-weight: bold;")
        self.fast_track_btn.clicked.connect(self.fast_track_bead)
        
        track_btn_layout.addWidget(self.start_tracking_btn)
        track_btn_layout.addWidget(self.pause_tracking_btn)
        track_btn_layout.addWidget(self.fast_track_btn) # Added to layout
        self.tools_layout.addLayout(track_btn_layout)
        
        track_btn_layout.addWidget(self.start_tracking_btn)
        track_btn_layout.addWidget(self.pause_tracking_btn)
        self.tools_layout.addLayout(track_btn_layout)

        self.bead_status_lbl = QLabel("Frames processed: 0 | Valid points: 0")
        self.bead_status_lbl.setStyleSheet("color: #787878; font-style: italic; border: none;")
        self.tools_layout.addWidget(self.bead_status_lbl)

        # --- PHASE 4 UI ---
        line4 = QFrame()
        line4.setFrameShape(QFrame.Shape.HLine)
        line4.setStyleSheet("background-color: #444; margin-top: 10px; margin-bottom: 10px;")
        self.tools_layout.addWidget(line4)

        self.phase4_title = QLabel("Phase 4: Data Analysis")
        self.phase4_title.setStyleSheet("font-size: 18px; font-weight: bold; border: none;")
        self.tools_layout.addWidget(self.phase4_title)

        self.show_results_btn = QPushButton("Launch Analysis Dashboard")
        self.show_results_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.show_results_btn.clicked.connect(self.open_results_dashboard)
        self.tools_layout.addWidget(self.show_results_btn)

        self.tools_layout.addStretch() 
        self.toggle_buttons(False)

    # ==========================================
    #             VIDEO LOGIC
    # ==========================================
    def toggle_buttons(self, state):
        self.prev_btn.setEnabled(state)
        self.play_btn.setEnabled(state)
        self.next_btn.setEnabled(state)
        self.slider.setEnabled(state)
        
        self.start_calib_btn.setEnabled(state)
        self.switch_neg_btn.setEnabled(False)
        self.finish_calib_btn.setEnabled(False)
        self.undo_btn.setEnabled(False)

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_path = file_name
            self.title_label.setText(f"Loaded: {file_name.split('/')[-1]}")
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30 
            
            self.slider.setMaximum(self.total_frames - 1)
            self.toggle_buttons(True)
            self.jump_to_frame(0)

    def jump_to_frame(self, frame_idx):
        if self.cap is None: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = frame_idx
            self.raw_frame = frame.copy()
            self.paint_frame(self.raw_frame)

    def play_next_frame(self):
        if self.cap is None: return
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx += 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame_idx)
            self.slider.blockSignals(False)
            self.raw_frame = frame.copy()

            self.check_tripwire(self.raw_frame)
            
            if self.bead_tracking_active and not self.pause_tracking_btn.isChecked():
                self.process_bead_frame(self.raw_frame)

            self.paint_frame(self.raw_frame)
        else:
            self.toggle_play() 

    def paint_frame(self, frame):
        display_frame = frame.copy()

        final_xc = self.base_xc + self.donut_x_off
        final_yc = self.base_yc + self.donut_y_off
        final_R = max(1, self.base_R + self.donut_r_off)
        final_thickness = max(1, self.donut_thickness)

        if self.preview_mask and self.base_R > 0 and self.bead_lower_color is not None:
            h, w = display_frame.shape[:2]
            donut_mask_img = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(donut_mask_img, (final_xc, final_yc), final_R, 255, thickness=final_thickness)
            
            masked_frame = cv2.bitwise_and(display_frame, display_frame, mask=donut_mask_img)
            hsv_current = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv_current, self.bead_lower_color, self.bead_upper_color)
            
            kernel = np.ones((3,3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            
            display_frame = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)

        if not self.preview_mask:
            if self.base_R > 0 and not self.bead_tracking_active:
                overlay = np.zeros_like(display_frame)
                cv2.circle(overlay, (final_xc, final_yc), final_R, (0, 255, 0), thickness=final_thickness)
                cv2.addWeighted(overlay, 0.4, display_frame, 1.0, 0, display_frame)

            for (sx, sy) in self.color_samples:
                cv2.circle(display_frame, (sx, sy), 3, (0, 255, 255), -1)

            if self.current_bead_pos:
                bx, by = self.current_bead_pos
                cv2.circle(display_frame, (bx, by), 6, (0, 0, 255), -1)
                
                # Retrieve the last valid angle to display
                last_theta = None
                for d in reversed(self.bead_data):
                    if not np.isnan(d[1]):
                        last_theta = d[1]
                        break
                if last_theta is not None:
                    cv2.putText(display_frame, f"{last_theta:.1f} deg", (bx + 10, by - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for angle, (cx, cy) in self.calibration_data.items():
                cv2.circle(display_frame, (cx, cy), 4, (0, 255, 0), -1)

            if (self.calibration_mode or self.picking_color) and self.hover_pos is not None:
                hx, hy = self.hover_pos
                patch_r, zoom = 25, 3     
                fh, fw = display_frame.shape[:2]
                y1, y2 = max(0, hy - patch_r), min(fh, hy + patch_r)
                x1, x2 = max(0, hx - patch_r), min(fw, hx + patch_r)
                patch = display_frame[y1:y2, x1:x2].copy()
                
                if patch.size > 0:
                    mag_patch = cv2.resize(patch, (0, 0), fx=zoom, fy=zoom)
                    mh, mw = mag_patch.shape[:2]
                    cv2.line(mag_patch, (mw//2, 0), (mw//2, mh), (0, 0, 255), 1)
                    cv2.line(mag_patch, (0, mh//2), (mw, mh//2), (0, 0, 255), 1)
                    cv2.rectangle(mag_patch, (0,0), (mw-1, mh-1), (255,255,255), 2)
                    
                    draw_x, draw_y = hx + 20, hy - mh - 20
                    if draw_x + mw > fw: draw_x = hx - mw - 20
                    if draw_y < 0: draw_y = hy + 20
                    
                    dy1, dy2 = max(0, draw_y), min(fh, draw_y + mh)
                    dx1, dx2 = max(0, draw_x), min(fw, draw_x + mw)
                    py1 = 0 if draw_y >= 0 else -draw_y
                    py2 = mh if (draw_y + mh) <= fh else fh - draw_y
                    px1 = 0 if draw_x >= 0 else -draw_x
                    px2 = mw if (draw_x + mw) <= fw else fw - draw_x
                    display_frame[dy1:dy2, dx1:dx2] = mag_patch[py1:py2, px1:px2]

            if self.testing_calibration and self.hover_pos is not None and self.hover_angle is not None:
                hx, hy = self.hover_pos
                text = f"{self.hover_angle:.1f} deg"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_frame, (hx + 10, hy - 35), (hx + 15 + tw, hy - 35 + th + 10), (0, 0, 0), -1)
                cv2.putText(display_frame, text, (hx + 15, hy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if self.drawing_box and self.box_start and self.box_end:
                cv2.rectangle(display_frame, self.box_start, self.box_end, (255, 165, 0), 2)
            elif self.tripwire_box:
                x, y, w, h = self.tripwire_box
                color = (0, 255, 0) if self.object_present else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)

        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        self.current_pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_screen.width(), self.video_screen.height(), 
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.video_screen.setPixmap(self.current_pixmap)

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_btn.setText("Play")
        else:
            self.play_timer.start(int(1000 / self.fps))
            self.play_btn.setText("Pause")

    def scrub_video(self, value):
        if value != self.current_frame_idx:
            self.target_frame = value
            if self.play_timer.isActive(): self.toggle_play()
            if not self.scrub_timer.isActive(): self.scrub_timer.start(100) 

    def update_scrub_frame(self):
        self.jump_to_frame(self.target_frame)

    def step_forward(self):
        if self.play_timer.isActive(): self.toggle_play() 
        if self.current_frame_idx < self.total_frames - 1:
            self.slider.setValue(self.current_frame_idx + 1)

    def step_backward(self):
        if self.play_timer.isActive(): self.toggle_play()
        if self.current_frame_idx > 0:
            self.slider.setValue(self.current_frame_idx - 1)

    def get_true_coords(self, event):
        label_w, label_h = self.video_screen.width(), self.video_screen.height()
        pix_w, pix_h = self.current_pixmap.width(), self.current_pixmap.height()
        offset_x = (label_w - pix_w) // 2
        offset_y = (label_h - pix_h) // 2
        
        click_x = event.position().x() - offset_x
        click_y = event.position().y() - offset_y
        
        if click_x < 0 or click_y < 0 or click_x > pix_w or click_y > pix_h: 
            return None
        
        frame_h, frame_w = self.raw_frame.shape[:2]
        true_x = int(click_x * (frame_w / pix_w))
        true_y = int(click_y * (frame_h / pix_h))
        return (true_x, true_y)

    # ==========================================
    #             MOUSE EVENTS
    # ==========================================
    def video_clicked(self, event):
        if self.raw_frame is None: return
        coords = self.get_true_coords(event)
        if not coords: return
        true_x, true_y = coords

        if self.picking_color:
            self.color_samples.append((true_x, true_y))
            pristine_frame = self.raw_frame.copy()
            hsv_frame = cv2.cvtColor(pristine_frame, cv2.COLOR_BGR2HSV)
            hsv_samples = [hsv_frame[y, x] for x, y in self.color_samples]
            
            h_vals = [int(s[0]) for s in hsv_samples]
            s_vals = [int(s[1]) for s in hsv_samples]
            v_vals = [int(s[2]) for s in hsv_samples]
            
            self.bead_lower_color = np.array([max(0, min(h_vals)-15), max(30, min(s_vals)-50), max(30, min(v_vals)-50)], dtype=np.uint8)
            self.bead_upper_color = np.array([min(179, max(h_vals)+15), 255, 255], dtype=np.uint8)

            self.sample_color_btn.setText(f"Sampling... ({len(self.color_samples)} points). Click to stop.")
            self.paint_frame(self.raw_frame)
            return

        if self.tripwire_mode:
            self.drawing_box = True
            self.box_start = coords
            self.box_end = coords
            return 

        if self.calibration_mode:
            self.calibration_data[self.current_angle] = (true_x, true_y)
            self.angle_history.append(self.current_angle)
            self.calib_list.addItem(f"{self.current_angle}°  ✓")
            self.calib_list.scrollToBottom()
            
            if self.current_angle >= 0: self.current_angle += self.step_size
            else: self.current_angle -= self.step_size
                
            self.update_calibration_ui()
            self.paint_frame(self.raw_frame)

    def video_mouse_move(self, event):
        if self.raw_frame is None: return
        self.hover_pos = self.get_true_coords(event)
        
        if self.drawing_box and self.hover_pos:
            self.box_end = self.hover_pos
            self.paint_frame(self.raw_frame)
            return

        if self.testing_calibration and self.hover_pos and self.angle_calculator:
            hx, hy = self.hover_pos
            try: self.hover_angle = self.angle_calculator(hx, hy)
            except ValueError: self.hover_angle = None 

        if self.calibration_mode or self.testing_calibration or self.picking_color:
            self.paint_frame(self.raw_frame)

    def video_mouse_release(self, event):
        if self.drawing_box:
            self.drawing_box = False
            coords = self.get_true_coords(event)
            if coords: self.box_end = coords
            
            x1, y1 = self.box_start
            x2, y2 = self.box_end
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            
            if w > 10 and h > 10: 
                self.tripwire_box = (x, y, w, h)
                self.total_pixels = w * h
                
                roi_patch = self.raw_frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2HSV)
                median_hsv = np.median(hsv_roi, axis=(0, 1))
                h_val = int(median_hsv[0])
                
                self.lower_color = np.array([max(0, h_val - 15), 50, 50], dtype=np.uint8)
                self.upper_color = np.array([min(179, h_val + 15), 255, 255], dtype=np.uint8)
                
                self.object_present = True
                self.last_lap_frame = self.current_frame_idx
                self.lap_times = []
                
                self.tripwire_status_lbl.setText("Tripwire: Ready (Tracking...)")
                self.tripwire_status_lbl.setStyleSheet("color: #4CAF50; font-weight: bold; border: none;")
                self.lap_count_lbl.setText("Laps: 0")
                self.calc_omega_btn.setEnabled(True)
            
            self.toggle_draw_box() 
            self.paint_frame(self.raw_frame)

    def video_leave(self, event):
        self.hover_pos = None
        if self.raw_frame is not None:
            self.paint_frame(self.raw_frame)

    # ==========================================
    #             PHASE 1 METHODS
    # ==========================================
    def start_calibration(self):
        self.calibration_mode = True
        self.calibration_data = {}
        self.angle_history = []
        self.calib_list.clear()
        self.step_size = self.step_spinbox.value()
        self.current_angle = 0
        self.step_spinbox.setEnabled(False)
        self.start_calib_btn.setEnabled(False)
        self.switch_neg_btn.setEnabled(True)
        self.finish_calib_btn.setEnabled(True)
        self.undo_btn.setEnabled(False)

        self.save_btn.setEnabled(True)
        self.load_profile_btn.setEnabled(True)

        self.update_calibration_ui()
        if self.raw_frame is not None: self.paint_frame(self.raw_frame)

    def switch_to_negatives(self):
        self.current_angle = -self.step_size
        self.switch_neg_btn.setEnabled(False)
        self.update_calibration_ui()

    def finish_calibration(self):
        self.calibration_mode = False
        self.hover_pos = None 
        self.current_angle_label.setText("Calibration Complete!")
        self.current_angle_label.setStyleSheet("font-size: 18px; color: #4CAF50; font-weight: bold; margin-top: 10px; border: none;")
        self.step_spinbox.setEnabled(True)
        self.start_calib_btn.setEnabled(True)
        self.start_calib_btn.setText("Restart Calibration")
        self.switch_neg_btn.setEnabled(False)
        self.finish_calib_btn.setEnabled(False)
        self.undo_btn.setEnabled(False)

        self.save_btn.setEnabled(False)
        self.load_profile_btn.setEnabled(False)
        
        self.test_calib_btn.setEnabled(True)
        self.angle_calculator = calibration.create_interpolation_map(self.calibration_data)
        self.calculate_base_geometry() 
        
        if self.raw_frame is not None: self.paint_frame(self.raw_frame)

    def update_calibration_ui(self):
        self.current_angle_label.setText(f"Click Angle: {self.current_angle}°")
        self.undo_btn.setEnabled(len(self.angle_history) > 0)

    def undo_calibration(self):
        if not self.angle_history: return
        last_angle = self.angle_history.pop()
        if last_angle in self.calibration_data: del self.calibration_data[last_angle]
        row = self.calib_list.count() - 1
        self.calib_list.takeItem(row)
        self.current_angle = last_angle
        if self.current_angle >= 0: self.switch_neg_btn.setEnabled(True)
        self.update_calibration_ui()
        self.paint_frame(self.raw_frame)

    def save_calibration(self):
        if not self.calibration_data: return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Calibration Profile", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'w') as f: json.dump(self.calibration_data, f, indent=4)
            self.title_label.setText("Calibration Saved Successfully!")

    def load_calibration(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Calibration Profile", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    data = json.load(f)
                    self.calibration_data = {int(k): tuple(v) for k, v in data.items()}
                self.calib_list.clear()
                for angle in sorted(self.calibration_data.keys()): self.calib_list.addItem(f"{angle}°  ✓")
                self.title_label.setText(f"Loaded Profile: {file_name.split('/')[-1]}")
                
                self.test_calib_btn.setEnabled(True)
                self.angle_calculator = calibration.create_interpolation_map(self.calibration_data)
                self.calculate_base_geometry() 
                
                if self.raw_frame is not None: self.paint_frame(self.raw_frame)
            except Exception as e: print(f"Failed to load: {e}")

    def toggle_test_calibration(self):
        self.testing_calibration = self.test_calib_btn.isChecked()
        if self.testing_calibration: self.test_calib_btn.setText("Stop Testing")
        else: 
            self.test_calib_btn.setText("Test Calibration (Hover)")
            self.hover_angle = None
        if self.raw_frame is not None: self.paint_frame(self.raw_frame)

    def calculate_base_geometry(self):
        if len(self.calibration_data) < 3: return
        
        points = np.array(list(self.calibration_data.values()))
        px, py = points[:, 0], points[:, 1]
        
        def calc_R(xc, yc): return np.sqrt((px - xc)**2 + (py - yc)**2)
        def f_2(c): Ri = calc_R(*c); return Ri - Ri.mean()
        
        center_estimate = (np.mean(px), np.mean(py))
        center_fit, _ = optimize.leastsq(f_2, center_estimate)
        
        self.base_xc, self.base_yc = int(center_fit[0]), int(center_fit[1])
        self.base_R = int(calc_R(self.base_xc, self.base_yc).mean())
        self.r_off_spin.setValue(-20)

    # ==========================================
    #             PHASE 2 METHODS
    # ==========================================
    def toggle_draw_box(self):
        self.tripwire_mode = not self.tripwire_mode
        if self.tripwire_mode:
            self.draw_box_btn.setText("Cancel Drawing")
            self.draw_box_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
            self.tripwire_box = None 
            self.tripwire_status_lbl.setText("Tripwire: Click and drag on video...")
            self.tripwire_status_lbl.setStyleSheet("color: #FFC107; font-weight: bold; border: none;")
        else:
            self.draw_box_btn.setText("Draw Tripwire Box")
            self.draw_box_btn.setStyleSheet("")
            self.drawing_box = False

    def check_tripwire(self, frame):
        if not self.tripwire_box or self.lower_color is None: return
        x, y, w, h = self.tripwire_box
        current_roi = frame[y:y+h, x:x+w]
        hsv_current = cv2.cvtColor(current_roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_current, self.lower_color, self.upper_color)
        matching_pixels = cv2.countNonZero(mask)
        fill_percentage = matching_pixels / self.total_pixels if self.total_pixels > 0 else 0
        
        if self.object_present:
            if fill_percentage < 0.20:
                self.object_present = False
                self.tripwire_status_lbl.setText("Tripwire: Waiting") 
                self.tripwire_status_lbl.setStyleSheet("color: #FFC107; font-weight: bold; border: none;")
        else:
            if fill_percentage >= 0.50:
                self.object_present = True
                self.tripwire_status_lbl.setText("Tripwire: Tracked") 
                self.tripwire_status_lbl.setStyleSheet("color: #4CAF50; font-weight: bold; border: none;")
                lap_frames = self.current_frame_idx - self.last_lap_frame
                lap_time = lap_frames / self.fps
                if lap_time > 0: self.lap_times.append(lap_time)
                self.lap_count_lbl.setText(f"Laps: {len(self.lap_times)}")
                self.last_lap_frame = self.current_frame_idx

    def calculate_omega(self):
        if not self.lap_times:
            self.lap_count_lbl.setText("Laps: 0 (No complete laps)")
            return
        lap_times_arr = np.array(self.lap_times)
        omegas_arr = (2 * np.pi) / lap_times_arr
        mean_omega = np.mean(omegas_arr)
        std_omega = np.std(omegas_arr)
        self.results_omega = {"mean_omega": mean_omega, "error_omega": std_omega}
        self.lap_count_lbl.setText(f"Laps: {len(self.lap_times)} | Omega: {mean_omega:.3f} ± {std_omega:.3f} rad/s")

    # ==========================================
    #             PHASE 3 METHODS
    # ==========================================
    def update_donut_radii(self):
        self.donut_x_off = self.x_off_spin.value()
        self.donut_y_off = self.y_off_spin.value()
        self.donut_r_off = self.r_off_spin.value()
        self.donut_thickness = self.thick_spin.value()
        if self.raw_frame is not None: self.paint_frame(self.raw_frame)

    def toggle_pick_color(self):
        self.picking_color = self.sample_color_btn.isChecked()
        if self.picking_color:
            self.sample_color_btn.setStyleSheet("background-color: #FFC107; color: black; font-weight: bold;")
            self.sample_color_btn.setText("Click the bead on video...")
        else:
            self.sample_color_btn.setStyleSheet("")
            self.sample_color_btn.setText("2. Sample Bead Color")
        if self.raw_frame is not None: self.paint_frame(self.raw_frame)

    def clear_color_samples(self):
        self.color_samples = []
        self.bead_lower_color = None
        self.bead_upper_color = None
        if self.picking_color:
            self.sample_color_btn.setText("Click the bead on video...")
        if self.raw_frame is not None: self.paint_frame(self.raw_frame)

    def toggle_mask_preview(self):
        self.preview_mask = self.preview_mask_btn.isChecked()
        if self.preview_mask:
            self.preview_mask_btn.setText("Turn Off Mask Preview")
            self.preview_mask_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        else:
            self.preview_mask_btn.setText("Toggle Mask Preview (B&W)")
            self.preview_mask_btn.setStyleSheet("")
        if self.raw_frame is not None: self.paint_frame(self.raw_frame)

    def toggle_bead_tracking(self):
        if not self.angle_calculator or self.bead_lower_color is None:
            self.start_tracking_btn.setChecked(False)
            self.bead_status_lbl.setText("Error: Calibrate Phase 1 & sample color first!")
            self.bead_status_lbl.setStyleSheet("color: #f44336; font-weight: bold; border: none;")
            return

        self.bead_tracking_active = self.start_tracking_btn.isChecked()
        if self.bead_tracking_active:
            self.start_tracking_btn.setText("STOP TRACKING")
            self.start_tracking_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
            
            self.bead_data = [] 
            self.pause_tracking_btn.setEnabled(True)
            self.pause_tracking_btn.setChecked(False)
            self.pause_tracking_btn.setText("Pause Tracking")
            self.bead_status_lbl.setText("Tracking... Playing video to collect data.")
            if not self.play_timer.isActive(): self.toggle_play() 
        else:
            self.start_tracking_btn.setText("START TRACKING")
            self.start_tracking_btn.setStyleSheet("background-color: #005A9E; color: white; font-weight: bold;")
            self.pause_tracking_btn.setEnabled(False)
            valid_pts = sum(1 for d in self.bead_data if not np.isnan(d[1]))
            self.bead_status_lbl.setText(f"Finished. Frames: {len(self.bead_data)} | Valid points: {valid_pts}")
            if self.play_timer.isActive(): self.toggle_play() 

    def pause_bead_tracking(self):
        if self.pause_tracking_btn.isChecked():
            self.bead_tracking_active = False 
            self.pause_tracking_btn.setText("Resume Tracking")
            if self.play_timer.isActive(): self.toggle_play() 
        else:
            self.bead_tracking_active = True 
            self.pause_tracking_btn.setText("Pause Tracking")
            if not self.play_timer.isActive(): self.toggle_play() 

    def fast_track_bead(self):
        # 1. Safety Checks
        if not self.angle_calculator or self.bead_lower_color is None:
            self.bead_status_lbl.setText("Error: Calibrate Phase 1 & sample color first!")
            self.bead_status_lbl.setStyleSheet("color: #f44336; font-weight: bold; border: none;")
            return

        # 2. Stop playback if it's currently running
        if self.play_timer.isActive():
            self.toggle_play()

        # 3. Disable UI elements so the user doesn't click around during processing
        self.toggle_buttons(False)
        self.fast_track_btn.setEnabled(False)
        self.start_tracking_btn.setEnabled(False)
        self.bead_data = [] # Clear old data
        
        # 4. Jump back to the beginning of the video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0

        self.bead_status_lbl.setText("Fast tracking... Please wait.")
        self.bead_status_lbl.setStyleSheet("color: #FFC107; font-weight: bold; border: none;")
        QApplication.processEvents() # Force the UI to update the text

        # 5. The high-speed processing loop
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break # End of video
            
            self.current_frame_idx += 1
            self.process_bead_frame(frame)

            # Update the UI slightly every 100 frames to show progress
            if self.current_frame_idx % 100 == 0:
                self.slider.blockSignals(True)
                self.slider.setValue(self.current_frame_idx)
                self.slider.blockSignals(False)
                
                valid_count = sum(1 for d in self.bead_data if not np.isnan(d[1]))
                self.bead_status_lbl.setText(f"Fast Tracking: {self.current_frame_idx}/{self.total_frames} frames | Valid: {valid_count}")
                QApplication.processEvents() # Keep GUI responsive

        # 6. Clean up and restore UI
        valid_count = sum(1 for d in self.bead_data if not np.isnan(d[1]))
        self.bead_status_lbl.setText(f"Fast Track Complete! Frames: {len(self.bead_data)} | Valid: {valid_count}")
        self.bead_status_lbl.setStyleSheet("color: #4CAF50; font-weight: bold; border: none;")
        
        self.toggle_buttons(True)
        self.fast_track_btn.setEnabled(True)
        self.start_tracking_btn.setEnabled(True)
        
        # Jump back to the start so the user sees the first frame again
        self.jump_to_frame(0)

    def process_bead_frame(self, frame):
        if self.base_R == 0 or self.bead_lower_color is None: return
        
        time_s = self.current_frame_idx / self.fps
        found_valid_bead = False

        final_xc = self.base_xc + self.donut_x_off
        final_yc = self.base_yc + self.donut_y_off
        final_R = max(1, self.base_R + self.donut_r_off)
        final_thickness = max(1, self.donut_thickness)

        h, w = frame.shape[:2]
        donut_mask_img = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(donut_mask_img, (final_xc, final_yc), final_R, 255, thickness=final_thickness)

        masked_frame = cv2.bitwise_and(frame, frame, mask=donut_mask_img)
        hsv_current = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_current, self.bead_lower_color, self.bead_upper_color)
        
        kernel = np.ones((3,3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 5:
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    try:
                        theta = self.angle_calculator(cx, cy)
                        
                        # --- KINEMATIC GATING ---
                        if len(self.bead_data) == 0:
                            self.bead_data.append((time_s, theta, cx, cy))
                            self.current_bead_pos = (cx, cy)
                            found_valid_bead = True
                        else:
                            last_valid_time = None
                            last_valid_theta = None
                            for i in range(len(self.bead_data)-1, -1, -1):
                                if not np.isnan(self.bead_data[i][1]):
                                    last_valid_time = self.bead_data[i][0]
                                    last_valid_theta = self.bead_data[i][1]
                                    break
                            
                            if last_valid_theta is not None:
                                diff = (theta - last_valid_theta + 180) % 360 - 180
                                dt = time_s - last_valid_time
                                if dt > 0:
                                    speed = abs(diff) / dt
                                    max_speed = self.max_speed_spin.value()
                                    if speed <= max_speed:
                                        self.bead_data.append((time_s, theta, cx, cy))
                                        self.current_bead_pos = (cx, cy)
                                        found_valid_bead = True
                            else:
                                self.bead_data.append((time_s, theta, cx, cy))
                                self.current_bead_pos = (cx, cy)
                                found_valid_bead = True

                    except ValueError:
                        pass 

        if not found_valid_bead:
            self.bead_data.append((time_s, np.nan, np.nan, np.nan))
            self.current_bead_pos = None

        valid_count = sum(1 for d in self.bead_data if not np.isnan(d[1]))
        self.bead_status_lbl.setText(f"Frames processed: {len(self.bead_data)} | Valid points: {valid_count}")

    def open_results_dashboard(self):
        valid_count = sum(1 for d in self.bead_data if not np.isnan(d[1]))
        if valid_count < 10:
            self.bead_status_lbl.setText("Error: Need more valid bead data to analyze!")
            self.bead_status_lbl.setStyleSheet("color: #f44336; font-weight: bold; border: none;")
            return
            
        self.dashboard = ResultsDashboard(self.bead_data, self.results_omega)
        self.dashboard.show()


# ==========================================
#             RESULTS DASHBOARD (Phase 4)
# ==========================================
class ResultsDashboard(QWidget):
    def __init__(self, bead_data, hoop_results):
        super().__init__()
        self.setWindowTitle("Kinematics Analysis Dashboard")
        self.setGeometry(150, 150, 1200, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: #E0E0E0;")

        # Prepare continuous time array and extract raw data
        self.raw_data = np.array(bead_data, dtype=float)
        self.t_raw = self.raw_data[:, 0]
        self.theta_raw = self.raw_data[:, 1]
        self.hoop_results = hoop_results

        # Isolate valid data points for interpolation
        valid_mask = ~np.isnan(self.theta_raw)
        self.t_valid = self.t_raw[valid_mask]
        self.theta_valid = self.theta_raw[valid_mask]

        # Unwrap the strictly valid data
        self.theta_valid_unwrapped = np.unwrap(self.theta_valid * np.pi / 180) * 180 / np.pi

        # --- CUBIC SPLINE INTERPOLATION ---
        # We build the spline on valid points, and evaluate it over the entire time array
        cs = CubicSpline(self.t_valid, self.theta_valid_unwrapped)
        self.theta_unwrapped_continuous = cs(self.t_raw) 

        self.layout = QVBoxLayout(self)

        # --- Top Controls ---
        self.controls_layout = QHBoxLayout()
        
        self.fit_checkbox = QCheckBox("Apply Savitzky-Golay Filter")
        self.fit_checkbox.setChecked(True)
        self.fit_checkbox.stateChanged.connect(self.update_plots)
        self.controls_layout.addWidget(self.fit_checkbox)

        self.controls_layout.addWidget(QLabel("  |  Window:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(5, 501)
        self.window_spin.setSingleStep(2) 
        self.window_spin.setValue(min(31, len(self.t_raw) - 1 if len(self.t_raw) % 2 == 0 else len(self.t_raw)))
        self.window_spin.valueChanged.connect(self.update_plots)
        self.controls_layout.addWidget(self.window_spin)

        self.controls_layout.addWidget(QLabel("Poly Order:"))
        self.poly_spin = QSpinBox()
        self.poly_spin.setRange(1, 5)
        self.poly_spin.setValue(3)
        self.poly_spin.valueChanged.connect(self.update_plots)
        self.controls_layout.addWidget(self.poly_spin)
        
        self.controls_layout.addStretch()

        self.export_btn = QPushButton("Export Data to CSV")
        self.export_btn.setStyleSheet("background-color: #005A9E; color: white; font-weight: bold; padding: 5px 15px;")
        self.export_btn.clicked.connect(self.export_csv)
        self.controls_layout.addWidget(self.export_btn)

        self.layout.addLayout(self.controls_layout)

        # --- Matplotlib Canvas ---
        self.figure = Figure(facecolor='#121212')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self) 
        
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        self.ax1 = self.figure.add_subplot(2, 2, 1) 
        self.ax2 = self.figure.add_subplot(2, 2, 3, sharex=self.ax1) 
        self.ax3 = self.figure.add_subplot(1, 2, 2) 

        self.update_plots()

    def update_plots(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Filtering Logic applied to the fully continuous interpolated array
        if self.fit_checkbox.isChecked():
            win_len = self.window_spin.value()
            if win_len % 2 == 0: win_len += 1
            if win_len > len(self.t_raw): win_len = len(self.t_raw) if len(self.t_raw) % 2 != 0 else len(self.t_raw) - 1
            poly = min(self.poly_spin.value(), win_len - 1)
            
            theta_smooth = savgol_filter(self.theta_unwrapped_continuous, window_length=win_len, polyorder=poly)
            
            self.window_spin.setEnabled(True)
            self.poly_spin.setEnabled(True)
        else:
            theta_smooth = self.theta_unwrapped_continuous
            self.window_spin.setEnabled(False)
            self.poly_spin.setEnabled(False)
        
        # Calculate Derivatives using continuous arrays
        omega_smooth = np.gradient(theta_smooth * np.pi / 180, self.t_raw) 

        # --- Plot 1: Position ---
        # Plot only the valid raw points so the user can see where data was successfully tracked
        self.ax1.plot(self.t_valid, self.theta_valid_unwrapped, '.', color='#555555', alpha=0.5, label="Raw Valid Data")
        
        if self.fit_checkbox.isChecked():
            self.ax1.plot(self.t_raw, theta_smooth, '-', color='#4DA8DA', linewidth=2, label="Interpolated & Smoothed")
        else:
            self.ax1.plot(self.t_raw, self.theta_unwrapped_continuous, '-', color='#4DA8DA', linewidth=2, label="Interpolated (No Filter)")
            
        self.ax1.set_ylabel("Position (Degrees)", color='white')
        self.ax1.set_title("Bead Position vs Time", color='white')
        self.ax1.legend(facecolor='#2b2b2b', edgecolor='none', labelcolor='white')
        self.ax1.grid(True, color='#333333', linestyle='--')

        # --- Plot 2: Velocity ---
        self.ax2.plot(self.t_raw, omega_smooth, '-', color='#FF5722', linewidth=2, label="Bead Vel (ω)")
        
        if self.hoop_results:
            hoop_w = self.hoop_results['mean_omega']
            self.ax2.axhline(hoop_w, color='#4CAF50', linestyle='--', linewidth=2, label=f"Hoop Vel (+{hoop_w:.2f})")
            self.ax2.axhline(-hoop_w, color='#4CAF50', linestyle='--', linewidth=2, label=f"Hoop Vel (-{hoop_w:.2f})")

        self.ax2.set_xlabel("Time (Seconds)", color='white')
        self.ax2.set_ylabel("Angular Velocity (rad/s)", color='white')
        self.ax2.set_title("Bead Velocity vs Time", color='white')
        self.ax2.legend(facecolor='#2b2b2b', edgecolor='none', labelcolor='white')
        self.ax2.grid(True, color='#333333', linestyle='--')

        # --- Plot 3: Phase Space (Omega vs Theta) ---
        self.ax3.plot(theta_smooth, omega_smooth, '-', color='#E040FB', linewidth=1.5)
        self.ax3.set_xlabel("Position (Degrees)", color='white')
        self.ax3.set_ylabel("Angular Velocity (rad/s)", color='white')
        self.ax3.set_title("Phase Space (\u03B8 vs \u03C9)", color='white')
        self.ax3.grid(True, color='#333333', linestyle='--')

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#555555')

        self.figure.tight_layout()
        self.canvas.draw()
        
        self.export_data = np.column_stack((self.t_raw, self.theta_raw, self.theta_unwrapped_continuous, theta_smooth, omega_smooth))

    def export_csv(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Results CSV", "kinematics_results.csv", "CSV Files (*.csv)")
        if file_name:
            try:
                with open(file_name, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time_s", "Raw_Theta_deg", "Interpolated_Theta_deg", "Smooth_Theta_deg", "Velocity_rad_s"])
                    writer.writerows(self.export_data)
                print(f"Data successfully exported to {file_name}")
            except Exception as e:
                print(f"Export failed: {e}")

    

# ==========================================
#             GLOBAL THEME
# ==========================================
dark_stylesheet = """
QMainWindow { background-color: #121212; }
QWidget { color: #E0E0E0; font-family: 'Segoe UI', Helvetica, Arial, sans-serif; }
QPushButton { background-color: #333333; border: 1px solid #555555; padding: 8px; border-radius: 4px; font-weight: bold; }
QPushButton:hover { background-color: #444444; }
QPushButton:pressed { background-color: #555555; }
QPushButton:disabled { background-color: #222222; color: #666666; border: 1px solid #333333; }
QSlider::groove:horizontal { border: 1px solid #444; height: 8px; background: #333; border-radius: 4px; }
QSlider::handle:horizontal { background: #4DA8DA; width: 14px; margin: -3px 0; border-radius: 7px; }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    app.setStyleSheet(dark_stylesheet) 
    
    window = HoopTrackerApp()
    window.show()
    sys.exit(app.exec())
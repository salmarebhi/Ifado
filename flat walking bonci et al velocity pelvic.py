# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:19:43 2025

@author: Rebhi
"""

# -*- coding: utf-8 -*-
"""
Enhanced Bonci et al. 2022 Method M10 - Gold Standard Gait Event Detection
Modified to show MIDDLE segment of data instead of beginning
Uses proper pelvis markers for anatomical reference frame
Created on Thu Oct 16 2025
@author: Rebhi (Enhanced with actual pelvis markers + Middle Segment Display)
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

def butter_lowpass_filter(data, cutoff=7, fs=100, order=4):
    """Apply zero-lag fourth-order Butterworth filter as per Bonci et al."""
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

def calculate_pelvis_reference_frame(left_front_hip, right_front_hip, left_back_hip, right_back_hip):
    """
    Calculate proper anatomical reference frame from actual pelvis markers
    Based on Bonci et al. 2022 methodology
    """
    # Calculate pelvis center
    pelvis_center = (left_front_hip + right_front_hip + left_back_hip + right_back_hip) / 4
    
    # Anterior-posterior axis: from back hip midpoint to front hip midpoint
    front_midpoint = (left_front_hip + right_front_hip) / 2
    back_midpoint = (left_back_hip + right_back_hip) / 2
    ap_direction = front_midpoint - back_midpoint
    ap_unit = ap_direction / (np.linalg.norm(ap_direction, axis=1)[:, np.newaxis] + 1e-10)
    
    # Medio-lateral axis: from left hip midpoint to right hip midpoint
    left_midpoint = (left_front_hip + left_back_hip) / 2
    right_midpoint = (right_front_hip + right_back_hip) / 2
    ml_direction = right_midpoint - left_midpoint
    ml_unit = ml_direction / (np.linalg.norm(ml_direction, axis=1)[:, np.newaxis] + 1e-10)
    
    # Vertical axis: cross product of AP and ML
    v_unit = np.cross(ap_unit, ml_unit)
    v_unit = v_unit / (np.linalg.norm(v_unit, axis=1)[:, np.newaxis] + 1e-10)
    
    return pelvis_center, ap_unit, ml_unit, v_unit

def transform_to_anatomical_frame(marker_pos, pelvis_center, ap_unit, ml_unit, v_unit):
    """Transform marker coordinates to pelvis-based anatomical reference frame"""
    relative_pos = marker_pos - pelvis_center
    
    # Project onto anatomical axes
    ap_displacement = np.sum(relative_pos * ap_unit, axis=1)
    ml_displacement = np.sum(relative_pos * ml_unit, axis=1)
    v_displacement = np.sum(relative_pos * v_unit, axis=1)
    
    return ap_displacement, ml_displacement, v_displacement

def calc_3d_velocity(x, y, z, t):
    """Calculate 3D velocity magnitude"""
    pos = np.stack([x, y, z], axis=1)
    dt = np.mean(np.diff(t))
    vel = np.gradient(pos, dt, axis=0)
    return np.linalg.norm(vel, axis=1)

def zeni_position_method(heel_ap, toe_ap, min_distance=30):
    """Zeni et al. position-based method for initial gait event detection"""
    # Heel strike: maximum anterior-posterior displacement
    heel_peaks, _ = find_peaks(heel_ap, distance=min_distance, prominence=0.005)
    
    # Toe-off: minimum anterior-posterior displacement (invert signal to find minima as peaks)
    toe_peaks, _ = find_peaks(-toe_ap, distance=min_distance, prominence=0.005)
    
    return heel_peaks, toe_peaks

def velocity_refinement_method(heel_vel, toe_vel, t, walking_speed, heel_initial, toe_initial, search_window=15):
    """Bonci et al. velocity-based refinement method with adaptive thresholds"""
    # Adaptive thresholds based on walking speed (key innovation of Bonci et al.)
    toe_velocity_threshold = 0.8 * walking_speed
    heel_velocity_threshold = 0.5 * walking_speed
    
    refined_heel_strikes = []
    refined_toe_offs = []
    
    # Refine heel strikes
    for hs in heel_initial:
        start_idx = max(0, hs - search_window)
        end_idx = min(len(heel_vel), hs + search_window)
        window_vel = heel_vel[start_idx:end_idx]
        
        # Find velocity peaks above adaptive threshold
        peaks, _ = find_peaks(window_vel, height=heel_velocity_threshold, distance=5)
        
        if len(peaks) > 0:
            # Choose peak closest to initial position-based detection
            closest_peak_idx = np.argmin(np.abs(peaks - search_window))
            refined_heel_strikes.append(start_idx + peaks[closest_peak_idx])
        else:
            refined_heel_strikes.append(hs)  # Keep original if no suitable velocity peak found
    
    # Refine toe-offs
    for to in toe_initial:
        start_idx = max(0, to - search_window)
        end_idx = min(len(toe_vel), to + search_window)
        window_vel = toe_vel[start_idx:end_idx]
        
        # Find velocity peaks above adaptive threshold
        peaks, _ = find_peaks(window_vel, height=toe_velocity_threshold, distance=5)
        
        if len(peaks) > 0:
            # Choose peak closest to initial position-based detection
            closest_peak_idx = np.argmin(np.abs(peaks - search_window))
            refined_toe_offs.append(start_idx + peaks[closest_peak_idx])
        else:
            refined_toe_offs.append(to)  # Keep original if no suitable velocity peak found
    
    return np.array(refined_heel_strikes), np.array(refined_toe_offs)

def main():
    """
    Complete Bonci et al. 2022 Method M10 implementation with proper pelvis markers
    MODIFIED TO SHOW MIDDLE SEGMENT OF DATA
    """
    
    # UPDATE THIS WITH YOUR ACTUAL FILE PATH
    file_path = r"C:\Users\Rebhi\Desktop\internship\raw data grail\WaS2_17\motion tracker\WaS2_017_walk_flat_2_marker+forceplate_data_2025-08-08_1215300001.txt"
    
    print("="*70)
    print("BONCI ET AL. 2022 METHOD M10 - MIDDLE SEGMENT ANALYSIS")
    print("="*70)
    
    # Load data - handle both Excel and txt formats
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            print("✓ Excel file loaded successfully")
        else:
            df = pd.read_csv(file_path, sep='\t')
            print("✓ Text file loaded successfully")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None, None, None, None
    
    print(f"Data shape: {df.shape}")
    
    # Extract timestamps and calculate sampling frequency
    t = df['TimeStamp'].values
    fs = 1 / np.mean(np.diff(t))
    print(f"✓ Sampling frequency: {fs:.1f} Hz")
    print(f"✓ Total duration: {t[-1]-t[0]:.1f} seconds")
    
    # ===============================
    # NEW SEGMENT SELECTION PARAMETERS
    # ===============================
    
    # Choose which segment to analyze (MODIFY THESE VALUES)
    segment_start_percent = 60  # Start at 40% through the data (middle)
    segment_duration_seconds = 20  # Show 20 seconds of data
    
    # Calculate segment indices
    total_duration = t[-1] - t[0]
    start_time = t[0] + (segment_start_percent / 100) * total_duration
    end_time = start_time + segment_duration_seconds
    
    # Find corresponding indices
    start_idx = np.argmin(np.abs(t - start_time))
    end_idx = np.argmin(np.abs(t - end_time))
    
    print(f"✓ Analyzing segment from {start_time:.1f}s to {end_time:.1f}s")
    print(f"✓ Segment position: {segment_start_percent}% through data")
    print(f"✓ Segment duration: {segment_duration_seconds} seconds")
    
    # Extract pelvis marker positions (FULL DATA - needed for filtering)
    left_front_hip = np.stack([df['LeftFrontHip.PosX'].values, 
                               df['LeftFrontHip.PosY'].values, 
                               df['LeftFrontHip.PosZ'].values], axis=1)
    
    right_front_hip = np.stack([df['RightFrontHip.PosX'].values, 
                                df['RightFrontHip.PosY'].values, 
                                df['RightFrontHip.PosZ'].values], axis=1)
    
    left_back_hip = np.stack([df['LeftBackHip.PosX'].values, 
                              df['LeftBackHip.PosY'].values, 
                              df['LeftBackHip.PosZ'].values], axis=1)
    
    right_back_hip = np.stack([df['RightBackHip.PosX'].values, 
                               df['RightBackHip.PosY'].values, 
                               df['RightBackHip.PosZ'].values], axis=1)
    
    # Extract foot marker positions (FULL DATA)
    heel_pos = np.stack([df['RightHeel.PosX'].values, 
                        df['RightHeel.PosY'].values, 
                        df['RightHeel.PosZ'].values], axis=1)
    
    toe_pos = np.stack([df['RightToe.PosX'].values, 
                       df['RightToe.PosY'].values, 
                       df['RightToe.PosZ'].values], axis=1)
    
    print("✓ All marker positions extracted")
    
    # Apply zero-lag Butterworth filtering (FULL DATA - must filter entire signal)
    print("Applying 4th-order Butterworth filter at 7 Hz...")
    
    left_front_hip_filt = np.array([butter_lowpass_filter(left_front_hip[:, i], cutoff=7, fs=fs) 
                                   for i in range(3)]).T
    right_front_hip_filt = np.array([butter_lowpass_filter(right_front_hip[:, i], cutoff=7, fs=fs) 
                                    for i in range(3)]).T
    left_back_hip_filt = np.array([butter_lowpass_filter(left_back_hip[:, i], cutoff=7, fs=fs) 
                                  for i in range(3)]).T
    right_back_hip_filt = np.array([butter_lowpass_filter(right_back_hip[:, i], cutoff=7, fs=fs) 
                                   for i in range(3)]).T
    
    heel_pos_filt = np.array([butter_lowpass_filter(heel_pos[:, i], cutoff=7, fs=fs) 
                             for i in range(3)]).T
    toe_pos_filt = np.array([butter_lowpass_filter(toe_pos[:, i], cutoff=7, fs=fs) 
                            for i in range(3)]).T
    
    print("✓ All signals filtered")
    
    # Calculate proper anatomical reference frame from pelvis markers (FULL DATA)
    print("Calculating anatomical reference frame from pelvis markers...")
    pelvis_center, ap_unit, ml_unit, v_unit = calculate_pelvis_reference_frame(
        left_front_hip_filt, right_front_hip_filt, left_back_hip_filt, right_back_hip_filt)
    
    print("✓ Pelvis-based anatomical reference frame established")
    
    # Transform foot markers to anatomical coordinate system (FULL DATA)
    heel_ap, heel_ml, heel_v = transform_to_anatomical_frame(
        heel_pos_filt, pelvis_center, ap_unit, ml_unit, v_unit)
    toe_ap, toe_ml, toe_v = transform_to_anatomical_frame(
        toe_pos_filt, pelvis_center, ap_unit, ml_unit, v_unit)
    
    print("✓ Foot markers transformed to anatomical frame")
    
    # Calculate 3D velocities (FULL DATA)
    heel_vel = calc_3d_velocity(heel_pos_filt[:, 0], heel_pos_filt[:, 1], heel_pos_filt[:, 2], t)
    toe_vel = calc_3d_velocity(toe_pos_filt[:, 0], toe_pos_filt[:, 1], toe_pos_filt[:, 2], t)
    
    # Calculate walking speed for adaptive thresholds (FULL DATA)
    pelvis_vel = calc_3d_velocity(pelvis_center[:, 0], pelvis_center[:, 1], pelvis_center[:, 2], t)
    walking_speed = np.median(pelvis_vel[pelvis_vel > 0.1])  # Remove near-zero velocities
    
    print(f"✓ Walking speed estimated: {walking_speed:.3f} m/s")
    
    # Initial detection using Zeni et al. position-based method (FULL DATA)
    print("Step 1/2: Position-based initial detection (Zeni et al.)...")
    heel_initial, toe_initial = zeni_position_method(heel_ap, toe_ap)
    print(f"✓ Initial detections - Heel strikes: {len(heel_initial)}, Toe-offs: {len(toe_initial)}")
    
    # Velocity-based refinement with adaptive thresholds (FULL DATA)
    print("Step 2/2: Velocity-based refinement with adaptive thresholds...")
    heel_events, toe_events = velocity_refinement_method(
        heel_vel, toe_vel, t, walking_speed, heel_initial, toe_initial)
    print(f"✓ Refined detections - Heel strikes: {len(heel_events)}, Toe-offs: {len(toe_events)}")
    
    # ===============================
    # FILTER EVENTS FOR SELECTED SEGMENT
    # ===============================
    
    # Filter events to only those within the selected segment
    heel_events_segment = heel_events[(heel_events >= start_idx) & (heel_events <= end_idx)]
    toe_events_segment = toe_events[(toe_events >= start_idx) & (toe_events <= end_idx)]
    heel_initial_segment = heel_initial[(heel_initial >= start_idx) & (heel_initial <= end_idx)]
    toe_initial_segment = toe_initial[(toe_initial >= start_idx) & (toe_initial <= end_idx)]
    
    print(f"✓ Events in selected segment - Heel strikes: {len(heel_events_segment)}, Toe-offs: {len(toe_events_segment)}")
    
    # Define plotting range (selected segment with small buffer)
    buffer_frames = 100
    plot_start = max(0, start_idx - buffer_frames)
    plot_end = min(len(t), end_idx + buffer_frames)
    show_idx = slice(plot_start, plot_end)
    
    # Create comprehensive visualization
    print("Creating detailed visualization for MIDDLE SEGMENT...")
    
    plt.figure(figsize=(18, 12))
    
    # Subplot 1: AP displacement with position-based detections
    plt.subplot(3, 1, 1)
    plt.plot(t[show_idx], heel_ap[show_idx], label='Right Heel AP Displacement', 
             color='blue', linewidth=1.5, alpha=0.8)
    plt.plot(t[show_idx], toe_ap[show_idx], label='Right Toe AP Displacement', 
             color='green', linewidth=1.5, alpha=0.8)
    
    # Plot initial position-based detections in segment
    if len(heel_initial_segment) > 0:
        plt.scatter(t[heel_initial_segment], heel_ap[heel_initial_segment], 
                   c='red', marker='o', s=80, label='Initial Heel Strike (Zeni)', 
                   edgecolors='darkred', linewidth=1, zorder=5)
    if len(toe_initial_segment) > 0:
        plt.scatter(t[toe_initial_segment], toe_ap[toe_initial_segment], 
                   c='orange', marker='s', s=80, label='Initial Toe-off (Zeni)', 
                   edgecolors='darkorange', linewidth=1, zorder=5)
    
    # Add vertical lines to show selected segment boundaries
    plt.axvline(x=start_time, color='black', linestyle='--', alpha=0.7, label='Segment Start')
    plt.axvline(x=end_time, color='black', linestyle='--', alpha=0.7, label='Segment End')
    
    plt.xlabel('Time (s)')
    plt.ylabel('AP Displacement (m)')
    plt.title(f'Step 1: Position-based Detection - MIDDLE SEGMENT ({start_time:.1f}s to {end_time:.1f}s)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: 3D velocity with refined detections
    plt.subplot(3, 1, 2)
    plt.plot(t[show_idx], heel_vel[show_idx], label='Right Heel 3D Velocity', 
             color='blue', linewidth=1.5, alpha=0.8)
    plt.plot(t[show_idx], toe_vel[show_idx], label='Right Toe 3D Velocity', 
             color='green', linewidth=1.5, alpha=0.8)
    
    # Plot refined velocity-based detections in segment
    if len(heel_events_segment) > 0:
        plt.scatter(t[heel_events_segment], heel_vel[heel_events_segment], 
                   c='red', marker='o', s=80, label='Final Heel Strike (M10)', 
                   edgecolors='darkred', linewidth=1, zorder=5)
    if len(toe_events_segment) > 0:
        plt.scatter(t[toe_events_segment], toe_vel[toe_events_segment], 
                   c='orange', marker='s', s=80, label='Final Toe-off (M10)', 
                   edgecolors='darkorange', linewidth=1, zorder=5)
    
    # Show adaptive velocity thresholds
    toe_threshold = 0.8 * walking_speed
    heel_threshold = 0.5 * walking_speed
    plt.axhline(y=toe_threshold, color='orange', linestyle='--', alpha=0.7, 
               label=f'Adaptive Toe Threshold ({toe_threshold:.3f} m/s)')
    plt.axhline(y=heel_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Adaptive Heel Threshold ({heel_threshold:.3f} m/s)')
    
    # Add vertical lines for segment boundaries
    plt.axvline(x=start_time, color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=end_time, color='black', linestyle='--', alpha=0.7)
    
    plt.xlabel('Time (s)')
    plt.ylabel('3D Velocity (m/s)')
    plt.title(f'Step 2: Velocity-based Refinement - MIDDLE SEGMENT ({start_time:.1f}s to {end_time:.1f}s)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Pelvis reference frame visualization
    plt.subplot(3, 1, 3)
    plt.plot(t[show_idx], pelvis_vel[show_idx], label='Pelvis 3D Velocity', 
             color='purple', linewidth=2, alpha=0.8)
    plt.axhline(y=walking_speed, color='purple', linestyle='-', alpha=0.7, 
               label=f'Median Walking Speed ({walking_speed:.3f} m/s)')
    
    # Add vertical lines for segment boundaries
    plt.axvline(x=start_time, color='black', linestyle='--', alpha=0.7)
    plt.axvline(x=end_time, color='black', linestyle='--', alpha=0.7)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Pelvis-based Walking Speed Reference - MIDDLE SEGMENT ({start_time:.1f}s to {end_time:.1f}s)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Results summary for the selected segment
    print("\n" + "="*70)
    print("BONCI ET AL. 2022 METHOD M10 - MIDDLE SEGMENT RESULTS")
    print("="*70)
    print(f"📊 Segment Analysis:")
    print(f"   • Segment time range: {start_time:.1f}s to {end_time:.1f}s")
    print(f"   • Segment position: {segment_start_percent}% through data")
    print(f"   • Segment duration: {segment_duration_seconds} seconds")
    print(f"   • Sample rate: {fs:.1f} Hz")
    
    print(f"\n🚶 Gait Analysis (Full Data):")
    print(f"   • Walking speed: {walking_speed:.3f} m/s ({walking_speed*3.6:.1f} km/h)")
    print(f"   • Toe velocity threshold: {toe_threshold:.3f} m/s")
    print(f"   • Heel velocity threshold: {heel_threshold:.3f} m/s")
    
    print(f"\n🎯 Event Detection Results (Selected Segment):")
    print(f"   • Position-based (Zeni) - Heel strikes: {len(heel_initial_segment)}, Toe-offs: {len(toe_initial_segment)}")
    print(f"   • Velocity-refined (M10) - Heel strikes: {len(heel_events_segment)}, Toe-offs: {len(toe_events_segment)}")
    
    if len(heel_events_segment) > 1:
        stride_times = np.diff(t[heel_events_segment])
        avg_stride_time = np.mean(stride_times)
        stride_freq = 1 / avg_stride_time
        cadence = stride_freq * 60  # steps per minute
        
        print(f"\n📈 Gait Parameters (Selected Segment):")
        print(f"   • Average stride time: {avg_stride_time:.3f} s")
        print(f"   • Stride frequency: {stride_freq:.2f} Hz")
        print(f"   • Cadence: {cadence:.0f} steps/min")
    
    print(f"\n✅ Analyzing MIDDLE segment instead of beginning provides better steady-state gait analysis")
    print("="*70)
    
    return heel_events, toe_events, heel_vel, toe_vel, t

if __name__ == '__main__':
    heel_events, toe_events, heel_vel, toe_vel, t = main()

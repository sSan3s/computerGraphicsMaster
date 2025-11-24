"""
MotionBERT Adapter for 3D Human Pose Estimation
Converts 2D keypoints sequence to 3D coordinates
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import os
import sys

class MotionBERTAdapter:
    """
    MotionBERT wrapper for lifting 2D poses to 3D
    Paper: https://github.com/Walter0807/MotionBERT
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.window_size = 243  # MotionBERT default window
        
        # For now, we'll use a simple baseline 3D lifting
        # Real implementation would load actual MotionBERT checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_motionbert(checkpoint_path)
        else:
            print("[WARNING] MotionBERT checkpoint not found. Using baseline 3D estimation.")
            self.use_baseline = True
    
    def _load_motionbert(self, checkpoint_path):
        """Load actual MotionBERT model (placeholder for real implementation)"""
        # This would load the real model
        # For now, we'll implement a baseline method
        self.use_baseline = True
    
    def _baseline_3d_lift(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        Simple baseline 3D lifting using heuristics
        Input: [17, 2] or [17, 3] (x, y, [conf])
        Output: [17, 3] (x, y, z)
        """
        if keypoints_2d.shape[1] == 3:
            keypoints_2d = keypoints_2d[:, :2]  # Remove confidence
        
        keypoints_3d = np.zeros((17, 3))
        keypoints_3d[:, :2] = keypoints_2d
        
        # Heuristic depth estimation based on joint hierarchy
        # Hip center as origin
        hip_center = (keypoints_2d[11] + keypoints_2d[12]) / 2  # Left/Right hip average
        
        # Depth values based on typical human pose
        depth_map = {
            0: -0.15,  # Nose (slightly forward)
            1: -0.1,   # Left eye
            2: -0.1,   # Right eye
            3: -0.1,   # Left ear
            4: -0.1,   # Right ear
            5: 0.0,    # Left shoulder
            6: 0.0,    # Right shoulder
            7: -0.05,  # Left elbow
            8: -0.05,  # Right elbow
            9: -0.1,   # Left wrist
            10: -0.1,  # Right wrist
            11: 0.1,   # Left hip
            12: 0.1,   # Right hip
            13: 0.05,  # Left knee
            14: 0.05,  # Right knee
            15: 0.0,   # Left ankle
            16: 0.0,   # Right ankle
        }
        
        # Apply depth and normalize to hip center
        for i in range(17):
            keypoints_3d[i, 0] = (keypoints_2d[i, 0] - hip_center[0]) / 100.0  # Normalize x
            keypoints_3d[i, 1] = (keypoints_2d[i, 1] - hip_center[1]) / 100.0  # Normalize y
            keypoints_3d[i, 2] = depth_map.get(i, 0.0)
        
        return keypoints_3d
    
    def process_sequence(self, persons_sequence: List[List[Dict]]) -> List[List[Dict]]:
        """
        Process a sequence of frames with person detections
        Input: List of frames, each containing list of persons with 2D keypoints
        Output: Same structure but with added 3D keypoints
        """
        output_sequence = []
        
        for frame_persons in persons_sequence:
            frame_output = []
            for person in frame_persons:
                person_copy = person.copy()
                
                if "keypoints" in person:
                    kp_2d = np.array(person["keypoints"])
                    kp_3d = self._baseline_3d_lift(kp_2d)
                    person_copy["keypoints_3d"] = kp_3d.tolist()
                
                frame_output.append(person_copy)
            output_sequence.append(frame_output)
        
        return output_sequence
    
    def process_single_person(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """
        Process single person's 2D keypoints to 3D
        Input: [17, 2] or [17, 3] array
        Output: [17, 3] array with 3D coordinates
        """
        return self._baseline_3d_lift(keypoints_2d)


# Helper function for temporal smoothing
def smooth_3d_poses(poses_3d: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply temporal smoothing to 3D pose sequence
    Input: [T, 17, 3] pose sequence
    Output: [T, 17, 3] smoothed sequence
    """
    from scipy.ndimage import uniform_filter1d
    
    if len(poses_3d) < window:
        return poses_3d
    
    smoothed = uniform_filter1d(poses_3d, size=window, axis=0, mode='nearest')
    return smoothed
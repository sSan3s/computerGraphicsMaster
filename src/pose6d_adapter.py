"""
6D Pose Estimation Module
Estimates 3D position and rotation for detected objects
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

class Pose6DEstimator:
    """
    6D Pose Estimation for objects
    Estimates [x, y, z] position and [roll, pitch, yaw] rotation
    """
    
    def __init__(self, model_name: str = "baseline", checkpoint_path: Optional[str] = None):
        self.model_name = model_name
        self.use_baseline = True  # For now, use heuristic baseline
        
        # Object size priors (in meters) for depth estimation
        self.object_sizes = {
            "cup": 0.08,
            "bottle": 0.25,
            "cell phone": 0.15,
            "laptop": 0.35,
            "keyboard": 0.45,
            "mouse": 0.10,
            "book": 0.20,
            "chair": 1.0,
            "couch": 2.0,
            "table": 1.2,
            "default": 0.30
        }
        
        # Camera intrinsics (default values, should be calibrated)
        self.fx = 525.0  # Focal length x
        self.fy = 525.0  # Focal length y
        self.cx = 320.0  # Principal point x
        self.cy = 240.0  # Principal point y
    
    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """Set camera intrinsic parameters"""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def estimate_depth_from_bbox(self, bbox: List[float], category: str) -> float:
        """
        Estimate object depth from bounding box size
        Uses object size priors and perspective projection
        """
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        
        # Get expected real-world size
        real_size = self.object_sizes.get(category.lower(), self.object_sizes["default"])
        
        # Estimate depth using perspective projection
        # depth = (real_size * focal_length) / bbox_height_pixels
        depth = (real_size * self.fy) / max(bbox_height, 1)
        
        # Clamp to reasonable range (0.3m to 5m)
        depth = np.clip(depth, 0.3, 5.0)
        
        return depth
    
    def pixel_to_3d(self, pixel_x: float, pixel_y: float, depth: float) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to 3D world coordinates
        Using pinhole camera model
        """
        x = (pixel_x - self.cx) * depth / self.fx
        y = (pixel_y - self.cy) * depth / self.fy
        z = depth
        
        return x, y, z
    
    def estimate_rotation_heuristic(self, bbox: List[float], category: str) -> Tuple[float, float, float]:
        """
        Heuristic rotation estimation based on object type and bbox aspect ratio
        Returns [roll, pitch, yaw] in radians
        """
        x1, y1, x2, y2 = bbox
        aspect_ratio = (x2 - x1) / max(y2 - y1, 1)
        
        # Default upright orientation
        roll = 0.0
        pitch = 0.0
        yaw = 0.0
        
        # Adjust based on category and aspect ratio
        if category.lower() in ["bottle", "cup"]:
            # Usually upright
            pitch = 0.0
        elif category.lower() in ["laptop", "keyboard", "book"]:
            # Usually flat on table
            pitch = np.radians(-15)  # Slight tilt
            
            # Estimate yaw from aspect ratio
            if aspect_ratio > 1.5:
                yaw = 0.0  # Facing forward
            elif aspect_ratio < 0.7:
                yaw = np.radians(90)  # Sideways
        
        return roll, pitch, yaw
    
    def estimate_6d_pose(self, image: np.ndarray, bbox: List[float], category: str) -> Dict:
        """
        Estimate 6D pose for an object
        Input:
            image: BGR image
            bbox: [x1, y1, x2, y2]
            category: object category string
        Output:
            Dictionary with position and rotation
        """
        # Get bbox center
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        # Estimate depth
        depth = self.estimate_depth_from_bbox(bbox, category)
        
        # Convert to 3D position
        x, y, z = self.pixel_to_3d(cx, cy, depth)
        
        # Estimate rotation
        roll, pitch, yaw = self.estimate_rotation_heuristic(bbox, category)
        
        return {
            "position": [float(x), float(y), float(z)],
            "rotation": [float(roll), float(pitch), float(yaw)],
            "quaternion": self.euler_to_quaternion(roll, pitch, yaw),
            "confidence": 0.7  # Baseline confidence
        }
    
    def process_objects(self, image: np.ndarray, objects: List[Dict]) -> List[Dict]:
        """
        Process all detected objects to estimate 6D poses
        """
        objects_with_pose = []
        
        for obj in objects:
            obj_copy = obj.copy()
            pose_6d = self.estimate_6d_pose(
                image, 
                obj["bbox"], 
                obj.get("category", "object")
            )
            obj_copy["pose_6d"] = pose_6d
            objects_with_pose.append(obj_copy)
        
        return objects_with_pose
    
    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
        """
        Convert Euler angles to quaternion
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return [float(x), float(y), float(z), float(w)]
    
    @staticmethod
    def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Create 3x3 rotation matrix from Euler angles
        """
        # Roll (X-axis rotation)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch (Y-axis rotation)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw (Z-axis rotation)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        return R
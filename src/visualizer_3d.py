"""
Open3D 3D Visualization Module
Renders human skeletons and objects in 3D space
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Optional, Tuple
import json
import cv2

class Open3DVisualizer:
    """
    3D Visualization using Open3D
    Renders human skeletons, objects, and their interactions
    """
    
    def __init__(self, window_name: str = "HOI 3D Visualization", width: int = 1280, height: int = 720):
        self.window_name = window_name
        self.width = width
        self.height = height
        
        # COCO skeleton connections
        self.skeleton_connections = [
            # Head
            [0, 1], [0, 2], [1, 3], [2, 4],
            # Torso
            [5, 6], [5, 11], [6, 12], [11, 12],
            # Left arm
            [5, 7], [7, 9],
            # Right arm
            [6, 8], [8, 10],
            # Left leg
            [11, 13], [13, 15],
            # Right leg
            [12, 14], [14, 16]
        ]
        
        # Colors (RGB normalized to 0-1)
        self.colors = {
            "skeleton": [0.0, 0.5, 1.0],      # Blue
            "joints": [1.0, 0.0, 0.0],        # Red
            "objects": [0.0, 1.0, 0.0],       # Green
            "interaction": [1.0, 0.0, 1.0],   # Magenta
            "coordinate_frame": [0.5, 0.5, 0.5]  # Gray
        }
        
        # Object meshes cache
        self.object_meshes = {}
        self._create_primitive_meshes()
        
        # Visualization state
        self.vis = None
        self.first_frame = True
        self.geometries = []
    
    def _create_primitive_meshes(self):
        """Create primitive 3D shapes for different object categories"""
        # Cup - cylinder
        cup = o3d.geometry.TriangleMesh.create_cylinder(radius=0.04, height=0.10)
        cup.paint_uniform_color([0.8, 0.6, 0.3])
        self.object_meshes["cup"] = cup
        
        # Bottle - cylinder
        bottle = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.25)
        bottle.paint_uniform_color([0.3, 0.6, 0.8])
        self.object_meshes["bottle"] = bottle
        
        # Laptop - box
        laptop = o3d.geometry.TriangleMesh.create_box(width=0.35, height=0.02, depth=0.25)
        laptop.paint_uniform_color([0.5, 0.5, 0.5])
        self.object_meshes["laptop"] = laptop
        
        # Keyboard - box
        keyboard = o3d.geometry.TriangleMesh.create_box(width=0.45, height=0.02, depth=0.15)
        keyboard.paint_uniform_color([0.3, 0.3, 0.3])
        self.object_meshes["keyboard"] = keyboard
        
        # Book - box
        book = o3d.geometry.TriangleMesh.create_box(width=0.20, height=0.03, depth=0.25)
        book.paint_uniform_color([0.7, 0.5, 0.3])
        self.object_meshes["book"] = book
        
        # Chair - simplified mesh
        chair = o3d.geometry.TriangleMesh.create_box(width=0.5, height=1.0, depth=0.5)
        chair.paint_uniform_color([0.6, 0.4, 0.2])
        self.object_meshes["chair"] = chair
        
        # Default object - sphere
        default = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        default.paint_uniform_color([0.7, 0.7, 0.7])
        self.object_meshes["default"] = default
    
    def create_skeleton(self, keypoints_3d: np.ndarray) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.LineSet]:
        """
        Create skeleton geometry from 3D keypoints
        Input: [17, 3] array of 3D joint positions
        Returns: (joint_cloud, bone_lines)
        """
        # Create point cloud for joints
        joint_cloud = o3d.geometry.PointCloud()
        joint_cloud.points = o3d.utility.Vector3dVector(keypoints_3d)
        joint_cloud.paint_uniform_color(self.colors["joints"])
        
        # Create lines for bones
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(keypoints_3d)
        line_set.lines = o3d.utility.Vector2iVector(self.skeleton_connections)
        line_set.paint_uniform_color(self.colors["skeleton"])
        
        return joint_cloud, line_set
    
    def create_object_mesh(self, category: str, position: List[float], rotation: List[float]) -> o3d.geometry.TriangleMesh:
        """
        Create and position object mesh
        Input:
            category: object type
            position: [x, y, z] 
            rotation: [roll, pitch, yaw] in radians
        """
        # Get base mesh
        base_mesh = self.object_meshes.get(category.lower(), self.object_meshes["default"])
        mesh = o3d.geometry.TriangleMesh(base_mesh)
        
        # Apply rotation
        from src.pose6d_adapter import Pose6DEstimator
        R = Pose6DEstimator.rotation_matrix_from_euler(rotation[0], rotation[1], rotation[2])
        mesh.rotate(R, center=(0, 0, 0))
        
        # Apply translation
        mesh.translate(position)
        
        return mesh
    
    def create_interaction_line(self, start_point: np.ndarray, end_point: np.ndarray) -> o3d.geometry.LineSet:
        """Create line representing interaction between human and object"""
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([start_point, end_point])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.paint_uniform_color(self.colors["interaction"])
        return line
    
    def initialize_window(self):
        """Initialize Open3D visualization window"""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(self.window_name, width=self.width, height=self.height)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(coord_frame)
        
        # Set view control
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_front([1, -0.5, -1])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.7)
    
    def update_frame(self, humans_3d: List[Dict], objects_6d: List[Dict], hoi_relations: List[Dict]):
        """
        Update visualization with new frame data
        Input:
            humans_3d: List of humans with 3D keypoints
            objects_6d: List of objects with 6D poses
            hoi_relations: List of HOI relations
        """
        if self.vis is None:
            self.initialize_window()
        
        # Clear previous geometries
        for geom in self.geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.geometries = []
        
        # Add human skeletons
        for human in humans_3d:
            if "keypoints_3d" in human:
                kp_3d = np.array(human["keypoints_3d"])
                joint_cloud, bone_lines = self.create_skeleton(kp_3d)
                self.vis.add_geometry(joint_cloud, reset_bounding_box=False)
                self.vis.add_geometry(bone_lines, reset_bounding_box=False)
                self.geometries.extend([joint_cloud, bone_lines])
        
        # Add objects
        for obj in objects_6d:
            if "pose_6d" in obj:
                pose = obj["pose_6d"]
                mesh = self.create_object_mesh(
                    obj.get("category", "default"),
                    pose["position"],
                    pose["rotation"]
                )
                self.vis.add_geometry(mesh, reset_bounding_box=False)
                self.geometries.append(mesh)
        
        # Add interaction lines
        for hoi in hoi_relations:
            human_id = hoi["human_id"]
            object_id = hoi["object_id"]
            
            # Find corresponding human and object
            human = next((h for h in humans_3d if h.get("track_id") == human_id), None)
            obj = next((o for o in objects_6d if o.get("track_id") == object_id), None)
            
            if human and obj and "keypoints_3d" in human and "pose_6d" in obj:
                # Use hand position for interaction (wrist joints: 9 and 10)
                kp_3d = np.array(human["keypoints_3d"])
                left_wrist = kp_3d[9] if len(kp_3d) > 9 else kp_3d[0]
                right_wrist = kp_3d[10] if len(kp_3d) > 10 else kp_3d[0]
                
                # Choose closest wrist to object
                obj_pos = np.array(obj["pose_6d"]["position"])
                if np.linalg.norm(left_wrist - obj_pos) < np.linalg.norm(right_wrist - obj_pos):
                    hand_pos = left_wrist
                else:
                    hand_pos = right_wrist
                
                # Create interaction line
                interaction_line = self.create_interaction_line(hand_pos, obj_pos)
                self.vis.add_geometry(interaction_line, reset_bounding_box=False)
                self.geometries.append(interaction_line)
        
        # Update visualization
        self.vis.poll_events()
        self.vis.update_renderer()
        
        if self.first_frame:
            self.vis.reset_view_point(True)
            self.first_frame = False
    
    def close(self):
        """Close visualization window"""
        if self.vis is not None:
            self.vis.destroy_window()
    
    def capture_screen(self) -> np.ndarray:
        """Capture current 3D view as image"""
        if self.vis is None:
            return None
        
        # Capture image from Open3D
        image = self.vis.capture_screen_float_buffer(do_render=True)
        image_np = np.asarray(image)
        image_np = (image_np * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        return image_bgr


class JsonlTo3DVisualizer:
    """
    Read JSONL output and visualize in 3D
    """
    
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.frames = []
        self.metadata = None
        self._load_jsonl()
    
    def _load_jsonl(self):
        """Load JSONL file"""
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if "session" in data:
                    self.metadata = data["session"]
                else:
                    self.frames.append(data)
    
    def visualize(self, fps: float = 30.0):
        """Play back the 3D visualization"""
        vis = Open3DVisualizer()
        
        for frame in self.frames:
            # Convert frame data to 3D format
            humans_3d = frame.get("humans", [])
            objects_6d = frame.get("objects", [])
            hoi_relations = frame.get("hoi", [])
            
            vis.update_frame(humans_3d, objects_6d, hoi_relations)
            
            # Control playback speed
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break
        
        vis.close()
        cv2.destroyAllWindows()
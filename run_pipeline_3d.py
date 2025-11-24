#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline_3d.py
Complete HOI Pipeline with 3D Processing
YOLOv11/12 -> ByteTrack -> HAKE -> MotionBERT (2D→3D) -> 6D Pose -> Open3D Visualization

Usage:
  python run_pipeline_3d.py \
    --source demo.mp4 \
    --yolo yolo12l.pt \
    --pose_weight yolov11l-pose.pt \
    --out result_3d.jsonl \
    --save_vis result_3d.mp4 \
    --enable_3d \
    --show_3d
"""

import os
import cv2
import argparse
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque

# Project imports
from src.detector_yolo import YoloDetector
from src.tracker import ByteTrackPair
from src.hake_adapter import HakeModel
from src.pose_adapter import PoseEstimator
from src.utils_io import JsonlWriter
from src.schemas import FrameRecord, HOI

# 3D processing imports
from src.motionbert_adapter import MotionBERTAdapter
from src.pose6d_adapter import Pose6DEstimator
from src.visualizer_3d import Open3DVisualizer

# ----------------------------
# Helper Functions
# ----------------------------
def _open_video(source: str) -> cv2.VideoCapture:
    try:
        cam_idx = int(source)
        cap = cv2.VideoCapture(cam_idx)
    except ValueError:
        cap = cv2.VideoCapture(source)
    return cap

def _build_meta(source: str, cap: cv2.VideoCapture, yolo_path: str, hake_path: Optional[str], enable_3d: bool) -> dict:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return {
        "id": os.path.basename(str(source)) + "_3d_session",
        "source": source,
        "fps": float(fps),
        "resolution": [W, H],
        "3d_enabled": enable_3d,
        "model": {
            "detector": os.path.basename(yolo_path),
            "hake": os.path.basename(hake_path) if hake_path else "heuristic",
            "tracker": "ByteTrack(supervision)",
            "pose_2d": "YOLO-Pose",
            "pose_3d": "MotionBERT" if enable_3d else None,
            "object_6d": "Baseline6D" if enable_3d else None
        }
    }

def _start_writer(save_path: Optional[str], fps: float, size: Tuple[int, int]):
    if not save_path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    W, H = size
    return cv2.VideoWriter(save_path, fourcc, fps, (W, H))

# Drawing helpers for 2D overlay
def draw_2d_overlay(frame, persons, objects, hoi_list, draw_skeleton=True):
    """Draw 2D visualization overlay"""
    # Skeleton connections
    COCO_EDGES = [
        (0,1), (0,2), (1,3), (2,4), (5,6), (5,11), (6,12), (11,12),
        (5,7), (7,9), (6,8), (8,10), (11,13), (13,15), (12,14), (14,16)
    ]
    
    # Draw humans
    for p in persons:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        hid = p.get("track_id", -1)
        cv2.putText(frame, f"H{hid}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        
        # Draw skeleton if available
        if draw_skeleton and "keypoints" in p:
            kps = p["keypoints"]
            for (a, b) in COCO_EDGES:
                if a < len(kps) and b < len(kps) and kps[a][2] > 0.3 and kps[b][2] > 0.3:
                    cv2.line(frame, 
                            (int(kps[a][0]), int(kps[a][1])), 
                            (int(kps[b][0]), int(kps[b][1])), 
                            (0, 255, 255), 2)
    
    # Draw objects
    for o in objects:
        x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 60), 2)
        oid = o.get("track_id", -1)
        cat = o.get("category", "obj")
        
        # Show 6D pose info if available
        label = f"O{oid}:{cat}"
        if "pose_6d" in o:
            z_depth = o["pose_6d"]["position"][2]
            label += f" Z:{z_depth:.2f}m"
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 60), 2)
    
    # Draw HOI relations
    for h in hoi_list:
        hi = h["human_id"]
        oi = h["object_id"]
        verb = h["verb"]
        
        ph = next((p for p in persons if int(p.get("track_id", -1)) == int(hi)), None)
        po = next((o for o in objects if int(o.get("track_id", -1)) == int(oi)), None)
        
        if ph and po:
            cx1 = int((ph["bbox"][0] + ph["bbox"][2]) / 2)
            cy1 = int((ph["bbox"][1] + ph["bbox"][3]) / 2)
            cx2 = int((po["bbox"][0] + po["bbox"][2]) / 2)
            cy2 = int((po["bbox"][1] + po["bbox"][3]) / 2)
            
            cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)
            mx = int((cx1 + cx2) / 2)
            my = int((cy1 + cy2) / 2)
            cv2.putText(frame, verb, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Complete HOI Pipeline with 3D Processing")
    ap.add_argument("--source", required=True, help="Video file path or camera index")
    ap.add_argument("--yolo", required=True, help="YOLO weight (.pt) path")
    ap.add_argument("--pose_weight", default=None, help="Pose weight (e.g., yolov11l-pose.pt)")
    ap.add_argument("--out", default="output_3d.jsonl", help="Output JSONL path")
    
    # Detection parameters
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence")
    ap.add_argument("--iou", type=float, default=0.5, help="YOLO IoU")
    ap.add_argument("--device", default=None, help="Device (e.g., 'cuda:0' or 'cpu')")
    
    # 3D processing
    ap.add_argument("--enable_3d", action="store_true", help="Enable 3D processing")
    ap.add_argument("--motionbert_ckpt", default=None, help="MotionBERT checkpoint path")
    ap.add_argument("--temporal_window", type=int, default=10, help="Temporal window for 3D lifting")
    
    # Visualization
    ap.add_argument("--save_vis", default=None, help="Save 2D annotated video")
    ap.add_argument("--save_3d_vis", default=None, help="Save 3D visualization video")
    ap.add_argument("--show", action="store_true", help="Display 2D frames")
    ap.add_argument("--show_3d", action="store_true", help="Display 3D visualization")
    
    args = ap.parse_args()

    # Initialize video capture
    cap = _open_video(args.source)
    assert cap.isOpened(), f"Failed to open source: {args.source}"

    # Initialize models
    print("Initializing models...")
    detector = YoloDetector(args.yolo, conf=args.conf, iou=args.iou, device=args.device)
    tracker = ByteTrackPair(object_id_offset=100000)
    hake = HakeModel()  # Using heuristic for now
    pose_est = PoseEstimator(args.pose_weight, conf=args.conf, iou=args.iou, device=args.device) if args.pose_weight else None
    
    # Initialize 3D processors if enabled
    motionbert = None
    pose6d_est = None
    vis_3d = None
    
    if args.enable_3d:
        print("Initializing 3D processing modules...")
        motionbert = MotionBERTAdapter(checkpoint_path=args.motionbert_ckpt, device=args.device or 'cpu')
        pose6d_est = Pose6DEstimator(model_name="baseline")
        
        # Set camera intrinsics (should be calibrated for actual camera)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fx = fy = W  # Approximate focal length
        cx = W / 2
        cy = H / 2
        pose6d_est.set_camera_intrinsics(fx, fy, cx, cy)
        
        if args.show_3d:
            vis_3d = Open3DVisualizer(window_name="HOI 3D Visualization")

    # Setup output writers
    meta = _build_meta(args.source, cap, args.yolo, None, args.enable_3d)
    writer = JsonlWriter(args.out, meta=meta)
    
    fps = meta["fps"]
    W, H = meta["resolution"]
    vis_writer = _start_writer(args.save_vis, fps=fps, size=(W, H)) if args.save_vis else None
    vis_3d_writer = _start_writer(args.save_3d_vis, fps=fps, size=(1280, 720)) if args.save_3d_vis else None
    
    # Temporal buffer for 3D lifting
    temporal_buffer = deque(maxlen=args.temporal_window)
    
    print("Processing video...")
    idx = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # 1) Object Detection
        persons, objects = detector(frame)
        
        # 2) Tracking
        persons, objects = tracker.step(persons, objects)
        
        # 3) 2D Pose Estimation
        if pose_est is not None:
            persons = pose_est(frame, persons)
        
        # 4) 3D Human Pose Lifting
        if args.enable_3d and motionbert is not None:
            for person in persons:
                if "keypoints" in person:
                    kp_2d = np.array(person["keypoints"])
                    kp_3d = motionbert.process_single_person(kp_2d)
                    person["keypoints_3d"] = kp_3d.tolist()
        
        # 5) 6D Object Pose Estimation
        if args.enable_3d and pose6d_est is not None:
            objects = pose6d_est.process_objects(frame, objects)
        
        # 6) HOI Inference
        hoi_raw = hake.infer(frame, persons, objects)
        
        # 7) Build HOI list with tracked IDs
        hoi_list = []
        for h in hoi_raw:
            hi, oi = h["human_idx"], h["object_idx"]
            if hi < len(persons) and oi < len(objects):
                hoi_list.append(
                    HOI(
                        human_id=int(persons[hi].get("track_id", -1)),
                        object_id=int(objects[oi].get("track_id", -1)),
                        verb=h["verb"],
                        score=float(h["score"]),
                        part=h.get("part"),
                        part_score=float(h.get("part_score", 0)) if h.get("part_score") else None,
                        triplet=["person", h["verb"], objects[oi].get("category", "object")],
                    ).model_dump()
                )
        
        # 8) 2D Visualization
        frame_vis = frame.copy()
        draw_2d_overlay(frame_vis, persons, objects, hoi_list)
        
        # 9) 3D Visualization
        if args.enable_3d and vis_3d is not None:
            vis_3d.update_frame(persons, objects, hoi_list)
            
            if vis_3d_writer is not None:
                # Capture 3D view and save
                frame_3d = vis_3d.capture_screen()
                if frame_3d is not None:
                    vis_3d_writer.write(frame_3d)
        
        # 10) Write JSONL
        rec = FrameRecord(
            frame_index=idx,
            timestamp_ms=int((idx / max(1.0, fps)) * 1000),
            humans=[
                {
                    "track_id": p.get("track_id"),
                    "bbox_xyxy": p["bbox"],
                    "score": float(p.get("score", 1.0)),
                    "keypoints_3d": p.get("keypoints_3d") if args.enable_3d else None
                }
                for p in persons
            ],
            objects=[
                {
                    "track_id": o.get("track_id"),
                    "bbox_xyxy": o["bbox"],
                    "category": o.get("category", "object"),
                    "score": float(o.get("score", 1.0)),
                    "pose_6d": o.get("pose_6d") if args.enable_3d else None
                }
                for o in objects
            ],
            hoi=hoi_list,
        ).model_dump()
        writer.write_line(rec)
        
        # 11) Display/Save
        if vis_writer is not None:
            vis_writer.write(frame_vis)
        
        if args.show:
            cv2.imshow("HOI Detection", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        idx += 1
        
        # Progress indicator
        if idx % 30 == 0:
            print(f"Processed {idx} frames...")
    
    # Cleanup
    writer.close()
    if vis_writer is not None:
        vis_writer.release()
    if vis_3d_writer is not None:
        vis_3d_writer.release()
    if vis_3d is not None:
        vis_3d.close()
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Processing complete!")
    print(f"   JSONL output: {args.out}")
    if args.save_vis:
        print(f"   2D video: {args.save_vis}")
    if args.save_3d_vis:
        print(f"   3D video: {args.save_3d_vis}")


if __name__ == "__main__":
    main()
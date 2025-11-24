# Human-Object Interaction (HOI) 3D Pipeline

## Project Overview

This project implements a complete pipeline for extracting and visualizing Human-Object Interactions (HOI) from 2D videos in 3D space.

### Pipeline Architecture
```
Video Input
    ↓
[2D Detection & Tracking]
├─ YOLO: Object detection (humans + objects)
└─ ByteTrack: Multi-object tracking
    ↓
[Pose Estimation]
├─ YOLO-Pose: 2D keypoint extraction (17 joints)
└─ MotionBERT: 2D→3D pose lifting
    ↓
[3D Reconstruction]
├─ 6D Pose Estimation: Object position + rotation
└─ HAKE: HOI relationship inference
    ↓
[Visualization]
├─ 2D Overlay: Annotated video with bounding boxes
└─ Open3D: 3D scene reconstruction
    ↓
[Output]
├─ JSONL: Structured data for each frame
├─ 2D Video: Annotated visualization
└─ 3D Video: 3D scene rendering
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10.13

### Installation

```bash
# 1. Clone repository
git clone https://github.com/leasw/123whddnjs.git
cd 123whddnjs

# 2. Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip wheel setuptools
pip install -r requirements_updated.txt
```

### Download Model Weights

```bash
# YOLO object detection
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt -O yolo12l.pt

# YOLO pose estimation
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt
```

## 💻 Usage

### Basic 2D Pipeline
```bash
python run_pipeline.py \
  --source test_video/demo_3.mp4 \
  --yolo yolo12l.pt \
  --pose_weight yolo11l-pose.pt \
  --out results/output_2d.jsonl \
  --save_vis results/output_2d.mp4 \
  --draw_parts \
  --show
```

### Full 3D Pipeline
```bash
python run_pipeline_3d.py \
  --source test_video/demo_3.mp4 \
  --yolo yolo12l.pt \
  --pose_weight yolo11l-pose.pt \
  --out results/output_3d.jsonl \
  --save_vis results/output_2d_vis.mp4 \
  --save_3d_vis results/output_3d_render.mp4 \
  --enable_3d \
  --show_3d \
  --temporal_window 10
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--source` | Input video path or camera index | Required |
| `--yolo` | YOLO model weights path | Required |
| `--pose_weight` | Pose estimation model path | None |
| `--out` | Output JSONL file path | output.jsonl |
| `--save_vis` | Save 2D annotated video | None |
| `--save_3d_vis` | Save 3D rendered video | None |
| `--enable_3d` | Enable 3D processing | False |
| `--show` | Display 2D visualization | False |
| `--show_3d` | Display 3D visualization | False |
| `--conf` | Detection confidence threshold | 0.25 |
| `--iou` | Detection IoU threshold | 0.5 |
| `--temporal_window` | Window size for temporal smoothing | 10 |

## 📁 Project Structure

```
├── src/
│   ├── detector_yolo.py      # YOLO object detection
│   ├── tracker.py             # ByteTrack multi-object tracking
│   ├── pose_adapter.py        # 2D pose estimation
│   ├── hake_adapter.py        # HOI relationship inference
│   ├── motionbert_adapter.py  # 2D→3D pose lifting [NEW]
│   ├── pose6d_adapter.py      # 6D object pose estimation [NEW]
│   ├── visualizer_3d.py       # Open3D 3D visualization [NEW]
│   ├── schemas.py             # Data schemas
│   └── utils_io.py            # I/O utilities
├── test_video/                # Sample videos
├── run_pipeline.py            # 2D pipeline
├── run_pipeline_3d.py         # 3D pipeline [NEW]
└── test_3d_pipeline.sh        # Test script [NEW]
```

## 📊 Output Format

### JSONL Structure
```json
{
  "frame_index": 0,
  "timestamp_ms": 0,
  "humans": [
    {
      "track_id": 1,
      "bbox_xyxy": [x1, y1, x2, y2],
      "score": 0.95,
      "keypoints_3d": [[x, y, z], ...]  // 17 joints
    }
  ],
  "objects": [
    {
      "track_id": 100001,
      "bbox_xyxy": [x1, y1, x2, y2],
      "category": "cup",
      "score": 0.89,
      "pose_6d": {
        "position": [x, y, z],
        "rotation": [roll, pitch, yaw],
        "quaternion": [x, y, z, w]
      }
    }
  ],
  "hoi": [
    {
      "human_id": 1,
      "object_id": 100001,
      "verb": "hold",
      "score": 0.80,
      "part": "hand",
      "triplet": ["person", "hold", "cup"]
    }
  ]
}
```

## 🔧 Module Details

### New 3D Processing Modules

1. **MotionBERT Adapter** (`src/motionbert_adapter.py`)
   - Converts 2D keypoints to 3D coordinates
   - Temporal smoothing for consistency
   - Baseline heuristic method included

2. **6D Pose Estimator** (`src/pose6d_adapter.py`)
   - Estimates object 3D position from bbox
   - Heuristic rotation estimation
   - Camera intrinsics calibration support

3. **Open3D Visualizer** (`src/visualizer_3d.py`)
   - Real-time 3D scene rendering
   - Human skeleton visualization
   - Object mesh rendering
   - HOI relationship visualization

## 📈 Performance

| Component | FPS | GPU Memory |
|-----------|-----|------------|
| YOLO Detection | ~30 | 2GB |
| Pose Estimation | ~25 | 1GB |
| 3D Processing | ~20 | 1GB |
| Open3D Rendering | ~15 | 512MB |

## 🎯 Future Improvements

- [ ] Real HAKE model integration (currently using heuristics)
- [ ] MotionBERT checkpoint integration
- [ ] Advanced 6D pose models (OnePose++, CosyPose)
- [ ] Multi-person 3D pose optimization
- [ ] Real-time optimization
- [ ] Web interface for visualization

## 📝 References

- YOLO: [Ultralytics](https://github.com/ultralytics/ultralytics)
- ByteTrack: [Supervision](https://github.com/roboflow/supervision)
- MotionBERT: [Paper](https://github.com/Walter0807/MotionBERT)
- HAKE: [Paper](http://hake-mvig.cn/)
- Open3D: [Documentation](http://www.open3d.org/)

## 📜 License

MIT License

## 👥 Team

- Boo Seokkyeong
- Kim Taehyeon
- Yoo Sunghwan
- Lee Jongwon

Course: Topics in Computer Graphics  
Professor: Sabina Umirzakova

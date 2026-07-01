import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import io
import json
import random
import math
import tempfile
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CowSperm CASA Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0a0e1a; }

    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1628 100%); }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #111827 0%, #1a2238 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,150,255,0.08);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: #5b7fa6; margin-bottom: 6px; }
    .kpi-value { font-size: 32px; font-weight: 700; color: #e8f0fe; line-height: 1; }
    .kpi-sub { font-size: 12px; color: #4a90d9; margin-top: 4px; }

    /* Status badges */
    .badge-progressive { background: #0d3320; color: #34d399; border: 1px solid #34d39960; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600; }
    .badge-nonprogressive { background: #2d2500; color: #fbbf24; border: 1px solid #fbbf2460; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600; }
    .badge-immotile { background: #2d0e0e; color: #f87171; border: 1px solid #f8717160; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600; }

    /* Section headers */
    .section-header {
        font-size: 13px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase;
        color: #4a90d9; border-bottom: 1px solid #1e3a5f; padding-bottom: 8px; margin-bottom: 16px;
    }

    /* Summary panel */
    .summary-panel {
        background: linear-gradient(135deg, #0d1a2e 0%, #0f2040 100%);
        border: 1px solid #1e3a8a;
        border-radius: 14px;
        padding: 24px;
    }

    /* Assessment block */
    .assessment-block {
        background: linear-gradient(135deg, #051a0a 0%, #082010 100%);
        border: 1px solid #166534;
        border-left: 4px solid #34d399;
        border-radius: 8px;
        padding: 16px 20px;
        color: #d1fae5;
        font-size: 14px;
        line-height: 1.7;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1526; border-right: 1px solid #1e3a5f; }
    [data-testid="stSidebar"] .stMarkdown h3 { color: #4a90d9; }

    /* Progress bar */
    .stProgress > div > div { background-color: #4a90d9; }

    /* Metric row */
    .metric-row { display: flex; justify-content: space-between; margin: 6px 0; padding: 6px 0; border-bottom: 1px solid #1a2840; }
    .metric-key { color: #8097b8; font-size: 13px; }
    .metric-val { color: #e8f0fe; font-size: 13px; font-weight: 600; }

    /* Alert */
    .alert-info { background: #0a1f3a; border: 1px solid #1e4080; border-radius: 8px; padding: 12px 16px; color: #93c5fd; font-size: 13px; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #111827, #1a2238);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 14px 18px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpermTrack:
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=150))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=150))
    speeds: deque = field(default_factory=lambda: deque(maxlen=150))
    confidences: deque = field(default_factory=lambda: deque(maxlen=150))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=150))
    motility_class: str = "Unknown"
    color: Tuple = (128, 128, 128)
    frames_seen: int = 0
    last_seen_frame: int = 0

    # CASA metrics
    vcl: float = 0.0
    vsl: float = 0.0
    vap: float = 0.0
    lin: float = 0.0
    str_metric: float = 0.0
    wob: float = 0.0
    alh: float = 0.0
    bcf: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK YOLO DETECTOR (Replace with actual YOLO model in production)
# ═══════════════════════════════════════════════════════════════════════════════
class RealYOLODetector:
    def __init__(self, model_path: str):
        # Initialize your custom YOLO model
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, frame_num: int, conf_threshold: float = 0.4):
        # Run inference
        results = self.model(frame, imgsz=640, conf=conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            detections.append((x1, y1, x2, y2, conf))
        return detections


class MockYOLODetector:
    """
    Simulates YOLO sperm detection.
    Replace detect() with actual ultralytics YOLO inference.
    """
    def __init__(self, num_sperm: int = 30):
        self.num_sperm = num_sperm
        self._init_sperm_entities()

    def _init_sperm_entities(self):
        """Initialize persistent sperm positions for simulation"""
        self.entities = []
        for i in range(self.num_sperm):
            sptype = random.choices(
                ["progressive", "nonprogressive", "immotile"],
                weights=[0.55, 0.25, 0.20]
            )[0]
            self.entities.append({
                "x": random.uniform(60, 560),
                "y": random.uniform(60, 420),
                "dx": random.uniform(2, 8) * (1 if sptype == "progressive" else 0.3) * random.choice([-1, 1]),
                "dy": random.uniform(1, 4) * (1 if sptype == "progressive" else 0.3) * random.choice([-1, 1]),
                "wobble": random.uniform(0.5, 3.0),
                "type": sptype,
                "frame": 0,
            })

    def detect(self, frame: np.ndarray, frame_num: int, conf_threshold: float = 0.4):
        """Returns list of (x1,y1,x2,y2,confidence) detections"""
        H, W = frame.shape[:2]
        detections = []
        for e in self.entities:
            if e["type"] == "progressive":
                e["x"] += e["dx"] + random.gauss(0, 0.5)
                e["y"] += e["dy"] + random.gauss(0, 0.5)
            elif e["type"] == "nonprogressive":
                angle = frame_num * 0.15 + e["wobble"]
                e["x"] += math.cos(angle) * 1.5 + random.gauss(0, 0.3)
                e["y"] += math.sin(angle) * 1.5 + random.gauss(0, 0.3)
            else:
                e["x"] += random.gauss(0, 0.2)
                e["y"] += random.gauss(0, 0.2)

            # Bounce off walls
            if e["x"] < 40 or e["x"] > W - 40: e["dx"] *= -1
            if e["y"] < 40 or e["y"] > H - 40: e["dy"] *= -1
            e["x"] = np.clip(e["x"], 20, W - 20)
            e["y"] = np.clip(e["y"], 20, H - 20)

            # Bounding box (sperm head ~8-14px at 640x640)
            w, h = random.randint(8, 14), random.randint(8, 14)
            x1 = int(e["x"] - w / 2)
            y1 = int(e["y"] - h / 2)
            x2 = x1 + w
            y2 = y1 + h
            conf = random.uniform(0.55, 0.98)
            if conf >= conf_threshold:
                detections.append((x1, y1, x2, y2, conf))

        return detections


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE TRACKER (ByteTrack-inspired IoU-based tracker)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleTracker:
    """
    Lightweight ByteTrack-inspired tracker using IoU matching.
    Replace with full ByteTrack/DeepSORT for production.
    """
    def __init__(self, iou_threshold: float = 0.2, max_age: int = 20):
        self.next_id = 1
        self.active_tracks: Dict[int, dict] = {}
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    def _center(self, bbox):
        return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

    def update(self, detections, frame_num):
        """Returns list of (track_id, x1,y1,x2,y2, confidence)"""
        if not self.active_tracks:
            results = []
            for det in detections:
                tid = self.next_id; self.next_id += 1
                cx, cy = self._center(det)
                self.active_tracks[tid] = {
                    "bbox": det[:4], "age": 0, "cx": cx, "cy": cy,
                    "frame": frame_num
                }
                results.append((tid, *det[:4], det[4]))
            return results

        matched = {}
        unmatched_dets = list(range(len(detections)))

        for tid, tr in self.active_tracks.items():
            best_iou = self.iou_threshold
            best_det_idx = -1
            for di in unmatched_dets:
                iou = self._iou(tr["bbox"], detections[di][:4])
                # Also use distance for cases where boxes don't overlap
                cx, cy = self._center(detections[di][:4])
                dist = math.hypot(cx - tr["cx"], cy - tr["cy"])
                combined = iou + max(0, 1 - dist / 30) * 0.3
                if combined > best_iou:
                    best_iou = combined; best_det_idx = di

            if best_det_idx >= 0:
                matched[tid] = best_det_idx
                unmatched_dets.remove(best_det_idx)

        results = []
        # Update matched tracks
        for tid, di in matched.items():
            det = detections[di]
            cx, cy = self._center(det[:4])
            self.active_tracks[tid].update({"bbox": det[:4], "age": 0, "cx": cx, "cy": cy, "frame": frame_num})
            results.append((tid, *det[:4], det[4]))

        # Age unmatched tracks
        to_remove = []
        for tid in self.active_tracks:
            if tid not in matched:
                self.active_tracks[tid]["age"] += 1
                if self.active_tracks[tid]["age"] > self.max_age:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.active_tracks[tid]

        # Register new detections
        for di in unmatched_dets:
            det = detections[di]
            tid = self.next_id; self.next_id += 1
            cx, cy = self._center(det[:4])
            self.active_tracks[tid] = {"bbox": det[:4], "age": 0, "cx": cx, "cy": cy, "frame": frame_num}
            results.append((tid, *det[:4], det[4]))

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# MOTION ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class MotionAnalyzer:
    def __init__(self, px_per_um: float, fps: float = 30.0,
                 min_frames_classify: int = 15):
        self.px_per_um = px_per_um          # pixels per micron
        self.fps = fps
        self.min_frames = min_frames_classify
        self.dt = 1.0 / fps                 # seconds per frame

        # Thresholds (µm/s and µm)
        self.progressive_speed_thr = 20.0   # µm/s
        self.progressive_disp_thr  = 15.0   # µm net displacement
        self.immotile_speed_thr    = 5.0    # µm/s
        self.immotile_disp_thr     = 2.0    # µm

    def px_to_um(self, px: float) -> float:
        return px / self.px_per_um

    def compute_speed(self, p1, p2) -> float:
        """Speed in µm/s between two positions (pixels)"""
        dist_px = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        dist_um = self.px_to_um(dist_px)
        return dist_um / self.dt

    def compute_casa_metrics(self, track: SpermTrack) -> dict:
        pos = list(track.positions)
        if len(pos) < 3:
            return {}

        times = list(track.timestamps)
        n = len(pos)

        # VCL – sum of frame-to-frame distances
        total_path_um = sum(
            self.px_to_um(math.hypot(pos[i][0]-pos[i-1][0], pos[i][1]-pos[i-1][1]))
            for i in range(1, n)
        )
        duration = times[-1] - times[0] if times[-1] > times[0] else self.dt
        vcl = total_path_um / duration if duration > 0 else 0

        # VSL – straight-line speed start→end
        sl_dist_um = self.px_to_um(math.hypot(pos[-1][0]-pos[0][0], pos[-1][1]-pos[0][1]))
        vsl = sl_dist_um / duration if duration > 0 else 0

        # VAP – smoothed trajectory (every 3rd point)
        smoothed = pos[::3]
        smoothed_dist = sum(
            self.px_to_um(math.hypot(smoothed[i][0]-smoothed[i-1][0], smoothed[i][1]-smoothed[i-1][1]))
            for i in range(1, len(smoothed))
        )
        vap = smoothed_dist / duration if duration > 0 else 0

        # LIN STR WOB
        lin = (vsl / vcl * 100) if vcl > 0 else 0
        str_ = (vsl / vap * 100) if vap > 0 else 0
        wob = (vap / vcl * 100) if vcl > 0 else 0

        # ALH – lateral deviation from mean path
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        mean_x = np.mean(xs); mean_y = np.mean(ys)
        if n > 1:
            dx = pos[-1][0] - pos[0][0]; dy = pos[-1][1] - pos[0][1]
            path_len = math.hypot(dx, dy)
            if path_len > 0:
                laterals = [abs((p[0]-pos[0][0])*dy - (p[1]-pos[0][1])*dx) / path_len for p in pos]
            else:
                laterals = [math.hypot(p[0]-mean_x, p[1]-mean_y) for p in pos]
            alh = self.px_to_um(np.mean(laterals))
        else:
            alh = 0

        # BCF – sign changes in lateral displacement (oscillation frequency)
        if n > 4:
            diffs = np.diff([p[0] for p in pos])
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            bcf = sign_changes / (2 * duration) if duration > 0 else 0
        else:
            bcf = 0

        # Net displacement for classification
        net_disp_um = sl_dist_um

        return {
            "vcl": vcl, "vsl": vsl, "vap": vap,
            "lin": lin, "str": str_, "wob": wob,
            "alh": alh, "bcf": bcf,
            "net_disp_um": net_disp_um,
            "total_path_um": total_path_um,
            "duration": duration,
        }

    def classify_motility(self, track: SpermTrack) -> Tuple[str, Tuple]:
        """Multi-frame classification with Jitter-Proof Immotile Detection"""
        pos = list(track.positions)
        
        # Lowering min frames slightly ensures we catch dead sperm even if 
        # YOLO drops confidence and loses track of them momentarily.
        if len(pos) < max(5, self.min_frames - 5):
            return "Unknown", (128, 128, 128)

        # 1. JITTER-PROOF IMMOTILE CHECK
        # Find the center point of all the sperm's recorded positions
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        mean_x, mean_y = sum(xs)/len(xs), sum(ys)/len(ys)
        
        # Calculate the maximum distance it has ever drifted from that center point
        max_drift_px = max(math.hypot(p[0] - mean_x, p[1] - mean_y) for p in pos)
        max_drift_um = self.px_to_um(max_drift_px)

        # If it never leaves a tiny 8µm radius, it's just YOLO jitter. It is DEAD.
        if max_drift_um < 8.0:
            return "Immotile", (220, 50, 50)

        # 2. ALIVE: PROGRESSIVE VS NON-PROGRESSIVE CHECK
        casa = self.compute_casa_metrics(track)
        speeds = list(track.speeds)
        avg_speed = np.mean(speeds) if speeds else 0
        net_disp = casa.get("net_disp_um", 0)

        if avg_speed >= self.progressive_speed_thr and net_disp >= self.progressive_disp_thr:
            return "Progressive", (50, 220, 80)
        else:
            return "Non-Progressive", (220, 190, 40)

# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class VideoProcessor:
    def __init__(self, calibration: dict):
        self.cal = calibration
        self.px_per_um = 1.0 / calibration["um_per_px"]
        self.fps = calibration.get("fps", 30.0)
        # Make sure to put the exact path to your best.pt file here
        self.detector = RealYOLODetector("./best_15-06_300.pt")
        # self.detector = MockYOLODetector(num_sperm=int(calibration.get("num_sperm", 40)))
        self.tracker = SimpleTracker(iou_threshold=0.15, max_age=25)
        self.analyzer = MotionAnalyzer(
            px_per_um=self.px_per_um,
            fps=self.fps
        )
        self.tracks: Dict[int, SpermTrack] = {}
        self.frame_data = []          # per-frame aggregate stats
        self.frame_num = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        self.frame_num += 1
        ts = self.frame_num / self.fps

        # 1. Detect
        detections = self.detector.detect(frame, self.frame_num, conf_threshold=0.4)

        # 2. Track
        tracked = self.tracker.update(detections, self.frame_num)

        # 3. Update track histories
        for item in tracked:
            tid, x1, y1, x2, y2, conf = item
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if tid not in self.tracks:
                self.tracks[tid] = SpermTrack(track_id=tid)

            trk = self.tracks[tid]
            if trk.positions:
                prev = trk.positions[-1]
                spd = self.analyzer.compute_speed(prev, (cx, cy))
            else:
                spd = 0.0

            trk.positions.append((cx, cy))
            trk.timestamps.append(ts)
            trk.speeds.append(spd)
            trk.confidences.append(conf)
            trk.bbox_history.append((x1, y1, x2, y2))
            trk.frames_seen += 1
            trk.last_seen_frame = self.frame_num

            # Classify after enough frames
            motility, color = self.analyzer.classify_motility(trk)
            trk.motility_class = motility
            trk.color = color

            # Update CASA metrics
            casa = self.analyzer.compute_casa_metrics(trk)
            if casa:
                trk.vcl = casa["vcl"]
                trk.vsl = casa["vsl"]
                trk.vap = casa["vap"]
                trk.lin = casa["lin"]
                trk.str_metric = casa["str"]
                trk.wob = casa["wob"]
                trk.alh = casa["alh"]
                trk.bcf = casa["bcf"]

        # 4. Draw overlays
        annotated = self._draw_overlays(frame.copy(), tracked)

        # 5. Compute frame stats
        stats = self._frame_stats(ts)
        self.frame_data.append(stats)

        return annotated, stats

    def _draw_overlays(self, frame, tracked_list):
        for item in tracked_list:
            tid, x1, y1, x2, y2, conf = item
            trk = self.tracks.get(tid)
            if trk is None:
                continue

            color = trk.color
            spd = trk.speeds[-1] if trk.speeds else 0

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

            # Draw path
            pos = list(trk.positions)
            if len(pos) > 2:
                pts = np.array([(int(p[0]), int(p[1])) for p in pos[-40:]], dtype=np.int32)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    c = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, pts[i-1], pts[i], c, 1)

            # Label
            label_y = int(y1) - 4
            cv2.putText(frame, f"ID:{tid}", (int(x1), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{spd:.0f}µm/s", (int(x1), label_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

            status_abbrev = {"Progressive":"P", "Non-Progressive":"NP", "Immotile":"IM"}.get(trk.motility_class, "?")
            cv2.putText(frame, status_abbrev, (int(x2)+1, int(y1)+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_num}", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 180, 255), 1)
        return frame

    def _frame_stats(self, ts) -> dict:
        active = {tid: trk for tid, trk in self.tracks.items()
                  if self.frame_num - trk.last_seen_frame <= 5}

        progressive  = [t for t in active.values() if t.motility_class == "Progressive"]
        nonprog      = [t for t in active.values() if t.motility_class == "Non-Progressive"]
        immotile     = [t for t in active.values() if t.motility_class == "Immotile"]
        alive        = progressive + nonprog
        classified   = progressive + nonprog + immotile

        all_speeds = [s for t in active.values() for s in list(t.speeds)[-5:] if s > 0]

        return {
            "timestamp": ts,
            "frame": self.frame_num,
            "total": len(active),
            "progressive": len(progressive),
            "nonprogressive": len(nonprog),
            "immotile": len(immotile),
            "alive": len(alive),
            "dead": len(immotile),
            "avg_speed": np.mean(all_speeds) if all_speeds else 0,
            "max_speed": np.max(all_speeds) if all_speeds else 0,
            "motility_pct": 100*len(alive)/len(classified) if classified else 0,
        }

    def get_global_summary(self) -> dict:
        """Compute final aggregate metrics"""
        all_tracks = list(self.tracks.values())
        classified = [t for t in all_tracks if t.motility_class != "Unknown" and t.frames_seen >= 5]

        progressive  = [t for t in classified if t.motility_class == "Progressive"]
        nonprog      = [t for t in classified if t.motility_class == "Non-Progressive"]
        immotile     = [t for t in classified if t.motility_class == "Immotile"]
        alive        = progressive + nonprog

        all_speeds = [s for t in classified for s in list(t.speeds) if s > 0]

        total = len(classified)
        alive_count = len(alive)
        dead_count = len(immotile)
        motility_pct = 100*alive_count/total if total > 0 else 0

        vol_ml = self.cal["sample_volume_ul"] / 1000.0
        concentration = total / vol_ml / 1e6  # million/mL
        tmsc = alive_count / vol_ml / 1e6

        # Average CASA metrics over all classified tracks
        def avg_metric(tracks, attr):
            vals = [getattr(t, attr) for t in tracks if getattr(t, attr, 0) > 0]
            return float(np.mean(vals)) if vals else 0.0

        return {
            # Count
            "total_count": total,
            "alive_count": alive_count,
            "dead_count": dead_count,
            "progressive_count": len(progressive),
            "nonprogressive_count": len(nonprog),
            "immotile_count": len(immotile),
            # Percentages
            "alive_pct": 100*alive_count/total if total else 0,
            "dead_pct": 100*dead_count/total if total else 0,
            "motility_pct": motility_pct,
            "progressive_pct": 100*len(progressive)/total if total else 0,
            # Speed
            "avg_speed": float(np.mean(all_speeds)) if all_speeds else 0,
            "max_speed": float(np.max(all_speeds)) if all_speeds else 0,
            "min_speed": float(np.min(all_speeds)) if all_speeds else 0,
            "std_speed": float(np.std(all_speeds)) if all_speeds else 0,
            # Concentration
            "concentration_mpm": round(concentration, 2),
            "sperm_per_ml": int(total / vol_ml),
            "tmsc_mpm": round(tmsc, 2),
            # CASA
            "vcl": avg_metric(classified, "vcl"),
            "vsl": avg_metric(classified, "vsl"),
            "vap": avg_metric(classified, "vap"),
            "lin": avg_metric(classified, "lin"),
            "str": avg_metric(classified, "str_metric"),
            "wob": avg_metric(classified, "wob"),
            "alh": avg_metric(classified, "alh"),
            "bcf": avg_metric(classified, "bcf"),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

CHART_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,20,40,0.6)",
    font=dict(color="#8097b8", family="Inter", size=11),
    margin=dict(l=30, r=15, t=35, b=30),
)

def motility_pie(stats: dict) -> go.Figure:
    labels = ["Progressive", "Non-Progressive", "Immotile"]
    vals   = [stats.get("progressive",0), stats.get("nonprogressive",0), stats.get("immotile",0)]
    colors = ["#34d399", "#fbbf24", "#f87171"]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals,
        marker_colors=colors,
        hole=0.52,
        textinfo="percent",
        textfont_size=12,
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **CHART_TEMPLATE,
        showlegend=True,
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                    font=dict(size=10, color="#8097b8")),
        height=230,
        title=dict(text="Motility Distribution", font=dict(size=12, color="#4a90d9"), x=0.5),
    )
    return fig


def speed_histogram(frame_data: list) -> go.Figure:
    speeds = [d["avg_speed"] for d in frame_data if d["avg_speed"] > 0]
    fig = go.Figure(go.Histogram(
        x=speeds, nbinsx=20,
        marker=dict(color="#4a90d9", opacity=0.8, line=dict(color="#1a3a6a", width=0.5)),
        hovertemplate="Speed: %{x:.1f} µm/s<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_TEMPLATE,
        xaxis=dict(title="Speed (µm/s)", color="#5b7fa6", gridcolor="#1a2840"),
        yaxis=dict(title="Count", color="#5b7fa6", gridcolor="#1a2840"),
        height=220,
        title=dict(text="Speed Distribution", font=dict(size=12, color="#4a90d9"), x=0.5),
    )
    return fig


def speed_trend(frame_data: list) -> go.Figure:
    ts = [d["timestamp"] for d in frame_data]
    avg_spd = [d["avg_speed"] for d in frame_data]
    max_spd = [d["max_speed"] for d in frame_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=avg_spd, name="Avg Speed",
        line=dict(color="#4a90d9", width=1.5),
        fill="tozeroy", fillcolor="rgba(74,144,217,0.07)"))
    fig.add_trace(go.Scatter(x=ts, y=max_spd, name="Max Speed",
        line=dict(color="#a78bfa", width=1, dash="dot")))
    fig.update_layout(
        **CHART_TEMPLATE,
        xaxis=dict(title="Time (s)", color="#5b7fa6", gridcolor="#1a2840"),
        yaxis=dict(title="Speed (µm/s)", color="#5b7fa6", gridcolor="#1a2840"),
        height=220,
        legend=dict(font=dict(size=10)),
        title=dict(text="Speed Trend", font=dict(size=12, color="#4a90d9"), x=0.5),
    )
    return fig


def count_trend(frame_data: list) -> go.Figure:
    ts     = [d["timestamp"] for d in frame_data]
    total  = [d["total"] for d in frame_data]
    alive  = [d["alive"] for d in frame_data]
    dead   = [d["dead"] for d in frame_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=total, name="Total", line=dict(color="#60a5fa", width=1.5)))
    fig.add_trace(go.Scatter(x=ts, y=alive, name="Alive", line=dict(color="#34d399", width=1.2)))
    fig.add_trace(go.Scatter(x=ts, y=dead, name="Dead", line=dict(color="#f87171", width=1.2)))
    fig.update_layout(
        **CHART_TEMPLATE,
        xaxis=dict(title="Time (s)", color="#5b7fa6", gridcolor="#1a2840"),
        yaxis=dict(title="Count", color="#5b7fa6", gridcolor="#1a2840"),
        height=220,
        legend=dict(font=dict(size=10)),
        title=dict(text="Count Trend", font=dict(size=12, color="#4a90d9"), x=0.5),
    )
    return fig


def casa_radar(summary: dict) -> go.Figure:
    cats = ["LIN", "STR", "WOB", "BCF×10", "ALH"]
    vals = [
        min(summary.get("lin", 0), 100),
        min(summary.get("str", 0), 100),
        min(summary.get("wob", 0), 100),
        min(summary.get("bcf", 0) * 10, 100),
        min(summary.get("alh", 0) * 5, 100),
    ]
    vals += [vals[0]]
    cats += [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats,
        fill="toself", fillcolor="rgba(74,144,217,0.15)",
        line=dict(color="#4a90d9", width=2),
    ))
    fig.update_layout(
        **CHART_TEMPLATE,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#5b7fa6", gridcolor="#1a2840"),
            angularaxis=dict(color="#5b7fa6"),
            bgcolor="rgba(10,20,40,0.6)",
        ),
        height=270,
        title=dict(text="CASA Metrics Radar", font=dict(size=12, color="#4a90d9"), x=0.5),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_assessment(summary: dict, cal: dict) -> str:
    motility = summary.get("motility_pct", 0)
    conc     = summary.get("concentration_mpm", 0)
    prog_pct = summary.get("progressive_pct", 0)
    avg_spd  = summary.get("avg_speed", 0)
    dead_pct = summary.get("dead_pct", 0)

    # 1. Base Quality Grade
    if motility >= 70 and conc >= 20 and prog_pct >= 50:
        grade_text = "The semen sample demonstrates excellent quality with high motility and progressive sperm percentage."
    elif motility >= 50 and conc >= 10:
        grade_text = "The semen sample demonstrates good motility with an acceptable proportion of progressive sperm."
    elif motility >= 30:
        grade_text = "The semen sample shows moderate motility. Further evaluation is recommended."
    else:
        grade_text = "The semen sample shows low motility. Fertility potential may be significantly reduced."

    # 2. Speed Analysis
    if avg_spd > 0:
        spd_text = f"Average swimming velocity of {avg_spd:.1f} µm/s {'falls within the expected range' if 20 <= avg_spd <= 120 else 'is outside the typical range'} for healthy bovine semen."
    else:
        spd_text = "No meaningful swimming velocity could be calculated."

    # 3. Concentration Analysis (Fix applied here)
    if conc >= 15:
        conc_text = f"The concentration estimate of {conc:.2f} million sperm/mL indicates a sufficiently dense sample suitable for standard commercial dilution."
    elif conc > 0:
        conc_text = f"The concentration estimate of {conc:.2f} million sperm/mL indicates a low-density sample, which may be unsuitable for standard AI dilution protocols."
    else:
        conc_text = "The concentration estimate of 0.00 million sperm/mL indicates an absence of detected sperm (azoospermia) or a potential error in calibration settings."

    # 4. Viability Analysis
    if dead_pct < 30:
        dead_text = f"Dead sperm percentage of {dead_pct:.1f}% remains relatively low, suggesting acceptable viability."
    else:
        dead_text = f"Dead sperm percentage of {dead_pct:.1f}% is elevated, indicating reduced viability."

    # Combine into a final paragraph
    assessment = f"{grade_text} {spd_text} {conc_text} {dead_text}"
    
    return assessment

# def generate_assessment(summary: dict, cal: dict) -> str:
#     motility = summary.get("motility_pct", 0)
#     conc     = summary.get("concentration_mpm", 0)
#     prog_pct = summary.get("progressive_pct", 0)
#     avg_spd  = summary.get("avg_speed", 0)
#     dead_pct = summary.get("dead_pct", 0)

#     if motility >= 70 and conc >= 20 and prog_pct >= 50:
#         quality = "excellent"
#         grade_text = "The semen sample demonstrates excellent quality with high motility and progressive sperm percentage."
#     elif motility >= 50 and conc >= 10:
#         quality = "good"
#         grade_text = "The semen sample demonstrates good motility with an acceptable proportion of progressive sperm."
#     elif motility >= 30:
#         quality = "moderate"
#         grade_text = "The semen sample shows moderate motility. Further evaluation is recommended."
#     else:
#         quality = "poor"
#         grade_text = "The semen sample shows low motility. Fertility potential may be significantly reduced."

#     assessment = (
#         f"{grade_text} Average swimming velocity of {avg_spd:.1f} µm/s "
#         f"{'falls within the expected range' if 20 <= avg_spd <= 120 else 'is outside the typical range'} "
#         f"for healthy bovine semen. The concentration estimate of {conc:.2f} million sperm/mL "
#         f"{'indicates a sufficiently dense sample' if conc >= 15 else 'indicates a low-density sample'} "
#         f"suitable for {'further fertility evaluation' if conc >= 15 else 'dilution analysis'}. "
#         f"Dead sperm percentage of {dead_pct:.1f}% "
#         f"{'remains relatively low, suggesting acceptable viability' if dead_pct < 30 else 'is elevated, indicating reduced viability'}."
#     )
#     return assessment


def to_csv(summary: dict, cal: dict, frame_data: list) -> bytes:
    rows = []
    rows.append(["=== CALIBRATION ===", ""])
    rows += [["Magnification", cal.get("magnification","")],
             ["Resolution", f"{cal.get('res_w',0)}x{cal.get('res_h',0)}"],
             ["Pixel-to-Micron", f"{cal.get('um_per_px',0)} µm/px"],
             ["Sample Volume", f"{cal.get('sample_volume_ul',50)} µL"]]
    rows.append(["", ""])
    rows.append(["=== ANALYSIS METRICS ===", ""])
    for k, v in summary.items():
        rows.append([k, round(v, 4) if isinstance(v, float) else v])
    rows.append(["", ""])
    rows.append(["=== FRAME DATA ===", ""])
    df_frames = pd.DataFrame(frame_data)
    df_main = pd.DataFrame(rows, columns=["Parameter", "Value"])

    buf = io.BytesIO()
    df_main.to_csv(buf, index=False)
    buf.write(b"\n")
    df_frames.to_csv(buf, index=False)
    return buf.getvalue()


def to_excel(summary: dict, cal: dict, frame_data: list) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Calibration
        pd.DataFrame([
            ["Magnification", cal.get("magnification","")],
            ["Camera Resolution", f"{cal.get('res_w',0)}x{cal.get('res_h',0)}"],
            ["µm per Pixel", cal.get("um_per_px",0)],
            ["Sample Volume (µL)", cal.get("sample_volume_ul",50)],
        ], columns=["Parameter","Value"]).to_excel(writer, sheet_name="Calibration", index=False)

        # Summary
        pd.DataFrame(
            [[k, round(v,4) if isinstance(v,float) else v] for k,v in summary.items()],
            columns=["Metric","Value"]
        ).to_excel(writer, sheet_name="Summary", index=False)

        # Frame data
        pd.DataFrame(frame_data).to_excel(writer, sheet_name="Frame Data", index=False)

    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "running": False,
        "processor": None,
        "frame_data": [],
        "current_stats": {},
        "summary": None,
        "calibration": {},
        "start_time": None,
        "frames_processed": 0,
        "analysis_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=120)  # Caches results for 2 minutes to prevent UI lag
def get_available_cameras(max_to_test=5):
    """Scans for available camera indices on the system."""
    available_cameras = []
    for i in range(max_to_test):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras
# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🔬 CowSperm CASA")
    st.markdown('<div class="section-header">Calibration Settings</div>', unsafe_allow_html=True)

    magnification = st.selectbox("Magnification", ["10x","20x","40x","100x"], index=2)

    res_options = {"1920×1080":"1920x1080","1280×720":"1280x720","640×480":"640x480"}
    res_label = st.selectbox("Camera Resolution", list(res_options.keys()), index=2)
    res_str = res_options[res_label]
    res_w, res_h = int(res_str.split("x")[0]), int(res_str.split("x")[1])

    um_per_px = st.selectbox(
        "Pixel-to-Micron Ratio",
        [0.125, 0.25, 0.5, 1.0],
        index=1,
        format_func=lambda v: f"1 px = {v} µm"
    )

    sample_vol = st.number_input("Sample Volume (µL)", min_value=1.0, max_value=500.0,
                                  value=50.0, step=1.0)

    st.divider()
    st.markdown('<div class="section-header">Detection Settings</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.99, 0.25, 0.05)
    analysis_duration = st.slider("Analysis Duration (s)", 5, 60, 15)


    st.divider()
    st.markdown('<div class="section-header">Input Source</div>', unsafe_allow_html=True)
    source = st.radio("Source", ["📷 Live Camera", "📁 Upload Video"], index=0)

    camera_id = 0
    uploaded_file = None

    if source == "📷 Live Camera":
        # Scan for active hardware
        available_cams = get_available_cameras()
        
        if not available_cams:
            st.error("No cameras detected! Please check your USB or hardware connections.")
            camera_id = None
        else:
            # Create a user-friendly dropdown
            cam_options = {f"Camera Device {i}": i for i in available_cams}
            selected_cam_label = st.selectbox("Select Available Camera", list(cam_options.keys()))
            camera_id = cam_options[selected_cam_label]
    else:
        uploaded_file = st.file_uploader("Upload microscope video", type=["mp4","avi","mov","wmv", "asf"])

    # st.divider()
    # st.markdown('<div class="section-header">Input Source</div>', unsafe_allow_html=True)
    # source = st.radio("Source", ["📷 Live Camera", "📁 Upload Video"], index=0)

    # camera_id = 0
    # uploaded_file = None

    # if source == "📷 Live Camera":
    #     camera_id = st.number_input("Camera ID (0 for default, 1 for external USB, etc.)", min_value=0, max_value=10, value=0, step=1)
    # else:
    #     uploaded_file = st.file_uploader("Upload microscope video", type=["mp4","avi","mov","wmv", "asf"])

    # st.divider()
    # st.markdown('<div class="section-header">Input Source</div>', unsafe_allow_html=True)
    # source = st.radio("Source", ["🎞️ Simulate Live Feed", "📁 Upload Video"], index=0)

    # uploaded_file = None
    # if "Upload Video" in source:
    #     uploaded_file = st.file_uploader("Upload microscope video", type=["mp4","avi","mov","wmv", "asf"])

    calibration = {
        "magnification": magnification,
        "res_w": res_w, "res_h": res_h,
        "um_per_px": um_per_px,
        "sample_volume_ul": sample_vol,
        "conf_threshold": conf_threshold,
        "fps": 30.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='color:#e8f0fe;font-size:24px;margin-bottom:2px;'>🔬 Advanced Cow Sperm CASA Analyzer</h1>"
    "<p style='color:#5b7fa6;font-size:13px;margin-top:0;'>Computer Assisted Semen Analysis Platform — Real-Time Motility & CASA Metrics</p>",
    unsafe_allow_html=True
)

# Control buttons
col_start, col_stop, col_reset, col_spacer = st.columns([1,1,1,5])
with col_start:
    start_btn = st.button("▶ Start Analysis", type="primary",
                           disabled=st.session_state.running)
with col_stop:
    stop_btn = st.button("⏹ Stop", disabled=not st.session_state.running)
with col_reset:
    reset_btn = st.button("↺ Reset")

if start_btn:
    st.session_state.running = True
    st.session_state.analysis_done = False
    st.session_state.frame_data = []
    st.session_state.current_stats = {}
    st.session_state.summary = None
    st.session_state.calibration = calibration
    st.session_state.processor = VideoProcessor(calibration)
    st.session_state.start_time = time.time()
    st.session_state.frames_processed = 0

if stop_btn:
    st.session_state.running = False
    if st.session_state.processor:
        st.session_state.summary = st.session_state.processor.get_global_summary()
        st.session_state.analysis_done = True

if reset_btn:
    for k in ["running","processor","frame_data","current_stats","summary","analysis_done","frames_processed"]:
        st.session_state[k] = False if k == "running" else ([] if k == "frame_data" else
                               {} if k in ["current_stats","calibration"] else None if k in ["processor","summary"] else 0 if k == "frames_processed" else False)
    st.rerun()

st.divider()

# KPI ROW
kpi_ph = st.empty()
st.divider()

# Main content tabs
tab_live, tab_charts, tab_casa, tab_summary = st.tabs(
    ["📹 Live Feed", "📊 Analytics", "🧬 CASA Metrics", "📋 Summary & Export"]
)

with tab_live:
    col_vid, col_stats = st.columns([3, 2])
    with col_vid:
        st.markdown('<div class="section-header">Live Video Feed with Overlays</div>', unsafe_allow_html=True)
        video_ph = st.empty()
        progress_ph = st.empty()
    with col_stats:
        st.markdown('<div class="section-header">Real-Time Detection</div>', unsafe_allow_html=True)
        pie_ph = st.empty()
        rt_stats_ph = st.empty()

with tab_charts:
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        hist_ph = st.empty()
    with chart_col2:
        spd_trend_ph = st.empty()
    cnt_trend_ph = st.empty()

with tab_casa:
    casa_col1, casa_col2 = st.columns([2, 3])
    with casa_col1:
        radar_ph = st.empty()
    with casa_col2:
        casa_table_ph = st.empty()

with tab_summary:
    summary_ph = st.empty()
    export_ph = st.empty()


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def render_kpis(stats: dict):
    s = stats
    kpi_ph.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;">
        <div class="kpi-card">
            <div class="kpi-label">Total Detected</div>
            <div class="kpi-value">{s.get('total',0)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Alive</div>
            <div class="kpi-value" style="color:#34d399">{s.get('alive',0)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Dead</div>
            <div class="kpi-value" style="color:#f87171">{s.get('dead',0)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Motility</div>
            <div class="kpi-value" style="color:#60a5fa">{s.get('motility_pct',0):.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Speed</div>
            <div class="kpi-value">{s.get('avg_speed',0):.1f}</div>
            <div class="kpi-sub">µm/s</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Max Speed</div>
            <div class="kpi-value" style="color:#a78bfa">{s.get('max_speed',0):.1f}</div>
            <div class="kpi-sub">µm/s</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_rt_stats(stats: dict):
    rt_stats_ph.markdown(f"""
    <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:10px;padding:14px;">
        <div class="metric-row"><span class="metric-key">Progressive</span><span class="metric-val" style="color:#34d399">{stats.get('progressive',0)}</span></div>
        <div class="metric-row"><span class="metric-key">Non-Progressive</span><span class="metric-val" style="color:#fbbf24">{stats.get('nonprogressive',0)}</span></div>
        <div class="metric-row"><span class="metric-key">Immotile</span><span class="metric-val" style="color:#f87171">{stats.get('immotile',0)}</span></div>
        <div class="metric-row"><span class="metric-key">Frame</span><span class="metric-val">{stats.get('frame',0)}</span></div>
        <div class="metric-row"><span class="metric-key">Time</span><span class="metric-val">{stats.get('timestamp',0):.1f}s</span></div>
    </div>
    """, unsafe_allow_html=True)

def render_summary(summary: dict, cal: dict):
    assessment = generate_assessment(summary, cal)
    
    summary_html = f"""
    <div class="summary-panel">
        <h3 style="color:#e8f0fe;margin-top:0;">📋 Analysis Summary</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
            <div>
                <div class="section-header">Calibration</div>
                <div class="metric-row"><span class="metric-key">Sample Volume</span><span class="metric-val">{cal.get('sample_volume_ul',50):.0f} µL</span></div>
                <div class="metric-row"><span class="metric-key">Magnification</span><span class="metric-val">{cal.get('magnification','')}</span></div>
                <div class="metric-row"><span class="metric-key">Pixel-to-Micron</span><span class="metric-val">{cal.get('um_per_px',0)} µm/px</span></div>
                <div class="metric-row"><span class="metric-key">Resolution</span><span class="metric-val">{cal.get('res_w',0)}×{cal.get('res_h',0)}</span></div>
            </div>
            <div>
                <div class="section-header">Count Metrics</div>
                <div class="metric-row"><span class="metric-key">Total Sperm Count</span><span class="metric-val">{summary.get('total_count',0):,}</span></div>
                <div class="metric-row"><span class="metric-key">Alive Sperm</span><span class="metric-val" style="color:#34d399">{summary.get('alive_count',0):,}</span></div>
                <div class="metric-row"><span class="metric-key">Dead Sperm</span><span class="metric-val" style="color:#f87171">{summary.get('dead_count',0):,}</span></div>
                <div class="metric-row"><span class="metric-key">Progressive</span><span class="metric-val">{summary.get('progressive_count',0):,}</span></div>
                <div class="metric-row"><span class="metric-key">Non-Progressive</span><span class="metric-val">{summary.get('nonprogressive_count',0):,}</span></div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
            <div>
                <div class="section-header">Percentage Metrics</div>
                <div class="metric-row"><span class="metric-key">Alive %</span><span class="metric-val">{summary.get('alive_pct',0):.1f}%</span></div>
                <div class="metric-row"><span class="metric-key">Dead %</span><span class="metric-val" style="color:#f87171">{summary.get('dead_pct',0):.1f}%</span></div>
                <div class="metric-row"><span class="metric-key">Motility %</span><span class="metric-val" style="color:#60a5fa">{summary.get('motility_pct',0):.1f}%</span></div>
                <div class="metric-row"><span class="metric-key">Progressive Motility %</span><span class="metric-val" style="color:#34d399">{summary.get('progressive_pct',0):.1f}%</span></div>
            </div>
            <div>
                <div class="section-header">Speed Metrics</div>
                <div class="metric-row"><span class="metric-key">Average Speed</span><span class="metric-val">{summary.get('avg_speed',0):.1f} µm/s</span></div>
                <div class="metric-row"><span class="metric-key">Maximum Speed</span><span class="metric-val">{summary.get('max_speed',0):.1f} µm/s</span></div>
                <div class="metric-row"><span class="metric-key">Minimum Speed</span><span class="metric-val">{summary.get('min_speed',0):.1f} µm/s</span></div>
                <div class="metric-row"><span class="metric-key">Speed Std Dev</span><span class="metric-val">{summary.get('std_speed',0):.1f} µm/s</span></div>
            </div>
        </div>
        <div>
            <div class="section-header">Concentration Metrics</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">
                <div class="kpi-card">
                    <div class="kpi-label">Concentration</div>
                    <div class="kpi-value" style="font-size:22px;">{summary.get('concentration_mpm',0):.4f}M</div>
                    <div class="kpi-sub">sperm/mL</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Total Sperm/mL</div>
                    <div class="kpi-value" style="font-size:22px;">{summary.get('sperm_per_ml',0):,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">TMSC</div>
                    <div class="kpi-value" style="font-size:22px;">{summary.get('tmsc_mpm',0):.4f}M</div>
                    <div class="kpi-sub">motile/mL</div>
                </div>
            </div>
        </div>
        <div style="margin-top:20px;">
            <div class="section-header">Overall Assessment</div>
            <div class="assessment-block">{assessment}</div>
        </div>
        <div style="margin-top:16px;color:#5b7fa6;font-size:11px;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            YOLO Detection + ByteTrack | Calibration: {cal.get('magnification','')} @ {cal.get('um_per_px',0)} µm/px
        </div>
    </div>
    """
    
    summary_ph.markdown(summary_html, unsafe_allow_html=True)

# def render_summary(summary: dict, cal: dict):
#     assessment = generate_assessment(summary, cal)
#     summary_ph.markdown(f"""
# <div class="summary-panel">
#     <h3 style="color:#e8f0fe;margin-top:0;">📋 Analysis Summary</h3>

#     <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
#         <div>
#             <div class="section-header">Calibration</div>
#             <div class="metric-row"><span class="metric-key">Sample Volume</span><span class="metric-val">{cal.get('sample_volume_ul',50):.0f} µL</span></div>
#             <div class="metric-row"><span class="metric-key">Magnification</span><span class="metric-val">{cal.get('magnification','')}</span></div>
#             <div class="metric-row"><span class="metric-key">Pixel-to-Micron</span><span class="metric-val">{cal.get('um_per_px',0)} µm/px</span></div>
#             <div class="metric-row"><span class="metric-key">Resolution</span><span class="metric-val">{cal.get('res_w',0)}×{cal.get('res_h',0)}</span></div>
#         </div>
#         <div>
#             <div class="section-header">Count Metrics</div>
#             <div class="metric-row"><span class="metric-key">Total Sperm Count</span><span class="metric-val">{summary.get('total_count',0):,}</span></div>
#             <div class="metric-row"><span class="metric-key">Alive Sperm</span><span class="metric-val" style="color:#34d399">{summary.get('alive_count',0):,}</span></div>
#             <div class="metric-row"><span class="metric-key">Dead Sperm</span><span class="metric-val" style="color:#f87171">{summary.get('dead_count',0):,}</span></div>
#             <div class="metric-row"><span class="metric-key">Progressive</span><span class="metric-val">{summary.get('progressive_count',0):,}</span></div>
#             <div class="metric-row"><span class="metric-key">Non-Progressive</span><span class="metric-val">{summary.get('nonprogressive_count',0):,}</span></div>
#         </div>
#     </div>

#     <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
#         <div>
#             <div class="section-header">Percentage Metrics</div>
#             <div class="metric-row"><span class="metric-key">Alive %</span><span class="metric-val">{summary.get('alive_pct',0):.1f}%</span></div>
#             <div class="metric-row"><span class="metric-key">Dead %</span><span class="metric-val" style="color:#f87171">{summary.get('dead_pct',0):.1f}%</span></div>
#             <div class="metric-row"><span class="metric-key">Motility %</span><span class="metric-val" style="color:#60a5fa">{summary.get('motility_pct',0):.1f}%</span></div>
#             <div class="metric-row"><span class="metric-key">Progressive Motility %</span><span class="metric-val" style="color:#34d399">{summary.get('progressive_pct',0):.1f}%</span></div>
#         </div>
#         <div>
#             <div class="section-header">Speed Metrics</div>
#             <div class="metric-row"><span class="metric-key">Average Speed</span><span class="metric-val">{summary.get('avg_speed',0):.1f} µm/s</span></div>
#             <div class="metric-row"><span class="metric-key">Maximum Speed</span><span class="metric-val">{summary.get('max_speed',0):.1f} µm/s</span></div>
#             <div class="metric-row"><span class="metric-key">Minimum Speed</span><span class="metric-val">{summary.get('min_speed',0):.1f} µm/s</span></div>
#             <div class="metric-row"><span class="metric-key">Speed Std Dev</span><span class="metric-val">{summary.get('std_speed',0):.1f} µm/s</span></div>
#         </div>
#     </div>

#     <div>
#         <div class="section-header">Concentration Metrics</div>
#         <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">
#             <div class="kpi-card">
#                 <div class="kpi-label">Concentration</div>
#                 <div class="kpi-value" style="font-size:22px;">{summary.get('concentration_mpm',0):.2f}M</div>
#                 <div class="kpi-sub">sperm/mL</div>
#             </div>
#             <div class="kpi-card">
#                 <div class="kpi-label">Total Sperm/mL</div>
#                 <div class="kpi-value" style="font-size:22px;">{summary.get('sperm_per_ml',0):,}</div>
#             </div>
#             <div class="kpi-card">
#                 <div class="kpi-label">TMSC</div>
#                 <div class="kpi-value" style="font-size:22px;">{summary.get('tmsc_mpm',0):.1f}M</div>
#                 <div class="kpi-sub">motile/mL</div>
#             </div>
#         </div>
#     </div>

#     <div style="margin-top:20px;">
#         <div class="section-header">Overall Assessment</div>
#         <div class="assessment-block">{assessment}</div>
#     </div>

#     <div style="margin-top:16px;color:#5b7fa6;font-size:11px;">
#         Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
#         YOLO Detection + ByteTrack | Calibration: {cal.get('magnification','')} @ {cal.get('um_per_px',0)} µm/px
#     </div>
# </div>
# """, unsafe_allow_html=True)

# def render_summary(summary: dict, cal: dict):
#     assessment = generate_assessment(summary, cal)
#     summary_ph.markdown(f"""
#     <div class="summary-panel">
#         <h3 style="color:#e8f0fe;margin-top:0;">📋 Analysis Summary</h3>

#         <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
#             <div>
#                 <div class="section-header">Calibration</div>
#                 <div class="metric-row"><span class="metric-key">Sample Volume</span><span class="metric-val">{cal.get('sample_volume_ul',50):.0f} µL</span></div>
#                 <div class="metric-row"><span class="metric-key">Magnification</span><span class="metric-val">{cal.get('magnification','')}</span></div>
#                 <div class="metric-row"><span class="metric-key">Pixel-to-Micron</span><span class="metric-val">{cal.get('um_per_px',0)} µm/px</span></div>
#                 <div class="metric-row"><span class="metric-key">Resolution</span><span class="metric-val">{cal.get('res_w',0)}×{cal.get('res_h',0)}</span></div>
#             </div>
#             <div>
#                 <div class="section-header">Count Metrics</div>
#                 <div class="metric-row"><span class="metric-key">Total Sperm Count</span><span class="metric-val">{summary.get('total_count',0):,}</span></div>
#                 <div class="metric-row"><span class="metric-key">Alive Sperm</span><span class="metric-val" style="color:#34d399">{summary.get('alive_count',0):,}</span></div>
#                 <div class="metric-row"><span class="metric-key">Dead Sperm</span><span class="metric-val" style="color:#f87171">{summary.get('dead_count',0):,}</span></div>
#                 <div class="metric-row"><span class="metric-key">Progressive</span><span class="metric-val">{summary.get('progressive_count',0):,}</span></div>
#                 <div class="metric-row"><span class="metric-key">Non-Progressive</span><span class="metric-val">{summary.get('nonprogressive_count',0):,}</span></div>
#             </div>
#         </div>

#         <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
#             <div>
#                 <div class="section-header">Percentage Metrics</div>
#                 <div class="metric-row"><span class="metric-key">Alive %</span><span class="metric-val">{summary.get('alive_pct',0):.1f}%</span></div>
#                 <div class="metric-row"><span class="metric-key">Dead %</span><span class="metric-val" style="color:#f87171">{summary.get('dead_pct',0):.1f}%</span></div>
#                 <div class="metric-row"><span class="metric-key">Motility %</span><span class="metric-val" style="color:#60a5fa">{summary.get('motility_pct',0):.1f}%</span></div>
#                 <div class="metric-row"><span class="metric-key">Progressive Motility %</span><span class="metric-val" style="color:#34d399">{summary.get('progressive_pct',0):.1f}%</span></div>
#             </div>
#             <div>
#                 <div class="section-header">Speed Metrics</div>
#                 <div class="metric-row"><span class="metric-key">Average Speed</span><span class="metric-val">{summary.get('avg_speed',0):.1f} µm/s</span></div>
#                 <div class="metric-row"><span class="metric-key">Maximum Speed</span><span class="metric-val">{summary.get('max_speed',0):.1f} µm/s</span></div>
#                 <div class="metric-row"><span class="metric-key">Minimum Speed</span><span class="metric-val">{summary.get('min_speed',0):.1f} µm/s</span></div>
#                 <div class="metric-row"><span class="metric-key">Speed Std Dev</span><span class="metric-val">{summary.get('std_speed',0):.1f} µm/s</span></div>
#             </div>
#         </div>

#         <div>
#             <div class="section-header">Concentration Metrics</div>
#             <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">
#                 <div class="kpi-card">
#                     <div class="kpi-label">Concentration</div>
#                     <div class="kpi-value" style="font-size:22px;">{summary.get('concentration_mpm',0):.2f}M</div>
#                     <div class="kpi-sub">sperm/mL</div>
#                 </div>
#                 <div class="kpi-card">
#                     <div class="kpi-label">Total Sperm/mL</div>
#                     <div class="kpi-value" style="font-size:22px;">{summary.get('sperm_per_ml',0):,}</div>
#                 </div>
#                 <div class="kpi-card">
#                     <div class="kpi-label">TMSC</div>
#                     <div class="kpi-value" style="font-size:22px;">{summary.get('tmsc_mpm',0):.1f}M</div>
#                     <div class="kpi-sub">motile/mL</div>
#                 </div>
#             </div>
#         </div>

#         <div style="margin-top:20px;">
#             <div class="section-header">Overall Assessment</div>
#             <div class="assessment-block">{assessment}</div>
#         </div>

#         <div style="margin-top:16px;color:#5b7fa6;font-size:11px;">
#             Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
#             YOLO Detection + ByteTrack | Calibration: {cal.get('magnification','')} @ {cal.get('um_per_px',0)} µm/px
#         </div>
#     </div>
#     """, unsafe_allow_html=True)


def render_export(summary: dict, cal: dict, frame_data: list):
    with export_ph.container():
        st.markdown('<div class="section-header" style="margin-top:20px;">Export Reports</div>', unsafe_allow_html=True)
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            csv_data = to_csv(summary, cal, frame_data)
            st.download_button("📥 Download CSV", data=csv_data,
                               file_name=f"sperm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv", use_container_width=True)
        with ec2:
            try:
                xlsx_data = to_excel(summary, cal, frame_data)
                st.download_button("📊 Download Excel", data=xlsx_data,
                                   file_name=f"sperm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            except Exception as e:
                st.info(f"Excel export requires openpyxl: pip install openpyxl")
        with ec3:
            # JSON export as PDF alternative (PDF requires reportlab)
            json_data = json.dumps({
                "calibration": cal,
                "summary": {k: (round(v,4) if isinstance(v,float) else v)
                            for k,v in summary.items()},
                "assessment": generate_assessment(summary, cal),
                "timestamp": datetime.now().isoformat()
            }, indent=2)
            st.download_button("📄 Download JSON Report", data=json_data,
                               file_name=f"sperm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                               mime="application/json", use_container_width=True)


def render_casa_table(summary: dict):
    casa_data = {
        "VCL (Curvilinear Velocity)": f"{summary.get('vcl',0):.1f} µm/s",
        "VSL (Straight-Line Velocity)": f"{summary.get('vsl',0):.1f} µm/s",
        "VAP (Average Path Velocity)": f"{summary.get('vap',0):.1f} µm/s",
        "LIN (Linearity)": f"{summary.get('lin',0):.1f}%",
        "STR (Straightness)": f"{summary.get('str',0):.1f}%",
        "WOB (Wobble)": f"{summary.get('wob',0):.1f}%",
        "ALH (Lateral Head Disp.)": f"{summary.get('alh',0):.2f} µm",
        "BCF (Beat Cross Freq.)": f"{summary.get('bcf',0):.1f} Hz",
    }
    rows_html = "".join(
        f'<div class="metric-row"><span class="metric-key">{k}</span>'
        f'<span class="metric-val">{v}</span></div>'
        for k, v in casa_data.items()
    )
    casa_table_ph.markdown(
        f'<div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:10px;padding:16px;">'
        f'<div class="section-header">CASA Metric Values</div>{rows_html}</div>',
        unsafe_allow_html=True
    )


# ─── Main analysis loop ──────────────────────────────────────────────────────
# if st.session_state.running:
#     processor: VideoProcessor = st.session_state.processor
#     cal = st.session_state.calibration
#     max_frames = int(cal["fps"] * analysis_duration)
#     frame_count = 0

#     # Generate blank canvas for simulation
#     W, H = cal["res_w"], cal["res_h"]

#     while st.session_state.running and frame_count < max_frames:
#         # Generate dark microscope-style background
#         bg = np.zeros((H, W, 3), dtype=np.uint8)
#         noise = np.random.randint(5, 25, (H, W, 3), dtype=np.uint8)
#         frame = cv2.add(bg, noise)

#         annotated, stats = processor.process_frame(frame)

if st.session_state.running:
    processor: VideoProcessor = st.session_state.processor
    cal = st.session_state.calibration
    max_frames = int(cal["fps"] * analysis_duration)
    frame_count = 0

    W, H = cal["res_w"], cal["res_h"]
    
    # 1. Setup Video Capture
    cap = None
    if source == "📁 Upload Video":
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            st.error("Please upload a video file first!")
            st.session_state.running = False
            st.rerun()

    elif source == "📷 Live Camera":
        if camera_id is None:
            st.warning("Analysis aborted: No valid camera selected. Please verify the hardware connection.")
            st.session_state.running = False
            st.rerun()
            
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            st.error(f"Cannot open Camera Device {camera_id}. It may be in use by another application.")
            st.session_state.running = False
            st.rerun()

    while st.session_state.running and frame_count < max_frames:
        
        # 2. Read frames strictly from capture source
        ret, frame = cap.read()
        if not ret:
            break # Reached the end of the video or camera disconnected
        
        # Resize video frame to match your calibration resolution
        frame = cv2.resize(frame, (W, H))

        annotated, stats = processor.process_frame(frame)
        frame_count += 1
        st.session_state.frames_processed = frame_count
        st.session_state.current_stats = stats
        st.session_state.frame_data = processor.frame_data

        # Convert frame for display
        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Render KPIs
        render_kpis(stats)

        # Live feed
        video_ph.image(img_rgb, channels="RGB", use_container_width=True)
        elapsed = time.time() - st.session_state.start_time
        progress_ph.progress(
            min(frame_count / max_frames, 1.0),
            text=f"Analyzing… {frame_count}/{max_frames} frames | {elapsed:.1f}s"
        )

        # Live charts
        pie_ph.plotly_chart(motility_pie(stats), use_container_width=True, key=f"pie_{frame_count}")
        render_rt_stats(stats)

        if len(processor.frame_data) > 3:
            hist_ph.plotly_chart(speed_histogram(processor.frame_data), use_container_width=True, key=f"hist_{frame_count}")
            spd_trend_ph.plotly_chart(speed_trend(processor.frame_data), use_container_width=True, key=f"spt_{frame_count}")
            cnt_trend_ph.plotly_chart(count_trend(processor.frame_data), use_container_width=True, key=f"cnt_{frame_count}")

        # CASA (update every 10 frames)
        if frame_count % 10 == 0:
            partial_summary = processor.get_global_summary()
            radar_ph.plotly_chart(casa_radar(partial_summary), use_container_width=True, key=f"rad_{frame_count}")
            render_casa_table(partial_summary)

        # Remove sleep here to let it process as fast as the camera/video allows
        # time.sleep(0.033) 

    # Done - Release hardware
    if cap is not None:
        cap.release()
        
    st.session_state.running = False
    st.session_state.summary = processor.get_global_summary()
    st.session_state.analysis_done = True
    st.rerun()

# ─── Static display after analysis ──────────────────────────────────────────
if not st.session_state.running and st.session_state.frame_data:
    stats = st.session_state.current_stats
    summary = st.session_state.summary
    cal = st.session_state.calibration
    frame_data = st.session_state.frame_data

    render_kpis(stats)

    if frame_data:
        pie_ph.plotly_chart(motility_pie(stats), use_container_width=True)
        render_rt_stats(stats)
        hist_ph.plotly_chart(speed_histogram(frame_data), use_container_width=True)
        spd_trend_ph.plotly_chart(speed_trend(frame_data), use_container_width=True)
        cnt_trend_ph.plotly_chart(count_trend(frame_data), use_container_width=True)

    if summary:
        radar_ph.plotly_chart(casa_radar(summary), use_container_width=True)
        render_casa_table(summary)
        render_summary(summary, cal)
        render_export(summary, cal, frame_data)

elif not st.session_state.running and not st.session_state.frame_data:
    # Welcome state
    kpi_ph.markdown("""
    <div class="alert-info">
        ⚙️ Configure calibration settings in the sidebar, then click <strong>▶ Start Analysis</strong> to begin.
    </div>
    """, unsafe_allow_html=True)

    video_ph.markdown("""
    <div style="background:#0d1526;border:2px dashed #1e3a5f;border-radius:12px;
                height:280px;display:flex;align-items:center;justify-content:center;
                color:#5b7fa6;font-size:14px;text-align:center;">
        <div>
            🔬<br><br>
            Live video feed will appear here<br>
            <small>Sperm tracked with bounding boxes, IDs, speed & motility status</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

import os
import re
import sys
import cv2
import numpy as np
import time
import threading
import argparse
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
from flask import Flask, Response, render_template_string, request, jsonify

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from snpehelper_manager import PerfProfile, Runtime, SnpeContext
except ImportError as e:
    print(f"Error importing SNPE-Helper modules: {e}")
    sys.exit(1)

# =============================================================================
# SCRFD Face Detection Logic
# =============================================================================
class SCRFD(SnpeContext):
    def __init__(self, dlc_path: str = "None",
                 input_layers: list = [],
                 output_layers: list = [],
                 output_tensors: list = [],
                 runtime: str = Runtime.CPU,
                 profile_level: str = PerfProfile.BALANCED,
                 enable_cache: bool = False,
                 input_size: tuple = (320, 320),
                 conf_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        
        super().__init__(dlc_path, input_layers, output_layers, output_tensors,
                        runtime, profile_level, enable_cache)

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.feat_stride_fpn = [8, 16, 32]
        self.num_anchors = 2

        self._anchor_centers = {}
        self._num_anchors = {}
        for stride in self.feat_stride_fpn:
            feat_h = input_size[0] // stride
            feat_w = input_size[1] // stride
            self._num_anchors[stride] = feat_h * feat_w * self.num_anchors

            anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1)
            anchor_centers = anchor_centers.astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))

            anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1)
            anchor_centers = anchor_centers.reshape((-1, 2))
            self._anchor_centers[stride] = anchor_centers

    def preprocess(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        self.orig_shape = image.shape[:2]

        input_image = cv2.resize(image, (self.input_size[1], self.input_size[0]))

        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        input_image = input_image.astype(np.float32)
        input_image = (input_image - 127.5) / 128.0

        input_image_flat = input_image.flatten()
        self.SetInputBuffer(input_image_flat, "input.1")

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]

        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])

        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]

            if max_shape is not None:
                px = np.clip(px, 0, max_shape[1])
                py = np.clip(py, 0, max_shape[0])

            preds.append(np.stack([px, py], axis=-1))

        return np.stack(preds, axis=1)

    def nms(self, dets, scores):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self):
        output_mapping = {
            8: {'score': '446', 'bbox': '449', 'kps': '452'},
            16: {'score': '466', 'bbox': '469', 'kps': '472'},
            32: {'score': '486', 'bbox': '489', 'kps': '492'}
        }

        all_bboxes = []
        all_scores = []
        all_kps = []

        for stride in self.feat_stride_fpn:
            mapping = output_mapping[stride]
            num_pred = self._num_anchors[stride]

            score_output = self.GetOutputBuffer(mapping['score'])
            bbox_output = self.GetOutputBuffer(mapping['bbox'])
            kps_output = self.GetOutputBuffer(mapping['kps'])

            scores = score_output.reshape((num_pred, 1))
            bboxes = bbox_output.reshape((num_pred, 4))
            kps = kps_output.reshape((num_pred, 10))

            anchor_centers = self._anchor_centers[stride]

            bboxes = bboxes * stride
            pos_bboxes = self.distance2bbox(anchor_centers, bboxes)

            kps = kps * stride
            pos_kps = self.distance2kps(anchor_centers, kps)

            all_bboxes.append(pos_bboxes)
            all_scores.append(scores)
            all_kps.append(pos_kps)

        all_bboxes = np.vstack(all_bboxes)
        all_scores = np.vstack(all_scores).squeeze()
        all_kps = np.vstack(all_kps)

        valid_mask = all_scores > self.conf_threshold
        bboxes = all_bboxes[valid_mask]
        scores = all_scores[valid_mask]
        kps = all_kps[valid_mask]

        if len(bboxes) > 0:
            keep = self.nms(bboxes, scores)
            bboxes = bboxes[keep]
            scores = scores[keep]
            kps = kps[keep]

        scale_x = self.orig_shape[1] / self.input_size[1]
        scale_y = self.orig_shape[0] / self.input_size[0]

        detections = []
        for bbox, score, kp in zip(bboxes, scores, kps):
            detection = {
                'bbox': [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y
                ],
                'score': float(score),
                'landmarks': kp * np.array([scale_x, scale_y])
            }
            detections.append(detection)

        return detections

# =============================================================================
# ArcFace Face Recognition Logic
# =============================================================================
class ArcFace(SnpeContext):
    def __init__(self, dlc_path: str = "None",
                 input_layers: list = [],
                 output_layers: list = [],
                 output_tensors: list = [],
                 runtime: str = Runtime.CPU,
                 profile_level: str = PerfProfile.BALANCED,
                 enable_cache: bool = False,
                 input_size: tuple = (112, 112)):

        super().__init__(dlc_path, input_layers, output_layers, output_tensors,
                        runtime, profile_level, enable_cache)

        self.input_size = input_size
        self.embedding_dim = 512

    def align_face(self, img, landmarks):
        src_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] 
        ], dtype=np.float32)
        
        dst_pts = np.array(landmarks, dtype=np.float32)
        
        tform, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.LMEDS)
        if tform is None:
            return None
            
        aligned_face = cv2.warpAffine(img, tform, (112, 112), borderValue=0.0)
        return aligned_face

    def preprocess(self, face_image):
        if isinstance(face_image, Image.Image):
            face_image = np.array(face_image)

        input_image = cv2.resize(face_image, (self.input_size[1], self.input_size[0]))

        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        input_image = input_image.astype(np.float32)
        input_image = (input_image - 127.5) / 128.0

        input_image_flat = input_image.flatten()
        self.SetInputBuffer(input_image_flat, "data")

    def postprocess(self):
        embedding = self.GetOutputBuffer("fc1")
        embedding = embedding.reshape(self.embedding_dim)
        normalized_embedding = self.normalize_embedding(embedding)

        return {
            'embedding': normalized_embedding,
            'raw_embedding': embedding.copy()
        }

    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def get_embedding(self, face_image):
        self.preprocess(face_image)

        if not self.Execute():
            print("Error: Failed to execute ArcFace model")
            return None

        result = self.postprocess()
        return result['embedding']

# =============================================================================
# Face Database & Check-in / Log Logic
# =============================================================================
class FaceDatabase:
    def __init__(self, db_path="face_database"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.metadata_file = self.db_path / "metadata.json"
        self.embeddings_file = self.db_path / "embeddings.pkl"
        self.log_file = self.db_path / "checkin_log.json"
        
        self.metadata = {}
        self.embeddings = {}
        self.logs = []
        self.load()

    def load(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        if self.embeddings_file.exists():
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)

    def save(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def add_person(self, person_id, name, embedding, image_path=None):
        self.metadata[person_id] = {
            'name': name,
            'enrolled_at': datetime.now().isoformat(),
            'checkin_count': 0,      
            'last_checkin': None,    
            'image_path': str(image_path) if image_path else None,
            'embedding_shape': embedding.shape
        }
        self.embeddings[person_id] = embedding
        self.save()
        return True

    def record_checkin(self, person_id):
        if person_id not in self.metadata:
            return False

        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        
        last_checkin_str = self.metadata[person_id].get('last_checkin')
        
        is_new_day = True
        if last_checkin_str:
            last_date = last_checkin_str.split(' ')[0]
            if last_date == current_date:
                is_new_day = False

        if is_new_day:
            self.metadata[person_id]['checkin_count'] = self.metadata[person_id].get('checkin_count', 0) + 1
            
        self.logs.append({
            'person_id': person_id,
            'name': self.metadata[person_id]['name'],
            'timestamp': current_time,
            'is_counted': is_new_day
        })
        
        # เมื่อ Log ถึง 10,000 รายการ ให้ตัดเหลือเฉพาะ 30 วันล่าสุด (ป้องกันไฟล์ใหญ่เกินไป)
        if len(self.logs) >= 10000:
            cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            self.logs = [log for log in self.logs
                         if log['timestamp'].split(' ')[0] >= cutoff]

        self.metadata[person_id]['last_checkin'] = current_time
        self.save()
        return is_new_day

    def search(self, query_embedding, threshold=0.5, top_k=5):
        if not self.embeddings:
            return []

        matches = []
        for person_id, db_embedding in self.embeddings.items():
            similarity = float(np.dot(query_embedding, db_embedding))
            if similarity >= threshold:
                meta = self.metadata.get(person_id, {})
                matches.append({
                    'person_id': person_id,
                    'name': meta.get('name', 'Unknown'),
                    'similarity': similarity,
                    'checkin_count': meta.get('checkin_count', 0),
                    'last_checkin': meta.get('last_checkin', 'Never'),
                    'enrolled_at': meta.get('enrolled_at')
                })
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:top_k]

    def __len__(self):
        return len(self.metadata)

# =============================================================================
# Web Application Logic
# =============================================================================

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
db_lock = threading.Lock()
face_results = []
system_ready = False

scrfd_model = None
arcface_model = None
database = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Face Recognition & Attendance</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
:root {
    --bg-1: #0f172a;
    --bg-2: #1e293b;
    --glass: rgba(255,255,255,0.08);
    --border: rgba(255,255,255,0.15);
    --primary: #6366f1;
    --success: #22c55e;
    --danger: #ef4444;
    --text-main: #f8fafc;
    --text-sub: #cbd5f5;
    --highlight: #f59e0b;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Inter, system-ui, sans-serif;
    background:
        radial-gradient(1200px 600px at 10% 10%, #1e1b4b, transparent),
        linear-gradient(135deg, var(--bg-1), var(--bg-2));
    color: var(--text-main);
    min-height: 100vh;
}

.container {
    max-width: 1440px;
    margin: auto;
    padding: 32px;
}

h1 { text-align: center; font-size: 2.4rem; font-weight: 700; margin-bottom: 32px;}

.main-content {
    display: grid;
    grid-template-columns: 3fr 1.2fr;
    gap: 24px;
}

@media (max-width: 1024px) {
    .main-content { grid-template-columns: 1fr; }
}

.video-panel {
    background: var(--glass);
    padding: 16px;
    border: 1px solid var(--border);
    backdrop-filter: blur(16px);
    box-shadow: 0 20px 60px rgba(0,0,0,.35);
    border-radius: 12px;
}

#videoStream { width: 100%; background: #000; border-radius: 8px;}

.faces-panel {
    background: var(--glass);
    padding: 20px;
    border: 1px solid var(--border);
    backdrop-filter: blur(16px);
    max-height: 680px;
    overflow-y: auto;
    border-radius: 12px;
}

.faces-panel h2 { margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 10px;}

.face-card {
    background: linear-gradient(180deg, rgba(255,255,255,.12), rgba(255,255,255,.05));
    padding: 16px;
    margin-bottom: 14px;
    border: 1px solid var(--border);
    transition: .25s;
    border-radius: 8px;
}

.face-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(0,0,0,.35);
}

.face-card.identified { border-left: 5px solid var(--success); }
.face-card.unknown    { border-left: 5px solid var(--danger); }

.face-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.face-name { font-weight: 600; font-size: 1.1rem; }
.face-similarity { font-size: .85rem; color: var(--success); font-weight: bold;}

.checkin-info {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.1);
    font-size: 0.85rem;
    color: var(--text-sub);
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.checkin-badge { color: var(--highlight); font-weight: bold; }

.enroll-btn {
    background: linear-gradient(135deg, var(--primary), #818cf8);
    border: none;
    color: white;
    padding: 6px 14px;
    cursor: pointer;
    font-size: .85rem;
    transition: .25s;
    border-radius: 4px;
}

.enroll-btn:hover { transform: scale(1.05); }

.stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin-top: 28px;
}

.stat-box {
    background: var(--glass);
    padding: 22px;
    border: 1px solid var(--border);
    backdrop-filter: blur(16px);
    text-align: center;
    border-radius: 8px;
}

.stat-value { font-size: 2rem; font-weight: 700; margin-top: 8px; }
.stat-label { font-size: .75rem; letter-spacing: .15em; text-transform: uppercase; color: var(--text-sub); }

/* สไตล์สำหรับ 30-Day Dashboard */
.dashboard-section { margin-top: 40px; }
.dashboard-grid { 
    display: grid; 
    grid-template-columns: 2fr 1fr; 
    gap: 24px; 
    margin-top: 20px; 
}
@media (max-width: 1024px) { 
    .dashboard-grid { grid-template-columns: 1fr; } 
}
.chart-panel { 
    background: var(--glass); padding: 20px; 
    border: 1px solid var(--border); border-radius: 12px; 
    backdrop-filter: blur(16px); 
    min-height: 350px;
}
.table-panel { 
    background: var(--glass); padding: 20px; 
    border: 1px solid var(--border); border-radius: 12px; 
    backdrop-filter: blur(16px); 
    max-height: 350px; overflow-y: auto; 
}
table { width: 100%; border-collapse: collapse; text-align: left; }
th, td { padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); }
th { color: var(--text-sub); position: sticky; top: 0; background: #1e293b; z-index: 10;}

.modal {
    position: fixed; inset: 0; background: rgba(0,0,0,.65);
    backdrop-filter: blur(10px); display: flex; align-items: center;
    justify-content: center; z-index: 999; opacity: 0;
    pointer-events: none; transition: opacity .3s ease;
}

.modal.show { opacity: 1; pointer-events: auto; }

.modal-content {
    position: relative;
    background: linear-gradient(180deg, rgba(15,23,42,.95), rgba(15,23,42,.85));
    padding: 32px; width: 90%; max-width: 480px;
    border: 1px solid var(--border); box-shadow: 0 40px 80px rgba(0,0,0,.5);
    transform: scale(.92) translateY(20px); opacity: 0;
    transition: all .35s cubic-bezier(.22,.61,.36,1);
    border-radius: 12px;
}

.modal.show .modal-content { transform: scale(1) translateY(0); opacity: 1; }

.close {
    position: absolute; top: 18px; right: 22px; font-size: 26px;
    color: var(--text-sub); cursor: pointer;
}

.form-group { margin-bottom: 16px; }
.form-group label { display: block; margin-bottom: 6px; }
.form-group input {
    width: 100%; padding: 10px; background: rgba(255,255,255,.08);
    border: 1px solid var(--border); color: white; border-radius: 4px;
}

.modal-buttons { display: flex; justify-content: flex-end; gap: 10px; }
.btn { padding: 10px 20px; border: none; cursor: pointer; border-radius: 4px; }
.btn-primary { background: linear-gradient(135deg, var(--primary), #818cf8); color: white; }
.btn-secondary { background: rgba(255,255,255,.1); color: white; }
</style>
</head>

<body>
<div class="container">
    <h1>Face Recognition & Attendance System</h1>

    <div class="main-content">
        <div class="video-panel">
            <img id="videoStream" src="{{ url_for('video_feed') }}">
        </div>

        <div class="faces-panel">
            <h2>Live Feed</h2>
            <div id="facesList"></div>

            <div style="margin-top: 32px;">
                <h2>Recent Activity Log</h2>
                <div id="logTable" style="margin-top: 10px; font-size: 0.85rem; background: rgba(0,0,0,0.2); border-radius: 8px; padding: 10px;">
                    <div style="display: grid; grid-template-columns: 1fr 2fr; border-bottom: 1px solid var(--border); padding-bottom: 5px; margin-bottom: 5px; font-weight: bold;">
                        <div>Name</div>
                        <div>Time</div>
                    </div>
                    <div id="logEntries"></div>
                </div>
            </div>
        </div>
    </div>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-label">People Enrolled</div>
            <div class="stat-value" id="dbCount">--</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Current Faces Detected</div>
            <div class="stat-value" id="facesCount">--</div>
        </div>
    </div>

    <div class="dashboard-section">
        <h2>30-Day Attendance Dashboard</h2>
        <div class="dashboard-grid">
            <div class="chart-panel">
                <canvas id="attendanceChart"></canvas>
            </div>
            <div class="table-panel">
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Total Check-ins</th>
                        </tr>
                    </thead>
                    <tbody id="dashboardTableBody">
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<div id="enrollModal" class="modal">
    <div class="modal-content">
        <span class="close">&times;</span>
        <h2>Enroll New Person</h2>

        <form id="enrollForm">
            <div class="form-group">
                <label>Name</label>
                <input id="personName" required autocomplete="off">
            </div>
            <input type="hidden" id="faceIndex">
            <div class="modal-buttons">
                <button type="button" class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                <button type="submit" class="btn btn-primary">Enroll</button>
            </div>
        </form>
    </div>
</div>

<script>
let currentFacesData = []; 
let stagedEmbedding = null;
let spokenRecently = {};
let attendanceChart = null;

// ตั้งค่าสีเริ่มต้นสำหรับ Chart.js ให้เข้ากับธีมมืด
Chart.defaults.color = '#cbd5f5';

function speakInBrowser(text) {
    if (!("speechSynthesis" in window)) return;
    try { speechSynthesis.cancel(); } catch(e){}
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0; 
    u.pitch = 1.0;
    speechSynthesis.speak(u);
}

function updateFaces() {
    fetch('/get_faces')
        .then(r => r.json())
        .then(data => {
            currentFacesData = data.faces;
            document.getElementById('facesCount').textContent = data.faces.length;
            document.getElementById('dbCount').textContent = data.db_size;

            const facesList = document.getElementById('facesList');
            
            facesList.innerHTML = data.faces.length === 0
                ? `<div style="opacity:.6;text-align:center;padding:20px;">No faces detected in frame</div>`
                : data.faces.map((face,i) => `
                    <div class="face-card ${face.identified ? 'identified' : 'unknown'}">
                        <div class="face-header">
                            <div class="face-name">
                                ${face.identified ? '👤 ' + face.matches[0].name : '❓ Unknown'}
                            </div>
                            ${face.identified ?
                                `<div class="face-similarity">${(face.matches[0].similarity*100).toFixed(1)}%</div>` :
                                `<button class="enroll-btn" onclick="openEnrollModal(${i})">Enroll Face</button>`
                            }
                        </div>
                        
                        ${face.identified ? `
                            <div class="checkin-info">
                                <div>Days Attended: <span class="checkin-badge">${face.matches[0].checkin_count || 0}</span></div>
                                <div>Last Seen: ${face.matches[0].last_checkin || 'Just now'}</div>
                            </div>
                        ` : ''}
                    </div>
                `).join('');

            const logEntries = document.getElementById('logEntries');
            logEntries.innerHTML = data.logs.slice().reverse().map(log => `
                <div style="display: grid; grid-template-columns: 1fr 2fr; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <div style="${log.is_counted ? 'color: var(--success); font-weight: bold;' : ''}">${log.name}</div>
                    <div style="color: var(--text-sub);">${log.timestamp.split(' ')[1]}</div>
                </div>
            `).join('');
                
            const now = Date.now();
            let namesToSpeak = [];

            data.faces.forEach(face => {
                if (face.identified) {
                    const name = face.matches[0].name;
                    if (!spokenRecently[name] || (now - spokenRecently[name] > 10000)) {
                        namesToSpeak.push(name);
                        spokenRecently[name] = now;
                    }
                }
            });

            if (namesToSpeak.length > 0) {
                const textToSpeak = "Hello " + namesToSpeak.join(" and ");
                speakInBrowser(textToSpeak);
            }
        })
        .catch(err => console.error("Error fetching faces:", err));
}

// ฟังก์ชันสำหรับอัปเดต 30-Day Dashboard
function updateDashboard() {
    fetch('/api/dashboard')
        .then(r => r.json())
        .then(data => {
            const dates = data.map(d => d.date);
            const counts = data.map(d => d.count);

            // 1. อัปเดตตาราง (Table) เรียงจากวันล่าสุดขึ้นก่อน
            const tbody = document.getElementById('dashboardTableBody');
            tbody.innerHTML = data.slice().reverse().map(d => `
                <tr>
                    <td>${d.date}</td>
                    <td style="font-weight:bold; color:var(--success)">${d.count}</td>
                </tr>
            `).join('');

            // 2. อัปเดตกราฟ (Chart.js)
            if (attendanceChart) {
                attendanceChart.data.labels = dates;
                attendanceChart.data.datasets[0].data = counts;
                attendanceChart.update();
            } else {
                const ctx = document.getElementById('attendanceChart').getContext('2d');
                attendanceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: dates,
                        datasets: [{
                            label: 'Daily Check-ins',
                            data: counts,
                            borderColor: '#6366f1',
                            backgroundColor: 'rgba(99, 102, 241, 0.2)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.3,
                            pointBackgroundColor: '#22c55e'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { 
                            legend: { display: false },
                            tooltip: { mode: 'index', intersect: false }
                        },
                        scales: {
                            y: { beginAtZero: true, ticks: { stepSize: 1 } },
                            x: { grid: { color: 'rgba(255,255,255,0.05)' } }
                        }
                    }
                });
            }
        })
        .catch(err => console.error("Error fetching dashboard data:", err));
}

function openEnrollModal(index) {
    stagedEmbedding = currentFacesData[index].embedding;
    const enrollModal = document.getElementById('enrollModal');
    enrollModal.classList.add('show');
    setTimeout(() => document.getElementById('personName').focus(), 150);
}

function closeModal() {
    document.getElementById('enrollModal').classList.remove('show');
    document.getElementById('enrollForm').reset();
    stagedEmbedding = null; 
}

document.querySelector('.close').onclick = closeModal;
window.onclick = e => { if (e.target === document.getElementById('enrollModal')) closeModal(); };
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

document.getElementById('enrollForm').addEventListener('submit', function(e) {
    e.preventDefault();
    if (!stagedEmbedding) { alert("Error: No face data was captured."); return; }

    const nameInput = document.getElementById('personName').value;
    const generatedPersonId = 'person_' + Math.random().toString(36).slice(2,9);
    
    const submitBtn = this.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = "Saving...";
    submitBtn.disabled = true;

    fetch('/enroll', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            embedding: stagedEmbedding,
            name: nameInput,
            person_id: generatedPersonId
        })
    })
    .then(r => r.json())
    .then(data => {
        if (!data.success) { alert("Enrollment failed: " + data.error); }
        closeModal();
        updateFaces(); 
        updateDashboard(); // รีเฟรชหน้า Dashboard ทันทีเมื่อมีคนใหม่
    })
    .catch(err => {
        console.error("Enrollment error:", err);
        alert("A network error occurred.");
        closeModal();
    })
    .finally(() => {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    });
});

// เริ่มต้นทำงาน
setInterval(updateFaces, 1000);
updateFaces();

// ดึงข้อมูล Dashboard ตอนเปิดเว็บ และอัปเดตทุกๆ 30 วินาที
updateDashboard();
setInterval(updateDashboard, 30000); 

</script>
</body>
</html>
"""

def detection_thread(camera_id, scrfd, arcface, db, skip_frames=1, threshold=0.5):
    global output_frame, lock, face_results, system_ready

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print(f"Webcam opened successfully")
    system_ready = True

    frame_count = 0
    last_faces = []
    
    fps = 0.0
    fps_frame_count = 0
    fps_start_time = time.time()

    checkin_cooldowns = {} 
    COOLDOWN_SECONDS = 5 

    os.makedirs("debug_faces", exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            # print("image size:", frame.shape)

            frame_count += 1

            if frame_count % (skip_frames + 1) == 1:
                scrfd.preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if scrfd.Execute():
                    detections = scrfd.postprocess()

                    faces = []
                    for det in detections:
                        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 > x1 and y2 > y1:
                            aligned_face = arcface.align_face(frame, det['landmarks'])
                            if aligned_face is None:
                                aligned_face = cv2.resize(frame[y1:y2, x1:x2], (112, 112))                      
                            
                            embedding = arcface.get_embedding(aligned_face)
                            if embedding is not None:
                                norm = np.linalg.norm(embedding)
                                matches = db.search(embedding, threshold=threshold, top_k=1)

                                faces.append({
                                    'bbox': np.array(det['bbox']).astype(float).tolist(),
                                    'detection_score': float(det['score']),
                                    'landmarks': np.array(det['landmarks']).astype(float).tolist(),
                                    'embedding': embedding,
                                    'matches': matches,
                                    'identified': len(matches) > 0,
                                })

                    if len(faces) > 1:
                        person_best_match = {}

                        for i, face in enumerate(faces):
                            if face['identified'] and len(face['matches']) > 0:
                                person_id = face['matches'][0]['person_id']
                                similarity = face['matches'][0]['similarity']
                                combined_score = similarity 

                                if person_id not in person_best_match:
                                    person_best_match[person_id] = (i, combined_score)
                                else:
                                    if combined_score > person_best_match[person_id][1]:
                                        person_best_match[person_id] = (i, combined_score)

                        best_indices = {idx for idx, _ in person_best_match.values()}
                        for i, face in enumerate(faces):
                            if face['identified'] and i not in best_indices:
                                face['matches'] = []
                                face['identified'] = False

                    # --- ระบบ CHECK-IN & LOGGING ---
                    current_time = time.time()
                    for face in faces:
                        if face['identified'] and len(face['matches']) > 0:
                            person_id = face['matches'][0]['person_id']

                            last_checked = checkin_cooldowns.get(person_id, 0)
                            if current_time - last_checked > COOLDOWN_SECONDS:
                                with db_lock:
                                    db.record_checkin(person_id)
                                    checkin_cooldowns[person_id] = current_time
                                    meta = db.metadata.get(person_id, {})
                                face['matches'][0]['checkin_count'] = meta.get('checkin_count')
                                face['matches'][0]['last_checkin'] = meta.get('last_checkin')
                    # --------------------------------

                    last_faces = faces

            annotated = frame.copy()
            for face in last_faces:
                bbox = face['bbox']
                x1, y1, x2, y2 = [int(v) for v in bbox]

                if face['identified']:
                    color = (0, 255, 0)
                    name = face['matches'][0]['name']
                    last_checkin = face['matches'][0].get('last_checkin', '').split()[1] if 'last_checkin' in face['matches'][0] else ''
                    label = f"{name} ({last_checkin})"
                else:
                    color = (0, 0, 255)
                    label = "Unknown"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                for lx, ly in face['landmarks']:
                    cv2.circle(annotated, (int(lx), int(ly)), 2, (255, 0, 0), -1)

            # Calculate and draw FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            with lock:
                output_frame = annotated.copy()
                face_results = last_faces.copy()

    except Exception as e:
        print(f"Error in detection thread: {e}")
    finally:
        cap.release()


def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                frame_copy = None
            else:
                frame_copy = output_frame.copy()

        if frame_copy is None:
            time.sleep(0.1)   # sleep OUTSIDE lock
            continue

        flag, encoded_image = cv2.imencode(".jpg", frame_copy)
        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_faces')
def get_faces():
    global face_results, lock, database, db_lock
    with lock:
        faces = [
            {
                'bbox': f['bbox'],
                'detection_score': f['detection_score'],
                'matches': f['matches'],
                'identified': f['identified'],
                'embedding': f['embedding'].tolist()
            }
            for f in face_results
        ]

    with db_lock:
        db_size = len(database)
        recent_logs = list(database.logs[-10:])

    return jsonify({
        'faces': faces,
        'db_size': db_size,
        'logs': recent_logs
    })


# API ใหม่สำหรับดึงข้อมูลสรุป 30 วัน
@app.route('/api/dashboard')
def api_dashboard():
    global database, db_lock

    # Snapshot logs under db_lock, release immediately — do heavy work outside lock
    with db_lock:
        logs_snapshot = list(database.logs)

    today = datetime.now().date()
    last_30_days = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(29, -1, -1)]
    daily_counts = {date: 0 for date in last_30_days}

    for log in logs_snapshot:
        if log.get('is_counted', False):
            date_str = log['timestamp'].split(' ')[0]
            if date_str in daily_counts:
                daily_counts[date_str] += 1

    result = [{"date": d, "count": c} for d, c in daily_counts.items()]
    return jsonify(result)


@app.route('/enroll', methods=['POST'])
def enroll():
    global database, db_lock
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No JSON body'}), 400

    name = data.get('name', '').strip()
    person_id = data.get('person_id', '')
    embedding_list = data.get('embedding')

    if not name or len(name) > 100:
        return jsonify({'success': False, 'error': 'Invalid name (1-100 chars)'}), 400
    if not person_id or not re.match(r'^[\w\-]{1,64}$', person_id):
        return jsonify({'success': False, 'error': 'Invalid person_id'}), 400
    if not embedding_list or not isinstance(embedding_list, list):
        return jsonify({'success': False, 'error': 'Missing embedding'}), 400

    try:
        embedding = np.array(embedding_list, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        with db_lock:
            database.add_person(person_id, name, embedding, None)
            database.record_checkin(person_id)

        return jsonify({'success': True, 'person_id': person_id, 'name': name})
    except Exception as e:
        print(f"Backend error during enrollment: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def main():
    global scrfd_model, arcface_model, database

    parser = argparse.ArgumentParser(description='Web-Based Face Recognition System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--db-path', default='face_database', help='Database directory')
    parser.add_argument('--scrfd-dlc', default='../SCRFD (Face Detection)/Model/scrfd.dlc')
    parser.add_argument('--arcface-dlc', default='../ArcFace (Face Recognition)/Model/arcface_quantized_6490.dlc')
    parser.add_argument('--runtime', default='DSP', choices=['CPU', 'DSP'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold')
    parser.add_argument('--skip-frames', type=int, default=1, help='Process every N frames')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')

    args = parser.parse_args()

    print("="*60)
    print("Web-Based Face Recognition System")
    print("="*60)

    print("\nInitializing database...")
    database = FaceDatabase(args.db_path)
    print(f"✓ Database loaded ({len(database)} people)")

    print("\nInitializing models...")
    runtime = Runtime.DSP if args.runtime == 'DSP' else Runtime.CPU

    scrfd_model = SCRFD(
        dlc_path=args.scrfd_dlc,
        input_layers=["input.1"],
        output_layers=[
            "Sigmoid_141", "Reshape_144", "Reshape_147",
            "Sigmoid_159", "Reshape_162", "Reshape_165",
            "Sigmoid_177", "Reshape_180", "Reshape_183"
        ],
        output_tensors=[
            "446", "449", "452",
            "466", "469", "472",
            "486", "489", "492"
        ],
        runtime=runtime,
        profile_level=PerfProfile.BURST
    )

    arcface_model = ArcFace(
        dlc_path=args.arcface_dlc,
        input_layers=["data"],
        output_layers=["pre_fc1"],
        output_tensors=["fc1"],
        runtime=runtime,
        profile_level=PerfProfile.BURST
    )

    if not scrfd_model.Initialize():
        print("Error: Failed to initialize SCRFD!")
        return 1

    if not arcface_model.Initialize():
        print("Error: Failed to initialize ArcFace!")
        return 1

    print("✓ Models initialized")

    print("\nStarting webcam detection...")
    detection_process = threading.Thread(
        target=detection_thread,
        args=(args.camera, scrfd_model, arcface_model, database, args.skip_frames, args.threshold),
        daemon=True
    )
    detection_process.start()

    time.sleep(2)

    print(f"\n{'='*60}")
    print(f"🌐 Web Interface Starting...")
    print(f"{'='*60}")
    print(f"\n📺 Open your browser:")
    print(f"   http://localhost:{args.port}")
    print(f"\n   Or from another device:")
    print(f"   http://<your-ip>:{args.port}")
    print(f"\nPress Ctrl+C to stop")
    print(f"{'='*60}\n")

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    return 0

if __name__ == "__main__":
    exit(main())
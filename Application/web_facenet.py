import os
import sys
import cv2
import numpy as np
import time
import threading
import argparse
import json
import pickle
import base64
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
import pytz

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

from flask import Flask, Response, render_template, request, jsonify 

try:
    import mediapipe as mp
except ImportError:
    print("Error: mediapipe is required for hand gestures. Run 'pip install mediapipe'")
    sys.exit(1)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from snpehelper_manager import PerfProfile, Runtime, SnpeContext
except ImportError as e:
    print(f"Error importing SNPE-Helper modules: {e}")
    sys.exit(1)

LOCAL_TZ = pytz.timezone('Asia/Bangkok')
# =============================================================================
# SCRFD Face Detection Logic (SNPE)
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
# FaceNet Face Recognition Logic (Deep Metric Learning)
# =============================================================================
class FaceNet_Model:
    def __init__(self, input_size=(160, 160)):
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading FaceNet model on {self.device}...")
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def align_face(self, img, landmarks):
        src_pts = np.array([
            [54.7065, 73.8514], [105.0454, 73.5734],
            [80.0360, 102.4808], [59.3561, 131.9507], [101.0427, 131.7201] 
        ], dtype=np.float32)
        
        dst_pts = np.array(landmarks, dtype=np.float32)
        
        tform, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.LMEDS)
        if tform is None: return None
            
        return cv2.warpAffine(img, tform, self.input_size, borderValue=0.0)

    def get_embedding(self, face_image):
        if isinstance(face_image, Image.Image):
            face_image = np.array(face_image)

        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_tensor = self.transform(rgb_face).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(face_tensor).cpu().numpy().flatten()
            
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


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
        with open(self.metadata_file, 'w') as f: json.dump(self.metadata, f, indent=2)
        with open(self.embeddings_file, 'wb') as f: pickle.dump(self.embeddings, f)
        with open(self.log_file, 'w') as f: json.dump(self.logs, f, indent=2)

    def add_person(self, person_id, name, embedding, image_path=None):
        self.metadata[person_id] = {
            'name': name, 'enrolled_at': datetime.now().isoformat(),
            'checkin_count': 0, 'last_checkin': None,
            'image_path': str(image_path) if image_path else None
        }
        self.embeddings[person_id] = embedding
        self.save()
        return True

    def record_checkin(self, person_id):
        if person_id not in self.metadata: return False
        now = datetime.now(LOCAL_TZ)
        current_date, current_time = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d %H:%M:%S")
        
        last_checkin_str = self.metadata[person_id].get('last_checkin')
        is_new_day = True
        if last_checkin_str and last_checkin_str.split(' ')[0] == current_date:
            is_new_day = False

        if is_new_day:
            self.metadata[person_id]['checkin_count'] = self.metadata[person_id].get('checkin_count', 0) + 1
            
        self.logs.append({
            'person_id': person_id, 'name': self.metadata[person_id]['name'],
            'timestamp': current_time, 'is_counted': is_new_day
        })
        if len(self.logs) > 5000: self.logs = self.logs[-5000:]
        
        self.metadata[person_id]['last_checkin'] = current_time
        self.save()
        return is_new_day

    def search(self, query_embedding, threshold=0.7, top_k=1):
        if not self.embeddings: return []

        matches = []
        for person_id, db_embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, db_embedding)
            
            if similarity >= threshold: 
                meta = self.metadata.get(person_id, {})
                matches.append({
                    'person_id': person_id,
                    'name': meta.get('name', 'Unknown'),
                    'similarity': float(similarity), 
                    'checkin_count': meta.get('checkin_count', 0),
                    'last_checkin': meta.get('last_checkin', 'Never'),
                })
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:top_k]

    def __len__(self): return len(self.metadata)


# =============================================================================
# Web Application Logic
# =============================================================================

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
face_results = []
system_ready = False
staged_auto_enroll = None 

scrfd_model = None
facenet_model = None
database = None

def is_open_palm(hand_landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers_up = 0
    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers_up += 1
    return fingers_up >= 3

def detection_thread(camera_id, scrfd, face_model, db, skip_frames=1, threshold=0.7):
    global output_frame, lock, face_results, system_ready, staged_auto_enroll

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    print(f"Webcam opened successfully")
    system_ready = True

    frame_count = 0
    last_faces = []
    
    checkin_cooldowns = {} 
    COOLDOWN_SECONDS = 5 

    is_counting = False
    countdown_start = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1
            annotated = frame.copy()

            if frame_count % (skip_frames + 1) == 1:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                hand_results = hands.process(rgb_frame)
                hand_detected = False

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        if is_open_palm(hand_landmarks):
                            hand_detected = True
                            break

                if hand_detected:
                    if not is_counting:
                        is_counting = True
                        countdown_start = time.time()
                else:
                    is_counting = False

                scrfd.preprocess(rgb_frame)
                if scrfd.Execute():
                    detections = scrfd.postprocess()

                    faces = []
                    for det in detections:
                        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 > x1 and y2 > y1:
                            aligned_face = face_model.align_face(frame, det['landmarks'])
                            if aligned_face is None:
                                aligned_face = cv2.resize(frame[y1:y2, x1:x2], (160, 160))
                            
                            _, buffer = cv2.imencode('.jpg', aligned_face)
                            face_b64 = base64.b64encode(buffer).decode('utf-8')

                            embedding = face_model.get_embedding(aligned_face)
                            if embedding is not None:
                                matches = db.search(embedding, threshold=threshold, top_k=1)

                                faces.append({
                                    'bbox': np.array(det['bbox']).astype(float).tolist(),
                                    'detection_score': float(det['score']),
                                    'landmarks': np.array(det['landmarks']).astype(float).tolist(),
                                    'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                                    'face_image_b64': face_b64,
                                    'matches': matches,
                                    'identified': len(matches) > 0,
                                })

                    if len(faces) > 1:
                        person_best_match = {}
                        for i, face in enumerate(faces):
                            if face['identified'] and len(face['matches']) > 0:
                                person_id = face['matches'][0]['person_id']
                                similarity = face['matches'][0]['similarity']
                                if person_id not in person_best_match or similarity > person_best_match[person_id][1]:
                                    person_best_match[person_id] = (i, similarity)

                        best_indices = {idx for idx, _ in person_best_match.values()}
                        for i, face in enumerate(faces):
                            if face['identified'] and i not in best_indices:
                                face['matches'] = []
                                face['identified'] = False

                    current_time = time.time()
                    for face in faces:
                        if face['identified'] and len(face['matches']) > 0:
                            person_id = face['matches'][0]['person_id']
                            last_checked = checkin_cooldowns.get(person_id, 0)
                            if current_time - last_checked > COOLDOWN_SECONDS:
                                db.record_checkin(person_id)
                                checkin_cooldowns[person_id] = current_time 
                                meta = db.metadata.get(person_id, {})
                                face['matches'][0]['checkin_count'] = meta.get('checkin_count')
                                face['matches'][0]['last_checkin'] = meta.get('last_checkin')

                    last_faces = faces

            for face in last_faces:
                bbox = face['bbox']
                x1, y1, x2, y2 = [int(v) for v in bbox]

                if face['identified']:
                    color = (0, 255, 0)
                    name = face['matches'][0]['name']
                    last_checkin = face['matches'][0].get('last_checkin', '').split()[1] if 'last_checkin' in face['matches'][0] else ''
                    label = f"{name}"
                else:
                    color = (0, 0, 255)
                    label = "Unknown"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x1, y1 - h_txt - 10), (x1 + w_txt, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                for lx, ly in face['landmarks']:
                    cv2.circle(annotated, (int(lx), int(ly)), 2, (255, 0, 0), -1)

            if is_counting:
                elapsed = time.time() - countdown_start
                remaining = 3 - int(elapsed)

                if remaining > 0:
                    cv2.putText(annotated, f"Auto-Enroll: {remaining}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
                else:
                    is_counting = False
                    
                    largest_face = None
                    max_a = 0
                    for f in last_faces:
                        if not f['identified']:
                            bbox = f['bbox']
                            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            if area > max_a:
                                max_a = area
                                largest_face = f

                    if largest_face:
                        with lock:
                            staged_auto_enroll = {
                                'embedding': largest_face['embedding'],
                                'image': largest_face['face_image_b64']
                            }
                        cv2.putText(annotated, "CAPTURED!", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

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
                time.sleep(0.1)
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
        time.sleep(0.033)

@app.route('/')
def index():
    return render_template('index_facenet.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_faces')
def get_faces():
    global face_results, lock, database, staged_auto_enroll
    
    auto_data = None
    with lock:
        faces = [
            {
                'bbox': f['bbox'],
                'detection_score': f['detection_score'],
                'matches': f['matches'],
                'identified': f['identified'],
                'embedding': f['embedding'],
                'face_image_b64': f.get('face_image_b64', '')
            }
            for f in face_results
        ]
        
        if staged_auto_enroll is not None:
            auto_data = staged_auto_enroll
            staged_auto_enroll = None
        
    return jsonify({ 
        'faces': faces, 
        'db_size': len(database),
        'logs': database.logs[-10:],
        'auto_enroll': auto_data
    })

@app.route('/api/dashboard')
def api_dashboard():
    global database, lock
    
    today = datetime.now(LOCAL_TZ).date()
    last_30_days = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(29, -1, -1)]
    
    daily_counts = {date: 0 for date in last_30_days}
    
    with lock:
        for log in database.logs:
            if log.get('is_counted', False):
                date_str = log['timestamp'].split(' ')[0]
                if date_str in daily_counts:
                    daily_counts[date_str] += 1
                    
    result = [{"date": d, "count": c} for d, c in daily_counts.items()]
    return jsonify(result)

@app.route('/enroll', methods=['POST'])
def enroll():
    global database
    data = request.json
    name = data.get('name')
    person_id = data.get('person_id')
    embedding_list = data.get('embedding')

    if not name or not embedding_list:
        return jsonify({'success': False, 'error': 'Missing face data or name'})

    try:
        embedding = np.array(embedding_list, dtype=np.float32)

        database.add_person(person_id, name, embedding, None)
        database.record_checkin(person_id) 
        
        return jsonify({ 'success': True, 'person_id': person_id, 'name': name })
    except Exception as e:
        print(f"Backend error during enrollment: {e}")
        return jsonify({'success': False, 'error': str(e)})

def main():
    global scrfd_model, facenet_model, database

    parser = argparse.ArgumentParser(description='Web-Based Face Recognition System')
    parser.add_argument('--camera', type=int, default=2, help='Camera ID')
    parser.add_argument('--db-path', default='face_database', help='Database directory')
    parser.add_argument('--scrfd-dlc', default='../SCRFD (Face Detection)/Model/scrfd.dlc')
    parser.add_argument('--runtime', default='DSP', choices=['CPU', 'DSP'])
    parser.add_argument('--threshold', type=float, default=0.7, help='Cosine similarity threshold (FaceNet)')
    parser.add_argument('--skip-frames', type=int, default=1, help='Process every N frames')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=8080, help='Web server port')

    args = parser.parse_args()

    print("="*60)
    print("Web-Based Face Recognition System (SCRFD + FaceNet + MediaPipe)")
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

    if not scrfd_model.Initialize():
        print("Error: Failed to initialize SCRFD!")
        return 1

    facenet_model = FaceNet_Model()

    print("✓ Models initialized")

    print("\nStarting webcam detection...")
    detection_process = threading.Thread(
        target=detection_thread,
        args=(args.camera, scrfd_model, facenet_model, database, args.skip_frames, args.threshold),
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
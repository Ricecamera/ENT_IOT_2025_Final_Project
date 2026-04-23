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
from MobileFacenet.mobilefacenet import MobileFacenet
from SCRFD.scrfd import SCRFD

from flask import Flask, Response, render_template, request, jsonify 
from face_database import FaceDatabase

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
    fps = 0.0
    fps_frame_count = 0
    fps_start_time = time.time()
    
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
            
            # Calculate and draw FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()
            cv2.putText(annotated, f"FPS: {fps:.1f}", (520, 30),
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

@app.route('/api/persons/<person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete a person from the database."""
    global database
    try:
        success = database.remove_person(person_id)
        if success:
            return jsonify({'success': True, 'message': f'Person {person_id} deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Person not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/database/stats', methods=['GET'])
def get_database_stats():
    """Get database statistics."""
    global database
    try:
        stats = database.get_statistics()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def main():
    global scrfd_model, facenet_model, database

    parser = argparse.ArgumentParser(description='Web-Based Face Recognition System')
    parser.add_argument('--camera', type=int, default=2, help='Camera ID')
    parser.add_argument('--db-path', default='face_database', help='Database directory')
    parser.add_argument('--scrfd-dlc', default='../SCRFD (Face Detection)/Model/scrfd.dlc')
    parser.add_argument('--facenet-dlc', default='../MobileFacenet/Model/mobileface_net.dlc')
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

    recognition_model = MobileFacenet(
        dlc_path=args.facenet_dlc,
        input_layers=["input"],
        output_layers=["/bn/BatchNormalization"],
        output_tensors=["/bn/BatchNormalization_output_0"],
        runtime="CPU", # FaceNet is not quantized, so we run it on CPU for better accuracy
        profile_level=PerfProfile.BURST
    )

    if not scrfd_model.Initialize():
        print("Error: Failed to initialize SCRFD!")
        return 1

    if not recognition_model.Initialize():
        print("Error: Failed to initialize MobileFace!")
        return 1

    print("✓ Models initialized")

    print("\nStarting webcam detection...")
    detection_process = threading.Thread(
        target=detection_thread,
        args=(args.camera, scrfd_model, recognition_model, database, args.skip_frames, args.threshold),
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
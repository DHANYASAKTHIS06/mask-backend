import os
import json
import uuid
import base64
import hashlib
import hmac
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from functools import wraps

import cv2
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_file, g
from flask_cors import CORS

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mask_system.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

SECRET_KEY = os.environ.get("SECRET_KEY", "mask-detection-secret-key-2024")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load ML Models ───────────────────────────────────────────────────────────
try:
    model = joblib.load(os.path.join(MODELS_DIR, "mask_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    gender_encoder = joblib.load(os.path.join(MODELS_DIR, "gender_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "model_stats.json")) as f:
        model_stats = json.load(f)
    logger.info("ML models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    model = scaler = gender_encoder = None
    model_stats = {}

# Load OpenCV face detector
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# ─── Database ─────────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    with app.app_context():
        db = sqlite3.connect(DB_PATH)
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                full_name   TEXT NOT NULL,
                email       TEXT UNIQUE NOT NULL,
                password    TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS detection_logs (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                username        TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                result          TEXT NOT NULL,
                confidence      REAL NOT NULL,
                entry_status    TEXT NOT NULL,
                image_path      TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        db.commit()
        db.close()
    logger.info("Database initialised.")

init_db()

# ─── Auth Helpers ─────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return salt.hex() + ":" + key.hex()

def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
        return hmac.compare_digest(key, new_key)
    except Exception:
        return False

def create_token(user_id: str, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": time.time() + 86400 * 7,   # 7 days
    }
    data = base64.b64encode(json.dumps(payload).encode()).decode()
    sig = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    return f"{data}.{sig}"

def verify_token(token: str):
    try:
        data, sig = token.rsplit(".", 1)
        expected = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.b64decode(data).decode())
        if payload["exp"] < time.time():
            return None
        return payload
    except Exception:
        return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "").strip()
        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        request.user = payload
        return f(*args, **kwargs)
    return decorated

# ─── ML Prediction Helpers ────────────────────────────────────────────────────
def simulate_image_features(img_array, n_features=20):
    """
    Extract pseudo-image features from a face region.
    Maps pixel statistics to the same feature space used in training.
    """
    if img_array is None or img_array.size == 0:
        return np.zeros(n_features)

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    gray = cv2.resize(gray, (64, 64))

    features = []
    # Region analysis: split face into regions
    regions = [
        gray[:32, :],      # top half (eyes/forehead)
        gray[32:, :],      # bottom half (nose/mouth)
        gray[16:48, 16:48],# center
        gray[:16, :],      # forehead
        gray[16:32, :],    # eye region
        gray[32:48, :],    # nose region
        gray[48:, :],      # mouth/chin region
    ]

    for region in regions:
        features.append(float(np.mean(region)))
        features.append(float(np.std(region)))

    # Texture: Laplacian variance (mask reduces texture)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    features.append(lap_var)

    # Edge density (masks reduce visible edge features)
    edges = cv2.Canny(gray, 100, 200)
    features.append(float(np.sum(edges) / edges.size))

    # Histogram contrast
    hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
    hist = hist.flatten() / hist.sum()
    features.extend(hist[:4].tolist())

    while len(features) < n_features:
        features.append(0.0)

    return np.array(features[:n_features], dtype=np.float32)


def predict_mask(image_b64: str):
    """
    Decode base64 image, detect face, extract features, predict mask.
    Returns dict with result, confidence, face_detected, image_path.
    """
    if model is None:
        return {"error": "Model not loaded"}

    # Decode image
    try:
        header, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        return {"error": f"Image decode failed: {e}"}

    # Save screenshot
    screenshot_name = f"{uuid.uuid4().hex}.jpg"
    screenshot_path = os.path.join(UPLOADS_DIR, screenshot_name)
    cv2.imwrite(screenshot_path, img)

    # Face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_detected = len(faces) > 0
    face_region = img  # fallback to full image if no face

    if face_detected:
        x, y, w, h = faces[0]
        # Add margin
        margin = int(0.2 * min(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        face_region = img[y1:y2, x1:x2]

    # Feature engineering (mirrors training)
    img_feats = simulate_image_features(face_region, n_features=20)

    # Estimate size_mb from image bytes
    size_mb = len(img_bytes) / (1024 * 1024)

    # Use neutral defaults for age/gender (not available at runtime)
    age = 30
    gender_enc = 1  # MALE encoded value as fallback

    X_base = np.array([[age, gender_enc, size_mb]])
    X = np.hstack([X_base, img_feats.reshape(1, -1)])
    X_scaled = scaler.transform(X)

    # Predict
    proba = model.predict_proba(X_scaled)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(np.max(proba))

    result = "mask" if pred_class == 1 else "no_mask"
    entry_status = "GRANTED" if result == "mask" else "DENIED"

    return {
        "result": result,
        "confidence": round(confidence * 100, 2),
        "entry_status": entry_status,
        "face_detected": face_detected,
        "faces_count": len(faces),
        "image_path": screenshot_name,
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_accuracy": model_stats.get("accuracy", 0),
        "timestamp": datetime.utcnow().isoformat(),
    })


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    full_name = data.get("full_name", "").strip()
    email     = data.get("email", "").strip().lower()
    password  = data.get("password", "").strip()
    confirm   = data.get("confirm_password", "").strip()

    if not all([full_name, email, password, confirm]):
        return jsonify({"error": "All fields are required"}), 400
    if password != confirm:
        return jsonify({"error": "Passwords do not match"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    db = get_db()
    if db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
        return jsonify({"error": "Email already registered"}), 409

    user_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO users (id, full_name, email, password, created_at) VALUES (?,?,?,?,?)",
        (user_id, full_name, email, hash_password(password), datetime.utcnow().isoformat()),
    )
    db.commit()

    token = create_token(user_id, email)
    return jsonify({
        "message": "Account created successfully",
        "token": token,
        "user": {"id": user_id, "full_name": full_name, "email": email},
    }), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data     = request.get_json() or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    db   = get_db()
    user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

    if not user or not verify_password(password, user["password"]):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_token(user["id"], email)
    return jsonify({
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user["id"],
            "full_name": user["full_name"],
            "email": user["email"],
            "created_at": user["created_at"],
        },
    })


@app.route("/api/auth/me", methods=["GET"])
@token_required
def me():
    db   = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (request.user["user_id"],)).fetchone()
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({
        "id": user["id"],
        "full_name": user["full_name"],
        "email": user["email"],
        "created_at": user["created_at"],
    })


# ── Detection ─────────────────────────────────────────────────────────────────

@app.route("/api/detect", methods=["POST"])
@token_required
def detect():
    data  = request.get_json() or {}
    image = data.get("image")
    if not image:
        return jsonify({"error": "No image provided"}), 400

    result = predict_mask(image)
    if "error" in result:
        return jsonify(result), 400

    # Persist log
    db       = get_db()
    user     = db.execute("SELECT full_name FROM users WHERE id=?", (request.user["user_id"],)).fetchone()
    username = user["full_name"] if user else "Unknown"
    log_id   = str(uuid.uuid4())
    now      = datetime.utcnow().isoformat()

    db.execute(
        """INSERT INTO detection_logs
           (id, user_id, username, timestamp, result, confidence, entry_status, image_path)
           VALUES (?,?,?,?,?,?,?,?)""",
        (log_id, request.user["user_id"], username, now,
         result["result"], result["confidence"],
         result["entry_status"], result.get("image_path")),
    )
    db.commit()

    return jsonify({
        **result,
        "log_id": log_id,
        "username": username,
        "timestamp": now,
    })


# ── Logs & History ────────────────────────────────────────────────────────────

@app.route("/api/logs", methods=["GET"])
@token_required
def get_logs():
    page     = max(1, int(request.args.get("page", 1)))
    per_page = min(100, int(request.args.get("per_page", 20)))
    offset   = (page - 1) * per_page

    db    = get_db()
    total = db.execute(
        "SELECT COUNT(*) FROM detection_logs WHERE user_id=?",
        (request.user["user_id"],),
    ).fetchone()[0]

    rows = db.execute(
        """SELECT * FROM detection_logs WHERE user_id=?
           ORDER BY timestamp DESC LIMIT ? OFFSET ?""",
        (request.user["user_id"], per_page, offset),
    ).fetchall()

    return jsonify({
        "logs": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    })


@app.route("/api/logs/all", methods=["GET"])
@token_required
def get_all_logs():
    """Admin view – returns all logs (useful for statistics)."""
    db   = get_db()
    rows = db.execute(
        "SELECT * FROM detection_logs ORDER BY timestamp DESC LIMIT 500"
    ).fetchall()
    return jsonify({"logs": [dict(r) for r in rows]})


@app.route("/api/logs/export", methods=["GET"])
@token_required
def export_csv():
    db   = get_db()
    rows = db.execute(
        "SELECT * FROM detection_logs WHERE user_id=? ORDER BY timestamp DESC",
        (request.user["user_id"],),
    ).fetchall()

    import io, csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Log ID", "Username", "Timestamp", "Result", "Confidence (%)", "Entry Status"])
    for r in rows:
        writer.writerow([r["id"], r["username"], r["timestamp"],
                         r["result"], r["confidence"], r["entry_status"]])

    output.seek(0)
    return app.response_class(
        output.read(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=detection_logs.csv"},
    )


# ── Statistics ────────────────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
@token_required
def stats():
    db  = get_db()
    uid = request.user["user_id"]

    total   = db.execute("SELECT COUNT(*) FROM detection_logs WHERE user_id=?", (uid,)).fetchone()[0]
    granted = db.execute("SELECT COUNT(*) FROM detection_logs WHERE user_id=? AND entry_status='GRANTED'", (uid,)).fetchone()[0]
    denied  = db.execute("SELECT COUNT(*) FROM detection_logs WHERE user_id=? AND entry_status='DENIED'",  (uid,)).fetchone()[0]
    avg_conf = db.execute("SELECT AVG(confidence) FROM detection_logs WHERE user_id=?", (uid,)).fetchone()[0] or 0

    compliance = round((granted / total * 100), 1) if total else 0

    # Last 7 days trend
    seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    trend_rows = db.execute(
        """SELECT date(timestamp) as day, COUNT(*) as count,
           SUM(CASE WHEN entry_status='GRANTED' THEN 1 ELSE 0 END) as granted
           FROM detection_logs
           WHERE user_id=? AND timestamp >= ?
           GROUP BY day ORDER BY day""",
        (uid, seven_days_ago),
    ).fetchall()

    return jsonify({
        "total_detections": total,
        "total_granted": granted,
        "total_denied": denied,
        "compliance_percentage": compliance,
        "avg_confidence": round(float(avg_conf), 2),
        "model_accuracy": round(model_stats.get("accuracy", 0) * 100, 2),
        "trend": [dict(r) for r in trend_rows],
    })


# ── Screenshots ───────────────────────────────────────────────────────────────

@app.route("/api/screenshot/<filename>", methods=["GET"])
@token_required
def get_screenshot(filename):
    path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="image/jpeg")


# ── Model Info ────────────────────────────────────────────────────────────────

@app.route("/api/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "algorithm": "Gradient Boosting Classifier",
        "framework": "scikit-learn 1.6.1",
        "features": "Face region pixel statistics + texture analysis",
        "classes": ["No Mask", "Mask"],
        "training_samples": model_stats.get("total", 40000),
        "accuracy": round(model_stats.get("accuracy", 0) * 100, 2),
        "face_detector": "OpenCV Haar Cascade",
        "dataset": "Medical Masks Dataset (CSV Metadata)",
    })


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

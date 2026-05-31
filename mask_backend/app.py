import os
import json
import uuid
import base64
import hashlib
import hmac
import time
import logging
from datetime import datetime, timedelta
from functools import wraps

import certifi
import cv2
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

SECRET_KEY = os.environ.get("SECRET_KEY", "mask-detection-secret-key-2024")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── MongoDB Connection ───────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is not set. Add it in Render → Environment.")

try:
    mongo_client = MongoClient(
        MONGO_URI,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000,
    )
    mongo_client.admin.command("ping")
    mongo_db  = mongo_client["maskguard"]
    users_col = mongo_db["users"]
    logs_col  = mongo_db["detection_logs"]
    users_col.create_index("email", unique=True)
    logger.info("MongoDB Atlas connected successfully.")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise

# ─── Load ML Models ───────────────────────────────────────────────────────────
try:
    model          = joblib.load(os.path.join(MODELS_DIR, "mask_model.pkl"))
    scaler         = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
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
face_cascade      = cv2.CascadeClassifier(face_cascade_path)

# ─── Auth Helpers ─────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key  = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return salt.hex() + ":" + key.hex()

def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(":")
        salt    = bytes.fromhex(salt_hex)
        key     = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
        return hmac.compare_digest(key, new_key)
    except Exception:
        return False

def create_token(user_id: str, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email":   email,
        "exp":     time.time() + 86400 * 7,
    }
    data = base64.b64encode(json.dumps(payload).encode()).decode()
    sig  = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    return f"{data}.{sig}"

def verify_token(token: str):
    try:
        data, sig = token.rsplit(".", 1)
        expected  = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
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
        auth    = request.headers.get("Authorization", "")
        token   = auth.replace("Bearer ", "").strip()
        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        request.user = payload
        return f(*args, **kwargs)
    return decorated

# ─── ML Prediction Helpers ────────────────────────────────────────────────────
def simulate_image_features(img_array, n_features=20):
    if img_array is None or img_array.size == 0:
        return np.zeros(n_features)

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    gray = cv2.resize(gray, (64, 64))

    features = []
    regions  = [
        gray[:32, :],
        gray[32:, :],
        gray[16:48, 16:48],
        gray[:16, :],
        gray[16:32, :],
        gray[32:48, :],
        gray[48:, :],
    ]
    for region in regions:
        features.append(float(np.mean(region)))
        features.append(float(np.std(region)))

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    features.append(lap_var)

    edges = cv2.Canny(gray, 100, 200)
    features.append(float(np.sum(edges) / edges.size))

    hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
    hist = hist.flatten() / hist.sum()
    features.extend(hist[:4].tolist())

    while len(features) < n_features:
        features.append(0.0)

    return np.array(features[:n_features], dtype=np.float32)


def predict_mask(image_b64: str):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        header, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
        img_bytes = base64.b64decode(encoded)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        return {"error": f"Image decode failed: {e}"}

    screenshot_name = f"{uuid.uuid4().hex}.jpg"
    screenshot_path = os.path.join(UPLOADS_DIR, screenshot_name)
    cv2.imwrite(screenshot_path, img)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_detected = len(faces) > 0
    face_region   = img

    if face_detected:
        x, y, w, h = faces[0]
        margin = int(0.2 * min(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        face_region = img[y1:y2, x1:x2]

    img_feats  = simulate_image_features(face_region, n_features=20)
    size_mb    = len(img_bytes) / (1024 * 1024)
    age        = 30
    gender_enc = 1

    X_base   = np.array([[age, gender_enc, size_mb]])
    X        = np.hstack([X_base, img_feats.reshape(1, -1)])
    X_scaled = scaler.transform(X)

    proba      = model.predict_proba(X_scaled)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(np.max(proba))

    result       = "mask" if pred_class == 1 else "no_mask"
    entry_status = "GRANTED" if result == "mask" else "DENIED"

    return {
        "result":        result,
        "confidence":    round(confidence * 100, 2),
        "entry_status":  entry_status,
        "face_detected": face_detected,
        "faces_count":   len(faces),
        "image_path":    screenshot_name,
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "model_loaded":   model is not None,
        "model_accuracy": model_stats.get("accuracy", 0),
        "database":       "MongoDB Atlas",
        "timestamp":      datetime.utcnow().isoformat(),
    })

# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def register():
    data      = request.get_json() or {}
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

    user_id  = str(uuid.uuid4())
    user_doc = {
        "id":         user_id,
        "full_name":  full_name,
        "email":      email,
        "password":   hash_password(password),
        "created_at": datetime.utcnow().isoformat(),
    }
    try:
        users_col.insert_one(user_doc)
    except DuplicateKeyError:
        return jsonify({"error": "Email already registered"}), 409

    token = create_token(user_id, email)
    return jsonify({
        "message": "Account created successfully",
        "token":   token,
        "user":    {"id": user_id, "full_name": full_name, "email": email},
    }), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data     = request.get_json() or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = users_col.find_one({"email": email})
    if not user or not verify_password(password, user["password"]):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_token(user["id"], email)
    return jsonify({
        "message": "Login successful",
        "token":   token,
        "user": {
            "id":         user["id"],
            "full_name":  user["full_name"],
            "email":      user["email"],
            "created_at": user["created_at"],
        },
    })


@app.route("/api/auth/me", methods=["GET"])
@token_required
def me():
    user = users_col.find_one({"id": request.user["user_id"]})
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({
        "id":         user["id"],
        "full_name":  user["full_name"],
        "email":      user["email"],
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

    user     = users_col.find_one({"id": request.user["user_id"]})
    username = user["full_name"] if user else "Unknown"
    log_id   = str(uuid.uuid4())
    now      = datetime.utcnow().isoformat()

    log_doc = {
        "id":           log_id,
        "user_id":      request.user["user_id"],
        "username":     username,
        "timestamp":    now,
        "result":       result["result"],
        "confidence":   result["confidence"],
        "entry_status": result["entry_status"],
        "image_path":   result.get("image_path"),
        "faces_count":  result.get("faces_count", 0),
    }
    logs_col.insert_one(log_doc)

    return jsonify({
        **result,
        "log_id":    log_id,
        "username":  username,
        "timestamp": now,
    })

# ── Logs & History ────────────────────────────────────────────────────────────

@app.route("/api/logs", methods=["GET"])
@token_required
def get_logs():
    page     = max(1, int(request.args.get("page", 1)))
    per_page = min(100, int(request.args.get("per_page", 20)))
    skip     = (page - 1) * per_page
    uid      = request.user["user_id"]

    total = logs_col.count_documents({"user_id": uid})
    rows  = list(logs_col.find(
        {"user_id": uid},
        {"_id": 0}
    ).sort("timestamp", -1).skip(skip).limit(per_page))

    return jsonify({
        "logs":     rows,
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    max(1, (total + per_page - 1) // per_page),
    })


@app.route("/api/logs/export", methods=["GET"])
@token_required
def export_csv():
    uid  = request.user["user_id"]
    rows = list(logs_col.find({"user_id": uid}, {"_id": 0}).sort("timestamp", -1))

    import io, csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Log ID", "Username", "Timestamp", "Result", "Confidence (%)", "Entry Status"])
    for r in rows:
        writer.writerow([r.get("id"), r.get("username"), r.get("timestamp"),
                         r.get("result"), r.get("confidence"), r.get("entry_status")])

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
    uid = request.user["user_id"]

    total   = logs_col.count_documents({"user_id": uid})
    granted = logs_col.count_documents({"user_id": uid, "entry_status": "GRANTED"})
    denied  = logs_col.count_documents({"user_id": uid, "entry_status": "DENIED"})

    pipeline_avg = [
        {"$match": {"user_id": uid}},
        {"$group": {"_id": None, "avg_conf": {"$avg": "$confidence"}}}
    ]
    avg_result = list(logs_col.aggregate(pipeline_avg))
    avg_conf   = avg_result[0]["avg_conf"] if avg_result else 0

    compliance = round((granted / total * 100), 1) if total else 0

    seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    pipeline_trend = [
        {"$match": {"user_id": uid, "timestamp": {"$gte": seven_days_ago}}},
        {"$group": {
            "_id":     {"$substr": ["$timestamp", 0, 10]},
            "count":   {"$sum": 1},
            "granted": {"$sum": {"$cond": [{"$eq": ["$entry_status", "GRANTED"]}, 1, 0]}}
        }},
        {"$sort": {"_id": 1}},
        {"$project": {"_id": 0, "day": "$_id", "count": 1, "granted": 1}}
    ]
    trend = list(logs_col.aggregate(pipeline_trend))

    return jsonify({
        "total_detections":      total,
        "total_granted":         granted,
        "total_denied":          denied,
        "compliance_percentage": compliance,
        "avg_confidence":        round(float(avg_conf), 2),
        "model_accuracy":        round(model_stats.get("accuracy", 0) * 100, 2),
        "trend":                 trend,
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
        "algorithm":        "Gradient Boosting Classifier",
        "framework":        "scikit-learn 1.6.1",
        "features":         "Face region pixel statistics + texture analysis",
        "classes":          ["No Mask", "Mask"],
        "training_samples": model_stats.get("total", 40000),
        "accuracy":         round(model_stats.get("accuracy", 0) * 100, 2),
        "face_detector":    "OpenCV Haar Cascade",
        "dataset":          "Medical Masks Dataset (CSV Metadata)",
    })

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

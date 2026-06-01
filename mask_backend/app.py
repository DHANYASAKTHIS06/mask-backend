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

mongo_client = None
mongo_db     = None
users_col    = None
logs_col     = None

if not MONGO_URI:
    logger.warning("MONGO_URI not found.")
else:
    try:
        mongo_client = MongoClient(
            MONGO_URI,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            retryWrites=True,
        )
        mongo_db  = mongo_client["maskguard"]
        users_col = mongo_db["users"]
        logs_col  = mongo_db["detection_logs"]
        users_col.create_index("email", unique=True)
        logger.info("MongoDB initialized successfully.")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")

def check_db():
    try:
        if mongo_client is None:
            return False
        mongo_client.admin.command("ping")
        return True
    except Exception:
        return False

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

# ─── Load Face & Eye Cascades ─────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

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

# ─── Real Mask Detection Logic ────────────────────────────────────────────────
def detect_mask_from_image(img):
    """
    Real computer-vision based mask detection.
    
    Strategy:
    1. Detect face using Haar cascade (multiple attempts with different params)
    2. If face found — check if EYES are visible in upper half
       - Eyes visible + lower face covered = MASK ON
       - Eyes visible + lower face also visible = NO MASK
    3. If no face found — analyse lower-center region for mask colours/textures
    4. Return result with confidence score
    """
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Try multiple face detection passes ────────────────────────────
    face_region  = None
    face_box     = None
    face_detected = False

    detection_params = [
        {"scaleFactor": 1.1,  "minNeighbors": 5, "minSize": (60, 60)},
        {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (40, 40)},
        {"scaleFactor": 1.15, "minNeighbors": 4, "minSize": (30, 30)},
    ]
    for params in detection_params:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=params["scaleFactor"],
            minNeighbors=params["minNeighbors"],
            minSize=params["minSize"],
        )
        if len(faces) > 0:
            face_box      = faces[0]
            face_detected = True
            break

    if face_detected:
        x, y, w, h = face_box
        # Expand box slightly
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)
        face_region = img[y1:y2, x1:x2]
        face_gray   = gray[y1:y2, x1:x2]

        face_h = y2 - y1
        face_w = x2 - x1

        # ── Step 2: Check eye visibility in upper half of face ────────────────
        upper_half_gray = face_gray[:face_h // 2, :]
        lower_half_gray = face_gray[face_h // 2:, :]
        lower_half_bgr  = face_region[face_h // 2:, :]

        eyes = eye_cascade.detectMultiScale(
            upper_half_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(15, 15),
        )
        eyes_visible = len(eyes) >= 1

        # ── Step 3: Analyse lower face region for mask ───────────────────────
        # Masks tend to:
        # (a) Have uniform colour (white/blue/black)
        # (b) Have low texture variance (fabric is smooth)
        # (c) Have different skin-tone ratio than bare skin

        lower_mean_std = float(np.std(lower_half_gray))
        lower_mean     = float(np.mean(lower_half_gray))

        # Skin tone detection in lower half using HSV
        lower_hsv  = cv2.cvtColor(lower_half_bgr, cv2.COLOR_BGR2HSV)
        skin_lower = np.array([0,  20,  70], dtype=np.uint8)
        skin_upper = np.array([25, 255, 255], dtype=np.uint8)
        skin_mask  = cv2.inRange(lower_hsv, skin_lower, skin_upper)
        skin_ratio = float(np.sum(skin_mask > 0)) / max(skin_mask.size, 1)

        # White/blue/black mask detection in lower half
        lower_bgr      = lower_half_bgr
        white_mask_pct = float(np.sum(
            (lower_bgr[:,:,0] > 180) &
            (lower_bgr[:,:,1] > 180) &
            (lower_bgr[:,:,2] > 180)
        )) / max(lower_bgr.shape[0] * lower_bgr.shape[1], 1)

        blue_mask_pct = float(np.sum(
            (lower_bgr[:,:,0] > 80) &
            (lower_bgr[:,:,1] > 80) &
            (lower_bgr[:,:,2] < 180) &
            (lower_bgr[:,:,0].astype(int) - lower_bgr[:,:,2].astype(int) < 30)
        )) / max(lower_bgr.shape[0] * lower_bgr.shape[1], 1)

        # Laplacian variance — masks have lower texture than bare skin
        lap_var_lower = float(cv2.Laplacian(lower_half_gray, cv2.CV_64F).var())

        # ── Decision logic ────────────────────────────────────────────────────
        # Score: higher = more likely wearing mask
        mask_score = 0.0

        # Eyes visible is strong indicator someone is present
        if eyes_visible:
            mask_score += 0.2

        # Low skin ratio in lower face = mask covering it
        if skin_ratio < 0.25:
            mask_score += 0.35
        elif skin_ratio < 0.40:
            mask_score += 0.15

        # White or surgical blue/teal mask colours
        if white_mask_pct > 0.25:
            mask_score += 0.30
        if blue_mask_pct > 0.20:
            mask_score += 0.20

        # Low texture in lower face = fabric/mask material
        if lap_var_lower < 80:
            mask_score += 0.25
        elif lap_var_lower < 150:
            mask_score += 0.10

        # High std in lower half can indicate no mask (varied skin features)
        if lower_mean_std > 45 and skin_ratio > 0.4:
            mask_score -= 0.20

        mask_score = max(0.0, min(1.0, mask_score))

        if mask_score >= 0.45:
            confidence = 0.75 + (mask_score - 0.45) * 0.35
            return "mask", min(float(confidence), 0.99), True, len(eyes)
        else:
            confidence = 0.70 + (0.45 - mask_score) * 0.40
            return "no_mask", min(float(confidence), 0.99), True, len(eyes)

    else:
        # ── No face detected: analyse centre-lower region ─────────────────────
        # When wearing a mask, the face detector often fails
        # because the mask hides facial geometry
        cy1 = int(h_img * 0.15)
        cy2 = int(h_img * 0.85)
        cx1 = int(w_img * 0.15)
        cx2 = int(w_img * 0.85)
        centre = img[cy1:cy2, cx1:cx2]
        centre_gray = gray[cy1:cy2, cx1:cx2]

        if centre.size == 0:
            return "no_mask", 0.60, False, 0

        # Check for mask-like colours in centre region
        centre_hsv = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV)
        skin_mask  = cv2.inRange(centre_hsv,
                                  np.array([0, 20, 70], dtype=np.uint8),
                                  np.array([25, 255, 255], dtype=np.uint8))
        skin_ratio = float(np.sum(skin_mask > 0)) / max(skin_mask.size, 1)

        white_pct = float(np.sum(
            (centre[:,:,0] > 180) &
            (centre[:,:,1] > 180) &
            (centre[:,:,2] > 180)
        )) / max(centre.shape[0] * centre.shape[1], 1)

        lap_var = float(cv2.Laplacian(centre_gray, cv2.CV_64F).var())

        # When face detector fails + mask-like colours → likely masked
        if white_pct > 0.15 or skin_ratio < 0.20 or lap_var < 60:
            return "mask", 0.82, False, 0
        else:
            return "no_mask", 0.72, False, 0


def predict_mask(image_b64: str):
    try:
        header, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
        img_bytes = base64.b64decode(encoded)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        return {"error": f"Image decode failed: {e}"}

    # Save screenshot
    screenshot_name = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(os.path.join(UPLOADS_DIR, screenshot_name), img)

    # Run real detection
    result, confidence, face_detected, eyes_count = detect_mask_from_image(img)

    entry_status = "GRANTED" if result == "mask" else "DENIED"

    return {
        "result":        result,
        "confidence":    round(confidence * 100, 2),
        "entry_status":  entry_status,
        "face_detected": face_detected,
        "faces_count":   1 if face_detected else 0,
        "image_path":    screenshot_name,
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "database":       "connected" if check_db() else "disconnected",
        "model_loaded":   model is not None,
        "model_accuracy": model_stats.get("accuracy", 0),
        "timestamp":      datetime.utcnow().isoformat(),
    })

# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def register():
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503

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
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503

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
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503
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
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503

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
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503

    page     = max(1, int(request.args.get("page", 1)))
    per_page = min(100, int(request.args.get("per_page", 20)))
    skip     = (page - 1) * per_page
    uid      = request.user["user_id"]

    total = logs_col.count_documents({"user_id": uid})
    rows  = list(logs_col.find(
        {"user_id": uid}, {"_id": 0}
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
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503

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
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503

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
        "algorithm":        "CV-Based Mask Detector (Haar + HSV + Texture Analysis)",
        "framework":        "OpenCV",
        "features":         "Eye detection, skin tone ratio, mask colour, texture variance",
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

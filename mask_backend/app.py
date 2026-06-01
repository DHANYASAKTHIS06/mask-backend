import os
import json
import uuid
import base64
import hashlib
import hmac
import time
import logging
import ssl
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
MONGO_URI    = os.environ.get("MONGO_URI", "")
mongo_client = None
users_col    = None
logs_col     = None

def try_connect(uri, **kwargs):
    """Try connecting with given kwargs, return client or None."""
    try:
        c = MongoClient(uri, serverSelectionTimeoutMS=15000,
                        connectTimeoutMS=15000, socketTimeoutMS=15000,
                        **kwargs)
        c.admin.command("ping")
        return c
    except Exception as e:
        logger.warning(f"Connection attempt failed: {e}")
        return None

def init_mongo():
    global mongo_client, users_col, logs_col

    if not MONGO_URI:
        logger.error("MONGO_URI not set")
        return

    # Build a standard (non-SRV) URI from the SRV URI
    # SRV: mongodb+srv://user:pass@cluster0.aeeqyj1.mongodb.net/db
    # Direct shards for aeeqyj1 cluster (Atlas standard port 27017)
    direct_hosts = (
        "ac-z6y3mgm-shard-00-00.aeeqyj1.mongodb.net:27017,"
        "ac-z6y3mgm-shard-00-01.aeeqyj1.mongodb.net:27017,"
        "ac-z6y3mgm-shard-00-02.aeeqyj1.mongodb.net:27017"
    )

    # Extract user:pass from SRV URI
    # mongodb+srv://user:pass@host/db?params
    try:
        without_scheme = MONGO_URI.replace("mongodb+srv://", "")
        credentials, rest = without_scheme.split("@", 1)
        _, db_and_params  = rest.split("/", 1) if "/" in rest else (rest, "maskguard")
        db_name = db_and_params.split("?")[0] or "maskguard"
    except Exception:
        credentials = ""
        db_name     = "maskguard"

    direct_uri = (
        f"mongodb://{credentials}@{direct_hosts}/{db_name}"
        f"?authSource=admin&replicaSet=atlas-xxxxxx-shard-0"
        f"&ssl=true&retryWrites=true&w=majority"
    )

    # ── Attempt 1: SRV URI + certifi ─────────────────────────────────────────
    logger.info("Attempt 1: SRV + certifi...")
    c = try_connect(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    if c:
        mongo_client = c
        logger.info("Connected via Attempt 1")

    # ── Attempt 2: SRV URI + tlsInsecure ────────────────────────────────────
    if not mongo_client:
        logger.info("Attempt 2: SRV + tlsInsecure...")
        c = try_connect(
            MONGO_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,
            tlsAllowInvalidHostnames=True,
        )
        if c:
            mongo_client = c
            logger.info("Connected via Attempt 2")

    # ── Attempt 3: SRV URI + custom ssl context ──────────────────────────────
    if not mongo_client:
        logger.info("Attempt 3: Custom SSL context...")
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.load_verify_locations(certifi.where())
            ctx.check_hostname  = False
            ctx.verify_mode     = ssl.CERT_NONE
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            c = try_connect(
                MONGO_URI,
                tls=True,
                tlsAllowInvalidCertificates=True,
                tlsAllowInvalidHostnames=True,
                ssl_context=ctx,
            )
            if c:
                mongo_client = c
                logger.info("Connected via Attempt 3")
        except Exception as e:
            logger.warning(f"Attempt 3 error: {e}")

    # ── Attempt 4: URI with tlsInsecure param ────────────────────────────────
    if not mongo_client:
        logger.info("Attempt 4: URI with tlsInsecure param...")
        insecure_uri = MONGO_URI
        if "?" in insecure_uri:
            insecure_uri += "&tlsInsecure=true&tlsAllowInvalidCertificates=true"
        else:
            insecure_uri += "?tlsInsecure=true&tlsAllowInvalidCertificates=true"
        c = try_connect(insecure_uri)
        if c:
            mongo_client = c
            logger.info("Connected via Attempt 4")

    # ── Attempt 5: No TLS params at all ─────────────────────────────────────
    if not mongo_client:
        logger.info("Attempt 5: Bare URI no TLS params...")
        c = try_connect(MONGO_URI)
        if c:
            mongo_client = c
            logger.info("Connected via Attempt 5")

    if mongo_client:
        db        = mongo_client["maskguard"]
        users_col = db["users"]
        logs_col  = db["detection_logs"]
        try:
            users_col.create_index("email", unique=True)
        except Exception:
            pass
        logger.info("MongoDB fully initialised.")
    else:
        logger.error("ALL MongoDB connection attempts failed.")

init_mongo()

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
    model       = joblib.load(os.path.join(MODELS_DIR, "mask_model.pkl"))
    scaler      = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "model_stats.json")) as f:
        model_stats = json.load(f)
    logger.info("ML models loaded.")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    model = scaler = None
    model_stats = {}

# Face & eye detectors
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ─── Auth Helpers ─────────────────────────────────────────────────────────────
def hash_password(p):
    salt = os.urandom(16)
    key  = hashlib.pbkdf2_hmac("sha256", p.encode(), salt, 100_000)
    return salt.hex() + ":" + key.hex()

def verify_password(p, stored):
    try:
        sh, kh = stored.split(":")
        return hmac.compare_digest(
            bytes.fromhex(kh),
            hashlib.pbkdf2_hmac("sha256", p.encode(), bytes.fromhex(sh), 100_000)
        )
    except Exception:
        return False

def create_token(uid, email):
    payload = {"user_id": uid, "email": email, "exp": time.time() + 86400 * 7}
    data = base64.b64encode(json.dumps(payload).encode()).decode()
    sig  = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    return f"{data}.{sig}"

def verify_token(token):
    try:
        data, sig = token.rsplit(".", 1)
        expected  = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.b64decode(data).decode())
        return None if payload["exp"] < time.time() else payload
    except Exception:
        return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token   = request.headers.get("Authorization","").replace("Bearer ","").strip()
        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        request.user = payload
        return f(*args, **kwargs)
    return decorated

# ─── Mask Detection ───────────────────────────────────────────────────────────
def analyse_region(bgr):
    if bgr is None or bgr.size == 0:
        return 0.5
    region = cv2.resize(bgr, (128, 128))
    gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv    = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(region)
    total   = region.shape[0] * region.shape[1]

    skin_pct = float(np.sum(cv2.inRange(hsv,
        np.array([0,15,60],  dtype=np.uint8),
        np.array([25,170,255],dtype=np.uint8)) > 0)) / total

    white_pct = float(np.sum((r>170)&(g>170)&(b>170))) / total
    blue_pct  = float(np.sum(
        (b.astype(int)-r.astype(int)>10) &
        (b.astype(int)-g.astype(int)>5)  & (b>120)
    )) / total
    dark_pct  = float(np.sum((r<60)&(g<60)&(b<60))) / total
    lap_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    texture   = min(lap_var / 300.0, 1.0)
    avg_std   = float((np.std(b)+np.std(g)+np.std(r)) / 3.0)
    uniformity = max(0.0, 1.0 - avg_std/60.0)

    score = 0.0
    if skin_pct < 0.10:   score += 0.40
    elif skin_pct < 0.20: score += 0.25
    elif skin_pct < 0.35: score += 0.10
    else:                  score -= 0.20
    score += min(white_pct*1.5, 0.30)
    score += min(blue_pct*2.0,  0.25)
    score += min(dark_pct*1.5,  0.20)
    if texture < 0.25:   score += 0.25
    elif texture < 0.50: score += 0.10
    else:                score -= 0.10
    score += uniformity * 0.20
    return float(np.clip(score, 0.0, 1.0))

def detect_mask(img):
    h, w   = img.shape[:2]
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    face_box = None

    for sf, mn, ms in [(1.1,5,(50,50)),(1.05,3,(30,30)),(1.15,4,(40,40)),(1.1,3,(25,25))]:
        faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=sf, minNeighbors=mn, minSize=ms)
        if len(faces) > 0:
            face_box = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            break

    if face_box is not None:
        fx, fy, fw, fh = face_box
        pad = int(fw*0.12)
        x1,y1 = max(0,fx-pad), max(0,fy-pad)
        x2,y2 = min(w,fx+fw+pad), min(h,fy+fh+pad)
        face_img  = img[y1:y2, x1:x2]
        face_gray = gray_eq[y1:y2, x1:x2]
        fh2 = y2-y1
        eyes = eye_cascade.detectMultiScale(
            face_gray[:fh2//2,:], scaleFactor=1.1, minNeighbors=2, minSize=(10,10)
        )
        lower = face_img[int(fh2*0.38):, :]
        score = analyse_region(lower)
        if not len(eyes):
            score = min(score+0.15, 1.0)
        if score >= 0.42:
            conf = 0.72 + (score-0.42)*0.40
            return "mask",    round(min(conf,0.99)*100,2), True
        else:
            conf = 0.70 + (0.42-score)*0.45
            return "no_mask", round(min(conf,0.99)*100,2), True
    else:
        cy1,cy2 = int(h*0.15), int(h*0.85)
        cx1,cx2 = int(w*0.15), int(w*0.85)
        score = analyse_region(img[cy1:cy2, cx1:cx2])
        if score >= 0.35:
            conf = 0.78 + (score-0.35)*0.30
            return "mask",    round(min(conf,0.99)*100,2), False
        else:
            conf = 0.68 + (0.35-score)*0.40
            return "no_mask", round(min(conf,0.99)*100,2), False

def predict_mask(image_b64):
    try:
        _, encoded = image_b64.split(",",1) if "," in image_b64 else ("", image_b64)
        img_bytes  = base64.b64decode(encoded)
        img        = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image")
    except Exception as e:
        return {"error": f"Image decode failed: {e}"}

    name = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(os.path.join(UPLOADS_DIR, name), img)
    result, confidence, face_detected = detect_mask(img)
    return {
        "result": result, "confidence": confidence,
        "entry_status": "GRANTED" if result=="mask" else "DENIED",
        "face_detected": face_detected,
        "faces_count": 1 if face_detected else 0,
        "image_path": name,
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "database": "connected" if check_db() else "disconnected",
        "timestamp": datetime.utcnow().isoformat(),
    })

@app.route("/api/auth/register", methods=["POST"])
def register():
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503
    data = request.get_json() or {}
    fn   = data.get("full_name","").strip()
    em   = data.get("email","").strip().lower()
    pw   = data.get("password","").strip()
    cpw  = data.get("confirm_password","").strip()
    if not all([fn,em,pw,cpw]):
        return jsonify({"error": "All fields are required"}), 400
    if pw != cpw:
        return jsonify({"error": "Passwords do not match"}), 400
    if len(pw) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    uid = str(uuid.uuid4())
    try:
        users_col.insert_one({
            "id": uid, "full_name": fn, "email": em,
            "password": hash_password(pw),
            "created_at": datetime.utcnow().isoformat(),
        })
    except DuplicateKeyError:
        return jsonify({"error": "Email already registered"}), 409
    return jsonify({
        "message": "Account created successfully",
        "token": create_token(uid, em),
        "user": {"id": uid, "full_name": fn, "email": em},
    }), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503
    data = request.get_json() or {}
    em   = data.get("email","").strip().lower()
    pw   = data.get("password","").strip()
    if not em or not pw:
        return jsonify({"error": "Email and password required"}), 400
    user = users_col.find_one({"email": em})
    if not user or not verify_password(pw, user["password"]):
        return jsonify({"error": "Invalid email or password"}), 401
    return jsonify({
        "message": "Login successful",
        "token": create_token(user["id"], em),
        "user": {
            "id": user["id"], "full_name": user["full_name"],
            "email": user["email"], "created_at": user["created_at"],
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
        "id": user["id"], "full_name": user["full_name"],
        "email": user["email"], "created_at": user["created_at"],
    })

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
    logs_col.insert_one({
        "id": log_id, "user_id": request.user["user_id"],
        "username": username, "timestamp": now,
        "result": result["result"], "confidence": result["confidence"],
        "entry_status": result["entry_status"],
        "image_path": result.get("image_path"),
        "faces_count": result.get("faces_count", 0),
    })
    return jsonify({**result, "log_id": log_id, "username": username, "timestamp": now})

@app.route("/api/logs", methods=["GET"])
@token_required
def get_logs():
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503
    page     = max(1, int(request.args.get("page",1)))
    per_page = min(100, int(request.args.get("per_page",20)))
    uid      = request.user["user_id"]
    total    = logs_col.count_documents({"user_id": uid})
    rows     = list(logs_col.find(
        {"user_id": uid},{"_id":0}
    ).sort("timestamp",-1).skip((page-1)*per_page).limit(per_page))
    return jsonify({
        "logs": rows, "total": total, "page": page,
        "per_page": per_page,
        "pages": max(1,(total+per_page-1)//per_page),
    })

@app.route("/api/logs/export", methods=["GET"])
@token_required
def export_csv():
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503
    import io, csv
    uid  = request.user["user_id"]
    rows = list(logs_col.find({"user_id":uid},{"_id":0}).sort("timestamp",-1))
    out  = io.StringIO()
    w    = csv.writer(out)
    w.writerow(["Log ID","Username","Timestamp","Result","Confidence (%)","Entry Status"])
    for r in rows:
        w.writerow([r.get("id"),r.get("username"),r.get("timestamp"),
                    r.get("result"),r.get("confidence"),r.get("entry_status")])
    out.seek(0)
    return app.response_class(
        out.read(), mimetype="text/csv",
        headers={"Content-Disposition":"attachment; filename=detection_logs.csv"},
    )

@app.route("/api/stats", methods=["GET"])
@token_required
def stats():
    if not check_db():
        return jsonify({"error": "Database unavailable"}), 503
    uid     = request.user["user_id"]
    total   = logs_col.count_documents({"user_id":uid})
    granted = logs_col.count_documents({"user_id":uid,"entry_status":"GRANTED"})
    denied  = logs_col.count_documents({"user_id":uid,"entry_status":"DENIED"})
    avg_r   = list(logs_col.aggregate([
        {"$match":{"user_id":uid}},
        {"$group":{"_id":None,"avg":{"$avg":"$confidence"}}}
    ]))
    avg_conf   = avg_r[0]["avg"] if avg_r else 0
    compliance = round(granted/total*100,1) if total else 0
    seven_ago  = (datetime.utcnow()-timedelta(days=7)).isoformat()
    trend = list(logs_col.aggregate([
        {"$match":{"user_id":uid,"timestamp":{"$gte":seven_ago}}},
        {"$group":{
            "_id":{"$substr":["$timestamp",0,10]},
            "count":{"$sum":1},
            "granted":{"$sum":{"$cond":[{"$eq":["$entry_status","GRANTED"]},1,0]}}
        }},
        {"$sort":{"_id":1}},
        {"$project":{"_id":0,"day":"$_id","count":1,"granted":1}}
    ]))
    return jsonify({
        "total_detections":total,"total_granted":granted,
        "total_denied":denied,"compliance_percentage":compliance,
        "avg_confidence":round(float(avg_conf),2),
        "model_accuracy":round(model_stats.get("accuracy",0)*100,2),
        "trend":trend,
    })

@app.route("/api/screenshot/<filename>", methods=["GET"])
@token_required
def get_screenshot(filename):
    path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="image/jpeg")

@app.route("/api/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "algorithm": "OpenCV HSV + Texture + Eye Detection",
        "classes": ["No Mask","Mask"],
        "accuracy": round(model_stats.get("accuracy",0)*100,2),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=False)

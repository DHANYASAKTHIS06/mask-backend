# 🎭 Smart Mask Detection — Backend API

Flask + scikit-learn backend for the AI-Based Smart Mask Verification and Entry Control System.

---

## 📁 Project Structure

```
mask_backend/
├── app.py              ← Main Flask application
├── train_model.py      ← Standalone model training script
├── requirements.txt    ← Python dependencies
├── gunicorn.conf.py    ← Gunicorn production config
├── render.yaml         ← Render.com one-click deploy config
├── build.sh            ← Build script (also used by Render)
├── df.csv              ← Medical Masks dataset (ADD THIS)
└── models/             ← Auto-generated after training
    ├── mask_model.pkl
    ├── scaler.pkl
    ├── gender_encoder.pkl
    └── model_stats.json
```

---

## 🚀 Deploy to Render

### One-Click (render.yaml)
1. Push this folder to a **GitHub repository**.
2. Add `df.csv` to the repository root.
3. Go to [render.com](https://render.com) → **New Web Service** → connect your repo.
4. Render auto-detects `render.yaml` and configures everything.
5. Click **Create Web Service** — done!

### Manual Setup on Render
| Field | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt && python train_model.py --csv df.csv` |
| **Start Command** | `gunicorn app:app -c gunicorn.conf.py` |
| **Environment Variable** | `SECRET_KEY` = (any random string) |

---

## 🔧 Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (put df.csv in this folder first)
python train_model.py

# 3. Start the dev server
python app.py
# OR with gunicorn:
gunicorn app:app --bind 0.0.0.0:5000 --reload
```

---

## 🤖 Machine Learning

| Property | Value |
|---|---|
| **Algorithm** | Gradient Boosting Classifier |
| **Framework** | scikit-learn 1.6.1 |
| **Training Samples** | 40,000 (Medical Masks Dataset) |
| **Test Accuracy** | **99.68%** |
| **Classes** | `no_mask` (TYPE=1) / `mask` (TYPE=2,3,4) |
| **Face Detector** | OpenCV Haar Cascade |

### Inference Pipeline
```
Base64 Image → Decode → Face Detection (OpenCV) →
Region Cropping → Pixel-Stat Features (20D) → StandardScaler →
GBM Predict → { result, confidence, entry_status }
```

---

## 📡 API Endpoints

Base URL (local): `http://localhost:5000`
Base URL (Render): `https://<your-service>.onrender.com`

### Authentication
All protected routes require:
```
Authorization: Bearer <token>
```

---

### `GET /api/health`
Health check. No auth required.
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_accuracy": 0.9968,
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### `POST /api/auth/register`
Create a new user account.

**Request:**
```json
{
  "full_name": "John Doe",
  "email": "john@example.com",
  "password": "secret123",
  "confirm_password": "secret123"
}
```
**Response (201):**
```json
{
  "message": "Account created successfully",
  "token": "<jwt_token>",
  "user": { "id": "...", "full_name": "John Doe", "email": "john@example.com" }
}
```

---

### `POST /api/auth/login`
**Request:**
```json
{ "email": "john@example.com", "password": "secret123" }
```
**Response (200):**
```json
{
  "message": "Login successful",
  "token": "<jwt_token>",
  "user": { "id": "...", "full_name": "John Doe", ... }
}
```

---

### `GET /api/auth/me` 🔒
Get current user profile.

---

### `POST /api/detect` 🔒
Run mask detection on a webcam frame.

**Request:**
```json
{ "image": "data:image/jpeg;base64,/9j/4AAQ..." }
```
**Response (200):**
```json
{
  "result": "mask",
  "confidence": 97.4,
  "entry_status": "GRANTED",
  "face_detected": true,
  "faces_count": 1,
  "log_id": "abc-123",
  "username": "John Doe",
  "timestamp": "2024-01-01T12:00:00"
}
```
- `result`: `"mask"` or `"no_mask"`
- `entry_status`: `"GRANTED"` or `"DENIED"`

---

### `GET /api/logs?page=1&per_page=20` 🔒
Paginated detection history for current user.

---

### `GET /api/logs/export` 🔒
Download detection history as CSV file.

---

### `GET /api/stats` 🔒
Dashboard statistics.
```json
{
  "total_detections": 50,
  "total_granted": 43,
  "total_denied": 7,
  "compliance_percentage": 86.0,
  "avg_confidence": 95.3,
  "model_accuracy": 99.68,
  "trend": [...]
}
```

---

### `GET /api/model/info`
Model metadata. No auth required.

---

### `GET /api/screenshot/<filename>` 🔒
Retrieve a saved screenshot image.

---

## 🔐 Security
- Passwords hashed with PBKDF2-SHA256 (100,000 rounds)
- JWT tokens signed with HMAC-SHA256
- Tokens expire after 7 days
- CORS enabled for all origins (configure for production)

---

## 📦 Dependencies
```
flask              — Web framework
flask-cors         — CORS headers
pandas             — CSV loading
numpy              — Numerical ops
scikit-learn==1.6.1— ML model
joblib             — Model serialisation
gunicorn           — Production WSGI server
opencv-python-headless — Face detection (no display needed)
```

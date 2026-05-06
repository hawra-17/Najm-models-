import cv2
import time
import requests
import os
from ultralytics import YOLO
from picamera2 import Picamera2
import pytesseract

# =========================
# SUPABASE CONFIG
# =========================
SUPABASE_URL = "https://uhsarpdtdbahjkcsmeqw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVoc2FycGR0ZGJhaGprY3NtZXF3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc3OTY3MzYsImV4cCI6MjA5MzM3MjczNn0.U_zHfzd4P5wLYiv8Cmu-p6_B7JpEbBjMhBl7MVci8Ng"
BUCKET_NAME = "accident-images"

CONFIDENCE_THRESHOLD = 0.5  # ✅ Confidence threshold

# =========================
# LOAD MODELS
# =========================
accident_model = YOLO("accident_yolov8n_best.pt")
plate_model = YOLO("plate_best.pt")
region_model = YOLO("region_best.pt")

print("Models loaded")

# =========================
# SUPABASE UPLOAD
# =========================
def upload_to_supabase(image_path, plate_text="unknown"):
    timestamp = int(time.time())
    clean_plate = plate_text.strip().replace(" ", "_").replace("\n", "") or "unknown"
    filename = f"{timestamp}_{clean_plate}.jpg"

    upload_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{filename}"

    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "image/jpeg",
    }

    with open(image_path, "rb") as f:
        response = requests.post(upload_url, headers=headers, data=f)

    if response.status_code in (200, 201):
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{filename}"
        print(f"✅ Image uploaded to Supabase: {public_url}")
        return public_url
    else:
        print(f"❌ Upload failed: {response.status_code} - {response.text}")
        return None

# =========================
# CAMERA
# =========================
def capture_image(filename):
    picam2 = Picamera2()
    picam2.start()
    time.sleep(2)

    picam2.capture_file(filename)

    picam2.stop()
    picam2.close()

    return filename

# =========================
# OCR
# =========================
def run_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

# =========================
# PIPELINE
# =========================
def detect_accident(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 384))

    accident = accident_model(img, verbose=False)
    if accident[0].boxes is None or len(accident[0].boxes) == 0:
        print("No accident ❌")
        return False

    accident_conf = accident[0].boxes.conf[0].item()
    print(f"Accident confidence: {accident_conf:.2f}")
    if accident_conf < CONFIDENCE_THRESHOLD:
        print("Accident confidence too low ❌")
        return False

    print("Accident detected ✅")
    return True


def detect_plate(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 384))

    # 1) Region detection
    region = region_model(img, verbose=False)
    if region[0].boxes is None or len(region[0].boxes) == 0:
        print("No region ❌")
        upload_to_supabase(path)
        return

    region_conf = region[0].boxes.conf[0].item()
    print(f"Region confidence: {region_conf:.2f}")
    if region_conf < CONFIDENCE_THRESHOLD:
        print("Region confidence too low ❌")
        upload_to_supabase(path)
        return

    rx1, ry1, rx2, ry2 = map(int, region[0].boxes.xyxy[0].cpu().numpy())
    region_crop = img[ry1:ry2, rx1:rx2]

    # 2) Plate detection inside region
    plate = plate_model(region_crop, verbose=False)
    if plate[0].boxes is None or len(plate[0].boxes) == 0:
        print("No plate ❌")
        upload_to_supabase(path)
        return

    plate_conf = plate[0].boxes.conf[0].item()
    print(f"Plate confidence: {plate_conf:.2f}")
    if plate_conf < CONFIDENCE_THRESHOLD:
        print("Plate confidence too low ❌")
        upload_to_supabase(path)
        return

    x1, y1, x2, y2 = map(int, plate[0].boxes.xyxy[0].cpu().numpy())
    plate_crop = region_crop[y1:y2, x1:x2]

    # 3) OCR
    text = run_ocr(plate_crop)
    print("Plate:", text)

    # 4) Upload
    upload_to_supabase(path, plate_text=text)

# =========================
# MAIN
# =========================
accident_img = capture_image("accident.jpg")
detect_accident(accident_img)

plate_img = capture_image("plate.jpg")
detect_plate(plate_img)
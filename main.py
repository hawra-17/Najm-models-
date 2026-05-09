# import cv2
# import time
# import requests
# import os
# from ultralytics import YOLO
# from picamera2 import Picamera2
# from libcamera import controls
# import pytesseract

# # =========================
# # SUPABASE CONFIG
# # =========================
# SUPABASE_URL = "https://uhsarpdtdbahjkcsmeqw.supabase.co"
# SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVoc2FycGR0ZGJhaGprY3NtZXF3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc3OTY3MzYsImV4cCI6MjA5MzM3MjczNn0.U_zHfzd4P5wLYiv8Cmu-p6_B7JpEbBjMhBl7MVci8Ng"
# BUCKET_NAME = "accident-images"

# CONFIDENCE_THRESHOLD = 0.5  # ✅ Confidence threshold

# # =========================
# # LOAD MODELS
# # =========================
# accident_model = YOLO("accident_yolov8n_best.pt")
# plate_model = YOLO("plate_best.pt")
# region_model = YOLO("region_best.pt")

# print("Models loaded")

# # =========================
# # SUPABASE UPLOAD
# # =========================
# def upload_to_supabase(image_path, plate_text="unknown"):
#     timestamp = int(time.time())
#     clean_plate = plate_text.strip().replace(" ", "_").replace("\n", "") or "unknown"
#     filename = f"{timestamp}_{clean_plate}.jpg"

#     upload_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{filename}"

#     headers = {
#         "Authorization": f"Bearer {SUPABASE_KEY}",
#         "Content-Type": "image/jpeg",
#     }

#     with open(image_path, "rb") as f:
#         response = requests.post(upload_url, headers=headers, data=f)

#     if response.status_code in (200, 201):
#         public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{filename}"
#         print(f"✅ Image uploaded to Supabase: {public_url}")
#         return public_url
#     else:
#         print(f"❌ Upload failed: {response.status_code} - {response.text}")
#         return None

# # =========================
# # CAMERA
# # =========================
# def capture_image():
#     picam2 = Picamera2()

#     # Full-res still config for IMX219 (Camera Module v2)
#     config = picam2.create_still_configuration(
#         main={"size": (3280, 2464), "format": "RGB888"}
#     )
#     picam2.configure(config)

#     # v2 has no autofocus — tune exposure/gain/AWB to reduce blur and noise
#     picam2.set_controls({
#         "AeEnable": True,
#         "AwbEnable": True,
#         "AwbMode": controls.AwbModeEnum.Daylight,
#         "AeExposureMode": controls.AeExposureModeEnum.Short,  # favor short exposure to freeze motion
#         "AnalogueGain": 1.0,
#         "Sharpness": 1.5,
#         "Contrast": 1.1,
#         "Saturation": 1.0,
#     })

#     picam2.start()
#     time.sleep(3)  # let AE/AWB settle

#     filename = "capture.jpg"
#     picam2.capture_file(filename)

#     picam2.stop()
#     picam2.close()

#     return filename

# # =========================
# # OCR
# # =========================
# def run_ocr(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return pytesseract.image_to_string(gray)

# # =========================
# # PIPELINE
# # =========================
# def run_pipeline(path):
#     img = cv2.imread(path)
#     img = cv2.resize(img, (640, 384))

#     # 1) Accident
#     accident = accident_model(img, verbose=False)
#     if accident[0].boxes is None or len(accident[0].boxes) == 0:
#         print("No accident ❌ — uploading anyway")
#         upload_to_supabase(path)
#         return

#     # ✅ Check accident confidence >= 0.5
#     accident_conf = accident[0].boxes.conf[0].item()
#     print(f"Accident confidence: {accident_conf:.2f}")
#     if accident_conf < CONFIDENCE_THRESHOLD:
#         print("Accident confidence too low ❌ — uploading anyway")
#         upload_to_supabase(path)
#         return

#     print("Accident detected ✅")

#     # 2) Plate
#     plate = plate_model(img, verbose=False)
#     if plate[0].boxes is None or len(plate[0].boxes) == 0:
#         print("No plate ❌")
#         upload_to_supabase(path)
#         return

#     # ✅ Check plate confidence >= 0.5
#     plate_conf = plate[0].boxes.conf[0].item()
#     print(f"Plate confidence: {plate_conf:.2f}")
#     if plate_conf < CONFIDENCE_THRESHOLD:
#         print("Plate confidence too low ❌")
#         upload_to_supabase(path)
#         return

#     box = plate[0].boxes.xyxy[0].cpu().numpy()
#     x1, y1, x2, y2 = map(int, box)
#     plate_crop = img[y1:y2, x1:x2]

#     # 3) OCR
#     text = run_ocr(plate_crop)
#     print("Plate:", text)

#     # 4) Upload to Supabase
#     upload_to_supabase(path, plate_text=text)

# # =========================
# # MAIN
# # =========================
# img = capture_image()
# run_pipeline(img)

import cv2
import time
import requests
import os
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import controls
import pytesseract

# =========================
# SUPABASE CONFIG
# =========================
SUPABASE_URL = "https://uhsarpdtdbahjkcsmeqw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVoc2FycGR0ZGJhaGprY3NtZXF3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc3OTY3MzYsImV4cCI6MjA5MzM3MjczNn0.U_zHfzd4P5wLYiv8Cmu-p6_B7JpEbBjMhBl7MVci8Ng"
BUCKET_NAME = "accident-images"

CONFIDENCE_THRESHOLD = 0.5

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
        print(f"✅ Uploaded: {public_url}")
        return public_url
    else:
        print(f"❌ Upload failed: {response.status_code} - {response.text}")
        return None

# =========================
# CAMERA
# =========================
def capture_image():
    picam2 = Picamera2()

    config = picam2.create_still_configuration(
        main={"size": (3280, 2464), "format": "RGB888"},
        controls={
            "AeEnable": True,
            "AwbEnable": True,
            "AwbMode": controls.AwbModeEnum.Daylight,
            "AeExposureMode": controls.AeExposureModeEnum.Short,
            "AnalogueGain": 1.0,       # Low noise
            "Sharpness": 8.0,          # ✅ Increased from 1.5 → much sharper
            "Contrast": 1.2,
            "Saturation": 1.0,
            "NoiseReductionMode": 2,   # ✅ High quality noise reduction
        }
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(3)

    filename = "capture.jpg"
    picam2.capture_file(filename)
    picam2.stop()
    picam2.close()

    return filename

# =========================
# SHARPEN IMAGE
# =========================
def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

# =========================
# OCR — Better preprocessing
# =========================
def run_ocr(plate_img):
    # Resize plate to be bigger → easier for OCR
    h, w = plate_img.shape[:2]
    scale = max(1, 200 // h)  # make height at least 200px
    plate_img = cv2.resize(plate_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Sharpen
    gray = cv2.filter2D(gray, -1, np.array([[0, -1, 0],
                                             [-1, 5, -1],
                                             [0, -1, 0]]))

    # Threshold → black & white for OCR
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save plate for debugging
    cv2.imwrite("plate_debug.jpg", thresh)

    # OCR config — treat as single line of text
    config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

# =========================
# PIPELINE
# ✅ Key fix: Don't shrink full image before detection
# =========================
def run_pipeline(path):
    full_img = cv2.imread(path)

    # ✅ Use medium resize for accident detection (keeps enough detail)
    accident_img = cv2.resize(full_img, (1280, 768))

    # 1) Accident detection
    accident = accident_model(accident_img, verbose=False)
    if accident[0].boxes is None or len(accident[0].boxes) == 0:
        print("No accident detected ❌ — uploading anyway")
        upload_to_supabase(path)
        return

    accident_conf = accident[0].boxes.conf[0].item()
    print(f"Accident confidence: {accident_conf:.2f}")
    if accident_conf < CONFIDENCE_THRESHOLD:
        print("Accident confidence too low ❌ — uploading anyway")
        upload_to_supabase(path)
        return

    print("Accident detected ✅")

    # ✅ Use full resolution for plate detection
    plate_img = sharpen(full_img)
    plate = plate_model(plate_img, verbose=False)

    if plate[0].boxes is None or len(plate[0].boxes) == 0:
        print("No plate detected ❌")
        upload_to_supabase(path)
        return

    plate_conf = plate[0].boxes.conf[0].item()
    print(f"Plate confidence: {plate_conf:.2f}")
    if plate_conf < CONFIDENCE_THRESHOLD:
        print("Plate confidence too low ❌")
        upload_to_supabase(path)
        return

    # ✅ Crop plate from full resolution image
    box = plate[0].boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    plate_crop = full_img[y1:y2, x1:x2]

    # Save cropped plate
    cv2.imwrite("plate_crop.jpg", plate_crop)

    # 3) OCR
    text = run_ocr(plate_crop)
    print(f"Plate text: '{text}'")

    # 4) Upload
    upload_to_supabase(path, plate_text=text)

# =========================
# MAIN
# =========================
img = capture_image()
run_pipeline(img)



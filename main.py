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

# import cv2
# import time
# import requests
# import os
# import numpy as np
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

# CONFIDENCE_THRESHOLD = 0.5

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
#         print(f"✅ Uploaded: {public_url}")
#         return public_url
#     else:
#         print(f"❌ Upload failed: {response.status_code} - {response.text}")
#         return None

# # =========================
# # CAMERA
# # =========================
# def capture_image():
#     picam2 = Picamera2()

#     config = picam2.create_still_configuration(
#         main={"size": (3280, 2464), "format": "RGB888"},
#         controls={
#             "AeEnable": True,
#             "AwbEnable": True,
#             "AwbMode": controls.AwbModeEnum.Daylight,
#             "AeExposureMode": controls.AeExposureModeEnum.Short,
#             "AnalogueGain": 1.0,       # Low noise
#             "Sharpness": 8.0,          # ✅ Increased from 1.5 → much sharper
#             "Contrast": 1.2,
#             "Saturation": 1.0,
#             "NoiseReductionMode": 2,   # ✅ High quality noise reduction
#         }
#     )
#     picam2.configure(config)
#     picam2.start()
#     time.sleep(3)

#     filename = "capture.jpg"
#     picam2.capture_file(filename)
#     picam2.stop()
#     picam2.close()

#     return filename

# # =========================
# # SHARPEN IMAGE
# # =========================
# def sharpen(img):
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     return cv2.filter2D(img, -1, kernel)

# # =========================
# # OCR — Better preprocessing
# # =========================
# def run_ocr(plate_img):
#     # Resize plate to be bigger → easier for OCR
#     h, w = plate_img.shape[:2]
#     scale = max(1, 200 // h)  # make height at least 200px
#     plate_img = cv2.resize(plate_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

#     # Convert to grayscale
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

#     # Denoise
#     gray = cv2.fastNlMeansDenoising(gray, h=10)

#     # Sharpen
#     gray = cv2.filter2D(gray, -1, np.array([[0, -1, 0],
#                                              [-1, 5, -1],
#                                              [0, -1, 0]]))

#     # Threshold → black & white for OCR
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Save plate for debugging
#     cv2.imwrite("plate_debug.jpg", thresh)

#     # OCR config — treat as single line of text
#     config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#     text = pytesseract.image_to_string(thresh, config=config)
#     return text.strip()

# # =========================
# # PIPELINE
# # ✅ Key fix: Don't shrink full image before detection
# # =========================
# def run_pipeline(path):
#     full_img = cv2.imread(path)

#     # ✅ Use medium resize for accident detection (keeps enough detail)
#     accident_img = cv2.resize(full_img, (1280, 768))

#     # 1) Accident detection
#     accident = accident_model(accident_img, verbose=False)
#     if accident[0].boxes is None or len(accident[0].boxes) == 0:
#         print("No accident detected ❌ — uploading anyway")
#         upload_to_supabase(path)
#         return

#     accident_conf = accident[0].boxes.conf[0].item()
#     print(f"Accident confidence: {accident_conf:.2f}")
#     if accident_conf < CONFIDENCE_THRESHOLD:
#         print("Accident confidence too low ❌ — uploading anyway")
#         upload_to_supabase(path)
#         return

#     print("Accident detected ✅")

#     # ✅ Use full resolution for plate detection
#     plate_img = sharpen(full_img)
#     plate = plate_model(plate_img, verbose=False)

#     if plate[0].boxes is None or len(plate[0].boxes) == 0:
#         print("No plate detected ❌")
#         upload_to_supabase(path)
#         return

#     plate_conf = plate[0].boxes.conf[0].item()
#     print(f"Plate confidence: {plate_conf:.2f}")
#     if plate_conf < CONFIDENCE_THRESHOLD:
#         print("Plate confidence too low ❌")
#         upload_to_supabase(path)
#         return

#     # ✅ Crop plate from full resolution image
#     box = plate[0].boxes.xyxy[0].cpu().numpy()
#     x1, y1, x2, y2 = map(int, box)
#     plate_crop = full_img[y1:y2, x1:x2]

#     # Save cropped plate
#     cv2.imwrite("plate_crop.jpg", plate_crop)

#     # 3) OCR
#     text = run_ocr(plate_crop)
#     print(f"Plate text: '{text}'")

#     # 4) Upload
#     upload_to_supabase(path, plate_text=text)

# # =========================
# # MAIN
# # =========================
# img = capture_image()
# run_pipeline(img)


import cv2
import time
import requests
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import pytesseract

# =========================
# SUPABASE CONFIG
# =========================
SUPABASE_URL = "https://uhsarpdtdbahjkcsmeqw.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"
BUCKET_NAME = "accident-images"

CONFIDENCE_THRESHOLD = 0.5

# =========================
# LOAD MODELS
# =========================
accident_model = YOLO("accident_yolov8n_best.pt")
plate_model = YOLO("plate_best.pt")

print("✅ Models loaded")

# =========================
# SUPABASE UPLOAD
# =========================
def upload_to_supabase(image_path, plate_text="unknown"):
    timestamp = int(time.time())

    clean_plate = (
        plate_text.strip()
        .replace(" ", "_")
        .replace("\n", "")
    )

    if clean_plate == "":
        clean_plate = "unknown"

    filename = f"{timestamp}_{clean_plate}.jpg"

    upload_url = (
        f"{SUPABASE_URL}/storage/v1/object/"
        f"{BUCKET_NAME}/{filename}"
    )

    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "image/jpeg",
    }

    with open(image_path, "rb") as f:
        response = requests.post(
            upload_url,
            headers=headers,
            data=f
        )

    if response.status_code in [200, 201]:
        public_url = (
            f"{SUPABASE_URL}/storage/v1/object/public/"
            f"{BUCKET_NAME}/{filename}"
        )

        print("✅ Uploaded to Supabase")
        print(public_url)

        return public_url

    else:
        print("❌ Upload failed")
        print(response.text)

        return None

# =========================
# CAMERA CAPTURE
# =========================
def capture_image():

    picam2 = Picamera2()

    config = picam2.create_still_configuration(
        main={
            "size": (3280, 2464),
            "format": "RGB888"
        }
    )

    picam2.configure(config)

    # Better controls for moving cars
    picam2.set_controls({

        # Manual exposure
        "AeEnable": False,
        "ExposureTime": 5000,
        "AnalogueGain": 1.5,

        # White balance
        "AwbEnable": True,

        # Image tuning
        "Sharpness": 2.0,
        "Contrast": 1.2,
        "Saturation": 1.0,
    })

    picam2.start()

    # Camera warm-up
    time.sleep(2)

    filename = "capture.jpg"

    picam2.capture_file(filename)

    picam2.stop()
    picam2.close()

    print("✅ Image captured")

    return filename

# =========================
# OCR PREPROCESSING
# =========================
def preprocess_plate(plate_crop):

    gray = cv2.cvtColor(
        plate_crop,
        cv2.COLOR_BGR2GRAY
    )

    # Upscale image
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Reduce noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Sharpen
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    gray = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh

# =========================
# OCR
# =========================
def run_ocr(plate_crop):

    processed = preprocess_plate(plate_crop)

    text = pytesseract.image_to_string(
        processed,
        config="--psm 7"
    )

    return text.strip()

# =========================
# DRAW RESULTS
# =========================
def draw_box(img, box, label):

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        img,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(path):

    upload_path = path
    best_plate_text = "unknown"

    img = cv2.imread(path)

    if img is None:
        print("❌ Failed to load image — uploading raw capture anyway")
        upload_to_supabase(path)
        return

    try:
        print("✅ Running accident detection")

        # =========================
        # ACCIDENT DETECTION
        # =========================
        accident_results = accident_model(img, verbose=False)

        accident_detected = False
        if (
            accident_results[0].boxes is not None
            and len(accident_results[0].boxes) > 0
        ):
            accident_conf = accident_results[0].boxes.conf[0].item()
            print(f"Accident confidence: {accident_conf:.2f}")

            if accident_conf >= CONFIDENCE_THRESHOLD:
                print("✅ Accident detected")
                accident_detected = True

                accident_box = (
                    accident_results[0].boxes.xyxy[0].cpu().numpy()
                )
                draw_box(
                    img,
                    accident_box,
                    f"Accident {accident_conf:.2f}"
                )
            else:
                print("❌ Accident confidence too low")
        else:
            print("❌ No accident detected")

        # =========================
        # PLATE DETECTION (runs regardless of accident result)
        # =========================
        print("✅ Running plate detection")
        plate_results = plate_model(img, verbose=False)

        if (
            plate_results[0].boxes is not None
            and len(plate_results[0].boxes) > 0
        ):
            for i, box in enumerate(plate_results[0].boxes.xyxy):
                confidence = plate_results[0].boxes.conf[i].item()
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                plate_crop = img[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                plate_text = run_ocr(plate_crop)
                print(f"✅ Plate Detected: {plate_text}")
                best_plate_text = plate_text

                draw_box(img, [x1, y1, x2, y2], plate_text)
                cv2.imwrite(f"plate_{i}.jpg", plate_crop)
        else:
            print("❌ No plate detected")

        # If we drew anything on the image, upload the annotated version.
        # Otherwise upload the raw capture.
        if accident_detected or best_plate_text != "unknown":
            result_path = "result.jpg"
            cv2.imwrite(result_path, img)
            print("✅ Result image saved")
            upload_path = result_path

    except Exception as e:
        print(f"⚠️ Pipeline error: {e} — uploading raw capture")
        upload_path = path

    # =========================
    # ALWAYS UPLOAD
    # =========================
    upload_to_supabase(upload_path, best_plate_text)

# =========================
# START
# =========================
if __name__ == "__main__":

    image_path = capture_image()

    run_pipeline(image_path)
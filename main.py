import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
import pytesseract

# =========================
# LOAD MODELS
# =========================
accident_model = YOLO("accident_yolov8n_best.pt")
plate_model = YOLO("plate_best.pt")
region_model = YOLO("region_best.pt")

print("Models loaded")

# =========================
# CAMERA
# =========================
def capture_image():
    picam2 = Picamera2()
    picam2.start()
    time.sleep(2)

    filename = "capture.jpg"
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
def run_pipeline(path):
    img = cv2.imread(path)

    # تصغير الصورة (مهم)
    img = cv2.resize(img, (640, 384))

    # 1) Accident
    accident = accident_model(img, verbose=False)
    if accident[0].boxes is None:
        print("No accident ❌")
        return

    print("Accident detected ✅")

    # 2) Plate
    plate = plate_model(img, verbose=False)

    if plate[0].boxes is None:
        print("No plate ❌")
        return

    box = plate[0].boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)

    plate_crop = img[y1:y2, x1:x2]

    # 3) OCR
    text = run_ocr(plate_crop)

    print("Plate:", text)

# =========================
# MAIN
# =========================
img = capture_image()
run_pipeline(img)
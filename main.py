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
import subprocess
import numpy as np
from ultralytics import YOLO
import pytesseract

# =========================
# SUPABASE CONFIG
# =========================
SUPABASE_URL = "https://mncsiqxctyqdrfqtnzha.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1uY3NpcXhjdHlxZHJmcXRuemhhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA0NDk3NjIsImV4cCI6MjA4NjAyNTc2Mn0.bP-0G6IAjAr-731aOEJSPkNNA_pwomOvSwUmipxKjP8"
BUCKET_NAME = "accident-images"

CONFIDENCE_THRESHOLD = 0.5

# Stop recording if no plate is found within this many seconds (safeguard)
MAX_RECORD_SECONDS = 60

# Video capture settings (from the rpicam-vid command)
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FRAMERATE = 30
DETECT_FPS = 5  # how many frames/sec to pull out for YOLO detection

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
def upload_file(local_path, dest_name, content_type):
    """Upload any file (image or video) to the Supabase bucket."""
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{dest_name}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": content_type,
    }

    with open(local_path, "rb") as f:
        response = requests.post(upload_url, headers=headers, data=f)

    if response.status_code in (200, 201):
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{dest_name}"
        print(f"✅ Uploaded: {public_url}")
        return public_url
    else:
        print(f"❌ Upload failed ({dest_name}): {response.status_code} - {response.text}")
        return None


def upload_event(video_path, accident_path, plate_path, plate_text="unknown"):
    """Upload the video + accident frame + plate crop for one detected event."""
    timestamp = int(time.time())
    clean_plate = plate_text.strip().replace(" ", "_").replace("\n", "") or "unknown"
    base = f"{timestamp}_{clean_plate}"

    results = {}
    if video_path and os.path.exists(video_path):
        results["video"] = upload_file(video_path, f"{base}.mp4", "video/mp4")
    if accident_path and os.path.exists(accident_path):
        results["accident"] = upload_file(accident_path, f"{base}_accident.jpg", "image/jpeg")
    if plate_path and os.path.exists(plate_path):
        results["plate"] = upload_file(plate_path, f"{base}_plate.jpg", "image/jpeg")
    return results

# =========================
# RECORDER — rpicam-vid records to video.mp4 while ffmpeg also
# pipes live MJPEG frames to us for YOLO detection.
# =========================
class VideoRecorder:
    """
    Pipeline:
        rpicam-vid (H264 to stdout)
            -> ffmpeg
                 -> video.mp4              (fragmented, safe to kill anytime)
                 -> MJPEG frames to stdout (decoded here for detection)

    The video keeps recording while we grab frames — nothing stops the
    recording until we explicitly call .stop().
    """

    def __init__(self, video_path="video.mp4"):
        self.video_path = video_path
        self.rpicam = None
        self.ffmpeg = None
        self._buf = b""
        # Errors from the subprocesses go here instead of being hidden,
        # so we can see WHY no frames arrive.
        self._rpicam_log = open("rpicam.log", "wb")
        self._ffmpeg_log = open("ffmpeg.log", "wb")

    def start(self):
        if os.path.exists(self.video_path):
            os.remove(self.video_path)

        # rpicam-vid: continuous (-t 0), H264 to stdout. --inline keeps
        # SPS/PPS headers in-stream so ffmpeg can start decoding immediately.
        # start_new_session=True puts the children in their OWN process
        # group, so Ctrl+C (SIGINT) does NOT get delivered to them — we
        # control their shutdown explicitly in stop().
        self.rpicam = subprocess.Popen(
            [
                "rpicam-vid",
                "-t", "0",
                "--width", str(VIDEO_WIDTH),
                "--height", str(VIDEO_HEIGHT),
                "--framerate", str(VIDEO_FRAMERATE),
                "--codec", "h264",
                "--inline",
                "--nopreview",
                "-o", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=self._rpicam_log,
            start_new_session=True,
        )

        # ffmpeg: one input, two outputs.
        #  1) copy H264 into a fragmented MP4 (playable even if killed)
        #  2) MJPEG frames at DETECT_FPS to stdout for detection
        # "-f h264" tells ffmpeg the pipe is a raw H264 stream so it does
        # not stall probing an unknown non-seekable input.
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel", "warning",
                "-fflags", "+genpts",
                "-f", "h264",
                "-i", "pipe:0",
                # Output 1 — the saved video
                "-map", "0:v", "-c:v", "copy",
                "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                "-f", "mp4", self.video_path,
                # Output 2 — detection frames
                "-map", "0:v", "-vf", f"fps={DETECT_FPS}",
                "-c:v", "mjpeg", "-q:v", "3",
                "-flush_packets", "1", "-f", "image2pipe", "pipe:1",
            ],
            stdin=self.rpicam.stdout,
            stdout=subprocess.PIPE,
            stderr=self._ffmpeg_log,
            start_new_session=True,
        )
        # Let the parent close its handle so ffmpeg gets EOF if rpicam dies.
        self.rpicam.stdout.close()
        print("🎥 Recording started (rpicam-vid -> ffmpeg). "
              "Logs: rpicam.log / ffmpeg.log")

    def read_frame(self, timeout=15):
        """
        Return the next decoded BGR frame.

        Returns None if the stream ended OR no data arrived within
        `timeout` seconds (so the loop never blocks forever).
        """
        import select

        SOI = b"\xff\xd8"  # JPEG start of image
        EOI = b"\xff\xd9"  # JPEG end of image
        while True:
            start = self._buf.find(SOI)
            end = self._buf.find(EOI, start + 2) if start != -1 else -1
            if start != -1 and end != -1:
                jpg = self._buf[start:end + 2]
                self._buf = self._buf[end + 2:]
                return cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                    cv2.IMREAD_COLOR)

            # Wait up to `timeout`s for data; keeps Ctrl+C responsive
            # and surfaces a stuck pipeline instead of hanging.
            ready, _, _ = select.select([self.ffmpeg.stdout], [], [], timeout)
            if not ready:
                if self.rpicam.poll() is not None:
                    print("❌ rpicam-vid exited — see rpicam.log")
                if self.ffmpeg.poll() is not None:
                    print("❌ ffmpeg exited — see ffmpeg.log")
                print(f"⚠️  No video frames for {timeout}s — check the logs")
                return None

            chunk = self.ffmpeg.stdout.read(65536)
            if not chunk:
                return None
            self._buf += chunk

    def stop(self):
        """Stop recording cleanly so the MP4 finalizes."""
        import signal

        for proc in (self.ffmpeg, self.rpicam):
            if proc and proc.poll() is None:
                try:
                    # Kill the whole child process group.
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        proc.kill()
        for f in (self._rpicam_log, self._ffmpeg_log):
            try:
                f.close()
            except Exception:
                pass
        print(f"🛑 Recording stopped → {self.video_path}")

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
# PIPELINE — runs on the LIVE video stream
#
#   1) video starts recording
#   2) on accident car detected -> grab a frame (video keeps recording)
#   3) on license plate detected -> grab another frame, crop the
#      plate region, OCR it, stop recording
#   4) upload picture(s) + video to Supabase
# =========================
def run_pipeline():
    recorder = VideoRecorder("video.mp4")
    recorder.start()

    accident_detected = False
    accident_frame_path = None
    plate_crop_path = None
    plate_text = "unknown"

    frame_count = 0
    start_time = time.time()
    try:
        while True:
            if time.time() - start_time > MAX_RECORD_SECONDS:
                print(f"⏱️  No plate within {MAX_RECORD_SECONDS}s — stopping")
                break

            frame = recorder.read_frame()
            if frame is None:
                print("⚠️  No more frames — stopping (check rpicam.log / "
                      "ffmpeg.log if this was immediate)")
                break

            frame_count += 1
            # Heartbeat: proves frames are flowing and detection is running.
            if frame_count == 1:
                print(f"✅ First frame received ({frame.shape[1]}x"
                      f"{frame.shape[0]}) — detection is live")
            elif frame_count % 10 == 0:
                state = "looking for plate" if accident_detected \
                    else "looking for accident"
                print(f"… {frame_count} frames processed ({state})")

            # ---- Step 2: accident detection (until one is found) ----
            if not accident_detected:
                det = accident_model(
                    cv2.resize(frame, (1280, 768)), verbose=False
                )
                boxes = det[0].boxes
                if boxes is None or len(boxes) == 0:
                    continue
                conf = boxes.conf[0].item()
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                accident_detected = True
                accident_frame_path = "accident_frame.jpg"
                cv2.imwrite(accident_frame_path, frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"💥 Accident detected ({conf:.2f}) — frame saved, "
                      f"video still recording")
                continue

            # ---- Step 3: plate detection (after an accident) ----
            plate = plate_model(sharpen(frame), verbose=False)
            pboxes = plate[0].boxes
            if pboxes is None or len(pboxes) == 0:
                continue
            pconf = pboxes.conf[0].item()
            if pconf < CONFIDENCE_THRESHOLD:
                continue

            print(f"🚗 Plate detected ({pconf:.2f}) — grabbing frame")

            # Save the full plate frame
            cv2.imwrite("plate_frame.jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Crop the plate region from this frame
            x1, y1, x2, y2 = map(int, pboxes.xyxy[0].cpu().numpy())
            plate_crop = frame[y1:y2, x1:x2]

            # Optionally refine to the inner region with region_model
            try:
                rdet = region_model(plate_crop, verbose=False)
                rboxes = rdet[0].boxes
                if rboxes is not None and len(rboxes) > 0:
                    rx1, ry1, rx2, ry2 = map(
                        int, rboxes.xyxy[0].cpu().numpy())
                    region = plate_crop[ry1:ry2, rx1:rx2]
                    if region.size > 0:
                        plate_crop = region
            except Exception as e:
                print(f"region_model skipped: {e}")

            plate_crop_path = "plate_crop.jpg"
            cv2.imwrite(plate_crop_path, plate_crop)

            plate_text = run_ocr(plate_crop)
            print(f"🔎 Plate text: '{plate_text}'")
            break
    except KeyboardInterrupt:
        print("\n⏹️  Ctrl+C — stopping recording and exiting")
    finally:
        # Step 4 prerequisite: stop recording so the MP4 finalizes
        recorder.stop()

    if not accident_detected:
        print("No accident detected — nothing uploaded")
        return

    # ---- Step 4: upload picture(s) + video ----
    upload_event(
        video_path=recorder.video_path,
        accident_path=accident_frame_path,
        plate_path=plate_crop_path,
        plate_text=plate_text,
    )

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run_pipeline()



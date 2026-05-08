import io
import socket
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
from libcamera import controls

PAGE = """\
<!DOCTYPE html>
<html>
<head><title>Camera Preview</title></head>
<body style="margin:0;background:#000;text-align:center;">
  <img src="stream.mjpg" style="max-width:100%;height:auto;" />
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.end_headers()
        elif self.path == "/index.html":
            content = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
            )
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception as e:
                print(f"Client disconnected: {e}")
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "<pi-ip>"


picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.set_controls({
    "AeEnable": True,
    "AwbEnable": True,
    "AwbMode": controls.AwbModeEnum.Daylight,
})

output = StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(output))

try:
    address = ("", 8000)
    httpd = StreamingServer(address, StreamingHandler)
    print(f"Open in your Mac browser: http://{get_ip()}:8000")
    print("Ctrl+C to stop.")
    httpd.serve_forever()
finally:
    picam2.stop_recording()

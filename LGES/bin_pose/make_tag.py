"""Printable AprilTag 36h11 id 0 for the wrist-camera calibration: black square
TAG_MM on A4 at 300 dpi, with a 100 mm scale bar to check the print scale.
Print at 100% / "actual size" (no fit-to-page), then MEASURE the black square."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TAG_MM, TAG_ID, DPI = 150, 0, 300
px = lambda mm: int(round(mm / 25.4 * DPI))
page = np.full((px(297), px(210)), 255, np.uint8)
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
tag = cv2.aruco.generateImageMarker(d, TAG_ID, px(TAG_MM), borderBits=1)
y0, x0 = px(40), (page.shape[1] - tag.shape[1]) // 2
page[y0:y0 + tag.shape[0], x0:x0 + tag.shape[1]] = tag
img = Image.fromarray(page); dr = ImageDraw.Draw(img)
try:
    f = ImageFont.truetype("DejaVuSans.ttf", px(5))
except OSError:
    f = ImageFont.load_default()
yb = y0 + tag.shape[0] + px(20)
dr.rectangle([px(55), yb, px(55) + px(100), yb + px(3)], fill=0)
dr.text((px(55), yb + px(6)), "100 mm scale bar - check after printing", fill=0, font=f)
dr.text((px(20), yb + px(20)), f"AprilTag 36h11  id {TAG_ID}  black square {TAG_MM} mm", fill=0, font=f)
dr.text((px(20), yb + px(30)), "Print at 100% (actual size). Keep flat (tape to a board).", fill=0, font=f)
img.save(f"apriltag_36h11_id{TAG_ID}_{TAG_MM}mm_A4.pdf", resolution=DPI)
img.save(f"apriltag_36h11_id{TAG_ID}_{TAG_MM}mm_A4.png", dpi=(DPI, DPI))
# self-check: the generated image must detect as id 0
det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
small = cv2.resize(page, None, fx=0.25, fy=0.25)
corners, ids, _ = det.detectMarkers(small)
side = np.linalg.norm(corners[0][0][0] - corners[0][0][1]) * 4 / DPI * 25.4 if ids is not None else None
print("detected ids", None if ids is None else ids.ravel().tolist(), "black square on page", None if side is None else f"{side:.1f} mm")

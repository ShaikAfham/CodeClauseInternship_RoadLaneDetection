# src/debug_shapes.py
import cv2
import sys
from src.pipeline import LaneProcessor

cap = cv2.VideoCapture("data/road.mp4")
ret, frame = cap.read()
print("Got frame:", ret)
if not ret:
    sys.exit("Can't read frame or file not found")

print("Frame shape:", frame.shape, "dtype:", frame.dtype)

lp = LaneProcessor()
out, info = lp.process_frame(frame, use_warp=True)

print("Returned out type:", type(out))
if hasattr(out, "shape"):
    print("Out shape:", out.shape, "dtype:", out.dtype)
else:
    print("Out is not an array")

if info is None:
    print("Info is None (no detection)")
else:
    keys = list(info.keys())
    print("Info keys:", keys)
    if 'leftx' in info:
        print("leftx len:", len(info['leftx']), "sample:", info['leftx'][:5])
    if 'rightx' in info:
        print("rightx len:", len(info['rightx']), "sample:", info['rightx'][:5])
    if 'ploty' in info:
        print("ploty len:", len(info['ploty']), "sample:", info['ploty'][:5])
cap.release()

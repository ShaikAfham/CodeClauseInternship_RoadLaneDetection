#!/usr/bin/env python3
"""
Main CLI entrypoint for Road Lane Detection project.

Usage examples:
# Process an image and save output
python main.py --mode image --input ../data/test_image.jpg --output ../data/test_image_out.jpg

# Process a video and save output + csv log
python main.py --mode video --input ../data/road.mp4 --output ../data/road_out.mp4 --log ../data/road_log.csv

# Run webcam (press 'q' in the OpenCV window to stop)
python main.py --mode webcam --input 0
"""
import argparse
import os
import sys
from pathlib import Path

# ensure package imports work when running main.py from src/
sys.path.append(os.path.dirname(__file__))

import cv2
import numpy as np
from pipeline import LaneProcessor
from video_processor import VideoProcessor
from road_classifier import classify_road_type

def process_image(input_path, output_path=None, use_warp=True, show=True):
    img = cv2.imread(input_path)
    if img is None:
        print(f"[ERROR] Could not read image: {input_path}")
        return False
    lp = LaneProcessor()
    out, info = lp.process_frame(img, use_warp=use_warp)
    road = classify_road_type(img)
    cv2.putText(out, f"Road: {road}", (30,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    if output_path:
        cv2.imwrite(output_path, out)
        print(f"[INFO] Saved output image to: {output_path}")
    if show:
        cv2.imshow("Lane Detection - Image", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return True

def process_video(input_path, output_path=None, log_path=None, use_warp=True, show=True):
    vp = VideoProcessor(input_path, output_path=output_path, log_csv=log_path, use_warp=use_warp)
    vp.start(show=show)
    print("[INFO] Video processing finished.")
    if output_path:
        print(f"[INFO] Output saved to: {output_path}")
    if log_path:
        print(f"[INFO] CSV log saved to: {log_path}")

def main():
    p = argparse.ArgumentParser(description="Road Lane Detection - CLI")
    p.add_argument("--mode", required=True, choices=["image", "video", "webcam"], help="Run mode")
    p.add_argument("--input", required=True, help="Input path. For webcam use 0 (or '0').")
    p.add_argument("--output", required=False, help="Output file path (image or video).")
    p.add_argument("--log", required=False, help="Optional CSV log path for video mode.")
    p.add_argument("--no-show", action="store_true", help="Don't show OpenCV windows (useful for server).")
    p.add_argument("--no-warp", action="store_true", help="Disable perspective warp (useful for quick tests).")
    args = p.parse_args()

    mode = args.mode
    input_path = args.input
    output_path = args.output
    log_path = args.log
    show = not args.no_show
    use_warp = not args.no_warp

    # Normalize paths if relative
    base = Path(__file__).parent
    def norm(pth):
        if pth is None:
            return None
        s = str(pth)
        # webcam numeric
        if s == "0":
            return "0"
        path = (Path.cwd() / s) if not Path(s).is_absolute() else Path(s)
        return str(path)

    input_path = norm(input_path)
    output_path = norm(output_path)
    log_path = norm(log_path)

    # Run appropriate mode
    try:
        if mode == "image":
            if not input_path or not Path(input_path).exists():
                print(f"[ERROR] Image not found: {input_path}")
                return
            if output_path is None:
                # default output filename
                out_p = str(Path(input_path).parent / (Path(input_path).stem + "_out" + Path(input_path).suffix))
            else:
                out_p = output_path
            process_image(input_path, out_p, use_warp=use_warp, show=show)
        elif mode == "video":
            if input_path == "0":
                print("[ERROR] For webcam as video mode use --mode webcam --input 0")
                return
            if not Path(input_path).exists():
                print(f"[ERROR] Video not found: {input_path}")
                return
            if output_path is None:
                out_p = str(Path(input_path).parent / (Path(input_path).stem + "_out.mp4"))
            else:
                out_p = output_path
            process_video(input_path, output_path=out_p, log_path=log_path, use_warp=use_warp, show=show)
        elif mode == "webcam":
            # For webcam, input may be '0' or a device index
            dev = input_path if input_path is not None else "0"
            process_video(dev, output_path=output_path, log_path=log_path, use_warp=use_warp, show=show)
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Exception: {e}")

if __name__ == "__main__":
    main()


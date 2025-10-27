#!/usr/bin/env python3
"""
Main CLI entrypoint for Road Lane Detection project.

Usage examples:
python src/main.py --mode image --input ../data/test_image.jpg --output ../data/test_image_out.jpg
python src/main.py --mode video --input ../data/road.mp4 --output ../data/road_out.mp4 --log ../data/road_log.csv --ldw --beep
python src/main.py --mode webcam --input 0 --ldw
"""
import argparse
import os
import sys
from pathlib import Path

# ensure package imports work when running main.py from src/
sys.path.append(os.path.dirname(__file__))

import cv2
# ensure package imports work when running main.py as a module
# Use absolute imports referencing the package 'src'
from src.pipeline import LaneProcessor
from src.video_processor import VideoProcessor
from src.road_classifier import classify_road_type
import cv2

def process_image(input_path, output_path=None, use_warp=True, show=True, enable_ldw=False, ldw_threshold=0.4, beep_on_ldw=False):
    img = cv2.imread(input_path)
    if img is None:
        print(f"[ERROR] Could not read image: {input_path}")
        return False
    lp = LaneProcessor(enable_ldw=enable_ldw, ldw_threshold_m=ldw_threshold, beep_on_ldw=beep_on_ldw)
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

def process_video(input_path, output_path=None, log_path=None, use_warp=True, enable_ldw=False, ldw_threshold=0.4, beep_on_ldw=False, show=True):
    vp = VideoProcessor(input_path, output_path=output_path, log_csv=log_path, use_warp=use_warp,
                        enable_ldw=enable_ldw, ldw_threshold=ldw_threshold, beep_on_ldw=beep_on_ldw)
    vp.start(show=show)
    print("[INFO] Video processing finished.")
    if output_path:
        print(f"[INFO] Output saved to: {output_path}")
    if log_path:
        print(f"[INFO] CSV log saved to: {log_path}")

def main():
    p = argparse.ArgumentParser(description="Road Lane Detection - CLI with LDW")
    p.add_argument("--mode", required=True, choices=["image", "video", "webcam"], help="Run mode")
    p.add_argument("--input", required=True, help="Input path. For webcam use 0 (or '0').")
    p.add_argument("--output", required=False, help="Output file path (image or video).")
    p.add_argument("--log", required=False, help="Optional CSV log path for video mode.")
    p.add_argument("--no-show", action="store_true", help="Don't show OpenCV windows (useful for server).")
    p.add_argument("--no-warp", action="store_true", help="Disable perspective warp (useful for quick tests).")
    # LDW flags
    p.add_argument("--ldw", action="store_true", help="Enable Lane Departure Warning (visual + optional beep).")
    p.add_argument("--ldw-threshold", type=float, default=0.4, help="LDW threshold in meters (default 0.4).")
    p.add_argument("--beep", action="store_true", help="Play beep on LDW (requires simpleaudio).")
    args = p.parse_args()

    mode = args.mode
    input_path = args.input
    output_path = args.output
    log_path = args.log
    show = not args.no_show
    use_warp = not args.no_warp
    enable_ldw = args.ldw
    ldw_threshold = args.ldw_threshold
    beep_on_ldw = args.beep

    # Normalize paths if relative
    base = Path(__file__).parent
    def norm(pth):
        if pth is None:
            return None
        s = str(pth)
        if s == "0":
            return "0"
        path = (Path.cwd() / s) if not Path(s).is_absolute() else Path(s)
        return str(path)

    input_path = norm(input_path)
    output_path = norm(output_path)
    log_path = norm(log_path)

    try:
        if mode == "image":
            if not input_path or not Path(input_path).exists():
                print(f"[ERROR] Image not found: {input_path}")
                return
            if output_path is None:
                out_p = str(Path(input_path).parent / (Path(input_path).stem + "_out" + Path(input_path).suffix))
            else:
                out_p = output_path
            process_image(input_path, out_p, use_warp=use_warp, show=show, enable_ldw=enable_ldw, ldw_threshold=ldw_threshold, beep_on_ldw=beep_on_ldw)
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
            process_video(input_path, output_path=out_p, log_path=log_path, use_warp=use_warp,
                          enable_ldw=enable_ldw, ldw_threshold=ldw_threshold, beep_on_ldw=beep_on_ldw, show=show)
        elif mode == "webcam":
            dev = input_path if input_path is not None else "0"
            # webcam uses video_processor; pass LDW options
            vp = VideoProcessor(dev, output_path=output_path, log_csv=log_path, use_warp=use_warp,
                                enable_ldw=enable_ldw, ldw_threshold=ldw_threshold, beep_on_ldw=beep_on_ldw)
            vp.start(show=show)
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Exception: {e}")

if __name__ == "__main__":
    main()

import streamlit as st
import sys, os
# Ensure src is on path
sys.path.append(os.path.dirname(__file__))

from pipeline import LaneProcessor
from video_processor import VideoProcessor
from road_classifier import classify_road_type
import numpy as np
import cv2, tempfile
from pathlib import Path

st.set_page_config(page_title="Road Lane Detection", layout="wide")
st.title("🚗 Road Lane Detection — Image & Video")

st.sidebar.header("Input")
mode = st.sidebar.selectbox("Mode", ["Image", "Video", "Webcam"])
use_warp = st.sidebar.checkbox("Use perspective warp", value=True)
show_debug = st.sidebar.checkbox("Show intermediate info", value=False)
run_button = st.sidebar.button("Run")

lp = LaneProcessor()

if mode == "Image":
    uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        out, info = lp.process_frame(img, use_warp=use_warp)
        road = classify_road_type(img)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=f"Output — {road}", use_column_width=True)
        if info:
            st.write("Offset (m):", round(info["offset"], 3))
elif mode == "Video":
    uploaded = st.file_uploader("Upload video", type=["mp4","avi","mov"])
    if uploaded and run_button:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        outpath = str(Path(tempfile.gettempdir()) / "out_demo.mp4")
        logpath = str(Path(tempfile.gettempdir()) / "log.csv")
        vp = VideoProcessor(tfile.name, output_path=outpath, log_csv=logpath, use_warp=use_warp)
        st.info("Processing video — this will create an output video in a temporary folder. Wait till processing finishes.")
        vp.start(show=False)
        st.success("Processing finished.")
        video_bytes = open(outpath,'rb').read()
        st.video(video_bytes)
        with open(logpath,'rb') as f:
            st.download_button("Download CSV log", f, file_name="lane_log.csv")
elif mode == "Webcam":
    st.info("Webcam mode uses local OpenCV window. Click Run to open the window. Press 'q' in the window to stop.")
    if run_button:
        vp = VideoProcessor('0', output_path=None, log_csv=None, use_warp=use_warp)
        vp.start(show=True)

st.sidebar.markdown("---")
st.sidebar.write("Tips:")
st.sidebar.write("- Use sample videos in `data/` folder for testing.")
st.sidebar.write("- For LinkedIn demo record only the output video (no code).")

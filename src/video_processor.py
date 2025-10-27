import cv2
import csv
import time
from .pipeline import LaneProcessor
from .road_classifier import classify_road_type

class VideoProcessor:
    def __init__(self, input_path, output_path=None, log_csv=None, use_warp=True):
        self.input_path = input_path
        self.output_path = output_path
        self.log_csv = log_csv
        self.cap = cv2.VideoCapture(input_path if input_path!='0' else 0)
        self.writer = None
        self.lp = LaneProcessor()
        self.use_warp = use_warp

    def start(self, show=True):
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w,h))
        csvfile = open(self.log_csv, 'w', newline='') if self.log_csv else None
        csvw = csv.writer(csvfile) if csvfile else None
        if csvw: csvw.writerow(["timestamp","frame","offset_m","road_type"])
        frame_idx = 0
        start = time.time()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            road_type = classify_road_type(frame)
            out_frame, info = self.lp.process_frame(frame, use_warp=self.use_warp)
            cv2.putText(out_frame, f"Road: {road_type}", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)
            if show:
                cv2.imshow("Output", out_frame)
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    break
            if self.writer:
                self.writer.write(out_frame)
            if csvw:
                offset = info['offset'] if info else None
                csvw.writerow([time.time()-start, frame_idx, offset, road_type])
            frame_idx += 1
        if csvfile: csvfile.close()
        if self.writer: self.writer.release()
        self.cap.release()
        cv2.destroyAllWindows()

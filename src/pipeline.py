import cv2
import numpy as np
from collections import deque
from .utils import perspective_matrices, warp, draw_lane_overlay, measure_vehicle_offset

def combined_threshold(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.absolute(sobelx)
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel)) if np.max(abs_sobel)!=0 else abs_sobel
    _, sxbinary = cv2.threshold(scaled, 50, 255, cv2.THRESH_BINARY)
    _, s_binary = cv2.threshold(s_channel, 90, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_or(sxbinary, s_binary)
    return combined

def region_of_interest(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    poly = np.array([[(int(w*0.05), h),(int(w*0.45), int(h*0.6)), (int(w*0.55), int(h*0.6)), (int(w*0.95), h)]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def sliding_window_poly(binary_warped, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0]); nonzerox = np.array(nonzero[1])
    leftx_current, rightx_current = leftx_base, rightx_base
    window_height = int(binary_warped.shape[0]//nwindows)
    left_lane_inds = []; right_lane_inds = []
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin; win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin; win_xright_high = rightx_current + margin
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds); right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        return None
    leftx = nonzerox[left_lane_inds]; lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]; righty = nonzeroy[right_lane_inds]
    if len(leftx) < 80 or len(rightx) < 80:
        return None
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx, right_fitx, ploty

class LaneProcessor:
    def __init__(self, smoothing_n=5):
        self.M = None; self.Minv = None
        self.left_fits = deque(maxlen=smoothing_n)
        self.right_fits = deque(maxlen=smoothing_n)

    def process_frame(self, frame, use_warp=True):
        orig = frame.copy()
        h, w = frame.shape[:2]
        if self.M is None:
            self.M, self.Minv = perspective_matrices(frame.shape)
        thresh = combined_threshold(frame)
        roi = region_of_interest(thresh)
        warped = warp(roi, self.M) if use_warp else roi
        res = sliding_window_poly(warped)
        if res is None:
            cv2.putText(orig, "No lanes detected", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            return orig, None
        left_fitx, right_fitx, ploty = res
        self.left_fits.append(left_fitx); self.right_fits.append(right_fitx)
        left_mean = np.mean(np.array(self.left_fits), axis=0)
        right_mean = np.mean(np.array(self.right_fits), axis=0)
        overlay = draw_lane_overlay(orig, left_mean, right_mean, ploty, self.Minv)
        offset = measure_vehicle_offset(w, left_mean, right_mean)
        cv2.putText(overlay, f"Offset: {offset:.2f} m", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        return overlay, {"offset": offset, "ploty": ploty, "leftx": left_mean, "rightx": right_mean}

import cv2
import numpy as np

def resize_keep_aspect(img, width=None, height=None):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = height / float(h)
        dim = (int(w * r), height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def perspective_matrices(img_shape):
    h, w = img_shape[:2]
    src = np.float32([
        [w*0.45, h*0.63],
        [w*0.55, h*0.63],
        [w*0.1,  h*0.95],
        [w*0.95, h*0.95]
    ])
    dst = np.float32([
        [w*0.2, 0],
        [w*0.8, 0],
        [w*0.2, h],
        [w*0.8, h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

def warp(img, M):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

def unwarp(img, Minv):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, Minv, (w, h), flags=cv2.INTER_LINEAR)

def draw_lane_overlay(orig, left_fitx, right_fitx, ploty, Minv, color=(0,255,0,)):
    warp_zero = np.zeros((ploty.shape[0], left_fitx.shape[0]), dtype=np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), color)
    newwarp = unwarp(color_warp, Minv)
    result = cv2.addWeighted(orig, 1, newwarp, 0.6, 0)
    return result

def measure_vehicle_offset(img_width, left_fitx, right_fitx):
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2.0
    vehicle_center = img_width / 2.0
    xm_per_pix = 3.7/700.0
    offset = (vehicle_center - lane_center) * xm_per_pix
    return offset

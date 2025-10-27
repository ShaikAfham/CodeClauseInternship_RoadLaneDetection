# src/utils.py
import cv2
import numpy as np
import math

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

def _ensure_array_lengths(ploty, left_fitx, right_fitx):
    """
    Ensure left_fitx/right_fitx arrays have same length as ploty by interpolation.
    Returns (ploty, left_fitx, right_fitx) as numpy arrays.
    """
    ploty = np.array(ploty)
    left_fitx = np.array(left_fitx)
    right_fitx = np.array(right_fitx)

    if left_fitx.shape[0] != ploty.shape[0]:
        left_fitx = np.interp(ploty, np.linspace(ploty[0], ploty[-1], left_fitx.shape[0]), left_fitx)
    if right_fitx.shape[0] != ploty.shape[0]:
        right_fitx = np.interp(ploty, np.linspace(ploty[0], ploty[-1], right_fitx.shape[0]), right_fitx)
    return ploty.astype(np.int32), left_fitx, right_fitx

def draw_lane_overlay(orig, left_fitx, right_fitx, ploty, Minv, color=(0,255,0), alpha=0.6):
    """
    Safely draw lane polygon overlay on `orig`.
    - Creates a color mask the same size as orig (h,w,3) and type uint8
    - Fills polygon between left_fitx & right_fitx using ploty coordinates
    - Unwarps using Minv with the same target size as orig
    - Uses addWeighted with matching shapes & dtypes
    """
    # Defensive conversions
    if orig is None:
        raise ValueError("orig image is None")

    h, w = orig.shape[:2]

    # Ensure arrays have consistent lengths
    ploty_arr, left_arr, right_arr = _ensure_array_lengths(ploty, left_fitx, right_fitx)

    # Build polygon points in bird-eye coordinates (x,y pairs)
    pts_left = np.vstack((left_arr.astype(np.int32), ploty_arr)).T
    pts_right = np.vstack((right_arr.astype(np.int32), ploty_arr)).T
    pts = np.vstack((pts_left, pts_right[::-1]))

    # Create warp-sized mask (same size as original)
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Clip x coordinates to valid image range to avoid invalid points
    pts[:,0] = np.clip(pts[:,0], 0, w-1)
    pts[:,1] = np.clip(pts[:,1], 0, h-1)

    # Fill polygon — use int32
    try:
        cv2.fillPoly(mask, [pts.astype(np.int32)], color)
    except Exception:
        # If fillPoly fails due to degenerate polygon, return original
        return orig

    # Unwarp mask back to original perspective (Minv should map to orig size)
    try:
        newwarp = cv2.warpPerspective(mask, Minv, (w, h), flags=cv2.INTER_LINEAR)
    except Exception:
        # If warp fails, return original
        return orig

    # Ensure types match: orig uint8 and newwarp uint8
    if orig.dtype != newwarp.dtype:
        newwarp = newwarp.astype(orig.dtype)

    # Finally overlay
    try:
        result = cv2.addWeighted(orig, 1.0, newwarp, alpha, 0)
    except Exception:
        # If addWeighted still fails, fallback to simple addition with clipping
        result = cv2.add(orig, newwarp)
    return result

def measure_vehicle_offset(img_width, left_fitx, right_fitx):
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2.0
    vehicle_center = img_width / 2.0
    xm_per_pix = 3.7/700.0
    offset = (vehicle_center - lane_center) * xm_per_pix
    return offset

import cv2
import numpy as np

def classify_road_type(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges>0) / edges.size
    brightness = np.mean(gray)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.var(lap)
    color_var = np.var(frame.reshape(-1,3), axis=0).mean()
    if brightness < 60:
        return "night"
    if texture < 50 and edge_density < 0.01:
        return "highway"
    if texture > 200 and edge_density > 0.03:
        return "urban"
    if edge_density > 0.02 and color_var > 200:
        return "mountain"
    return "curvy"

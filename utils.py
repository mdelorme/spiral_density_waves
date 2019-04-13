import numpy as np

def rotation(pts, angle):
    x = pts[:,0] * np.cos(angle) - pts[:,1] * np.sin(angle)
    y = pts[:,0] * np.sin(angle) + pts[:,1] * np.cos(angle)            
    return np.stack((x, y)).T

def ellipse(b, e, angles):
    r = b / np.sqrt(1 - (e * np.cos(angles)) ** 2.0)
    return np.stack((r * np.cos(angles), r * np.sin(angles))).T

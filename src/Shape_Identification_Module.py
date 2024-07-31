import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, distance
from matplotlib.patches import Ellipse
import numpy.linalg as linalg
import cv2

def is_straight_line(points, tolerance=0.01):
    if len(points) < 2:
        return False
    line_vec = points[-1] - points[0]
    line_dist = np.linalg.norm(line_vec)
    if line_dist == 0:
        return True

    unit_line_vec = line_vec / line_dist
    vec_from_start = points - points[0]
    proj_length = np.dot(vec_from_start, unit_line_vec)
    closest_point = np.outer(proj_length, unit_line_vec) + points[0]
    dist_from_line = np.linalg.norm(vec_from_start - closest_point, axis=1)

    return np.all(dist_from_line < tolerance)


def is_circle(points, tolerance=0.02):
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return np.std(distances) < tolerance

def is_rectangle(points, tolerance=0.02):
    if len(points) != 4:
        return False
    # Calculate the distances between consecutive points
    distances = np.linalg.norm(np.diff(points, axis=0, append=[points[0]]), axis=1)
    # Calculate angles between consecutive line segments
    angles = []
    for i in range(len(points)):
        p1 = points[i - 1]  # Previous point
        p2 = points[i]      # Current point
        p3 = points[(i + 1) % len(points)]  # Next point
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    # Check if all distances are equal and angles are right angles
    return np.std(distances) < tolerance and np.all(np.isclose(angles, np.pi/2, atol=tolerance))

def is_regular_polygon(points, side_count, tolerance=0.02):
    distances = np.linalg.norm(np.diff(points, axis=0, append=[points[0]]), axis=1)
    angles = []
    for i in range(len(points)):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[(i + 1) % len(points)]
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    return np.std(distances) < tolerance and np.std(angles) < tolerance

def is_star_shape(points, tolerance=0.05, spike_ratio=2):
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    sorted_distances = np.sort(distances)
    mean_inner = np.mean(sorted_distances[::2])  # Assuming spikes are at even indices
    mean_outer = np.mean(sorted_distances[1::2])  # Assuming dips are at odd indices
    return mean_outer / mean_inner > spike_ratio

def is_ellipse(points, tolerance=0.1):
    # Convert points to float32 if not already
    points = np.array(points, dtype=np.float32)
    
    # Check if there are enough points to fit an ellipse
    if len(points) < 5:
        return False

    try:
        # Fit an ellipse to the points
        ellipse = cv2.fitEllipse(points)
        
        # Draw the ellipse on a mask and calculate the area of the ellipse
        mask = np.zeros((int(points[:,1].max()) + 1, int(points[:,0].max()) + 1), dtype=np.uint8)
        cv2.ellipse(mask, ellipse, (255), -1)
        ellipse_area = np.sum(mask == 255)
        
        # Calculate the bounding box area of the points to compare
        x, y, w, h = cv2.boundingRect(points)
        bounding_box_area = w * h
        
        # Avoid division by zero
        if bounding_box_area == 0:
            return False

        # Calculate the area ratio and check against tolerance
        area_ratio = ellipse_area / bounding_box_area
        return abs(area_ratio - 1) < tolerance

    except Exception as e:
        print(f"Error fitting ellipse: {e}")
        return False
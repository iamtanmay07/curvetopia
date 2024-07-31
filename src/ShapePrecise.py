import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import leastsq
from matplotlib.patches import PathPatch, Ellipse
from matplotlib.path import Path
from collections import Counter

# Function to read CSV file
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Function to fit a line
def fit_line(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression().fit(X, y)
    return model

# Function to calculate line fit quality
def line_fit_quality(points):
    model = fit_line(points)
    X = points[:, 0].reshape(-1, 1)
    y_pred = model.predict(X)
    distances = np.abs(points[:, 1] - y_pred)
    return np.mean(distances)

# Function to fit a circle
def circle_fit(points):
    def residuals(params, points):
        x0, y0, r = params
        return np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2) - r

    x_m = np.mean(points[:, 0])
    y_m = np.mean(points[:, 1])
    r_guess = np.mean(np.sqrt((points[:, 0] - x_m) ** 2 + (points[:, 1] - y_m) ** 2))

    initial_guess = (x_m, y_m, r_guess)
    result, _ = leastsq(residuals, initial_guess, args=(points,))
    return result

# Function to calculate circle fit quality
def circle_fit_quality(points):
    x0, y0, r = circle_fit(points)
    distances = np.abs(np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2) - r)
    return np.mean(distances)

# Function to fit an ellipse
def ellipse_fit(points):
    def residuals(params, points):
        x0, y0, a, b, angle = params
        x, y = points[:, 0], points[:, 1]
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        x_centered = x - x0
        y_centered = y - y0
        x_rot = cos_angle * x_centered + sin_angle * y_centered
        y_rot = -sin_angle * x_centered + cos_angle * y_centered
        return ((x_rot / a) ** 2 + (y_rot / b) ** 2 - 1)

    x_m = np.mean(points[:, 0])
    y_m = np.mean(points[:, 1])
    a_guess = np.std(points[:, 0])
    b_guess = np.std(points[:, 1])
    angle_guess = 0

    initial_guess = (x_m, y_m, a_guess, b_guess, angle_guess)
    result, _ = leastsq(residuals, initial_guess, args=(points,))
    return result

# Function to calculate ellipse fit quality
def ellipse_fit_quality(points):
    x0, y0, a, b, angle = ellipse_fit(points)
    residuals = lambda p: np.sqrt(((p[:, 0] - x0) / a) ** 2 + ((p[:, 1] - y0) / b) ** 2) - 1
    return np.mean(np.abs(residuals(points)))

# Function to calculate polygon fit quality
def polygon_fit_quality(points, num_sides):
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    centroid = np.mean(points, axis=0)
    distances = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
    radius = np.mean(distances)
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    polygon_points = np.array([[centroid[0] + radius * np.cos(angle),
                                centroid[1] + radius * np.sin(angle)] for angle in angles])
    return np.mean(np.abs(np.sqrt(np.sum((polygon_points - centroid) ** 2, axis=1)) - radius))

# Function to determine the best matching shape
def identify_shape(points):
    shapes = {
        'line': line_fit_quality,
        'circle': circle_fit_quality,
        'ellipse': ellipse_fit_quality,
        'polygon': polygon_fit_quality
    }

    best_shape = None
    best_quality = float('inf')

    for shape, fit_function in shapes.items():
        if shape == 'polygon':
            for num_sides in range(3, 11):  # Test polygons with 3 to 10 sides
                quality = fit_function(points, num_sides)
                if quality < best_quality:
                    best_quality = quality
                    best_shape = (shape, num_sides)
        else:
            quality = fit_function(points)
            if quality < best_quality:
                best_quality = quality
                best_shape = (shape,)

    return best_shape

# Function to visualize the curves and identified shapes on a PNG file
def visualize_shapes_on_png(paths_XYs, identified_shapes, png_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    for i, (path_XYs, shapes) in enumerate(zip(paths_XYs, identified_shapes)):
        for XY in path_XYs:
            ax.plot(XY[:, 0], XY[:, 1], label=f'Path {i+1}', linewidth=2)

        for XY, shape in zip(path_XYs, shapes):
            if shape[0] == 'line':
                ax.plot(XY[:, 0], XY[:, 1], 'r-', linewidth=3)
            elif shape[0] == 'circle':
                center, radius = shape[1], shape[2]
                circle = plt.Circle(center, radius, edgecolor='b', facecolor='none', linestyle='--')
                ax.add_patch(circle)
            elif shape[0] == 'ellipse':
                x0, y0, a, b, angle = shape[1:]
                ellipse = Ellipse((x0, y0), 2*a, 2*b, angle=angle, edgecolor='purple', facecolor='none', linestyle='-.')
                ax.add_patch(ellipse)
            elif shape[0] == 'polygon':
                num_sides = shape[1]
                for XY in path_XYs:
                    path = Path(XY, closed=True)
                    patch = PathPatch(path, edgecolor='g', facecolor='none', linestyle='-.')
                    ax.add_patch(patch)

    plt.legend()
    plt.savefig(png_path, format='png')
    plt.show()

# Main execution
csv_path = './problems/isolated.csv'  # Update this path to your CSV file
png_path = 'output_image.png'

# Read polyline data from CSV
paths_XYs = read_csv(csv_path)

# Identify shapes in the data
identified_shapes = []
for path_XYs in paths_XYs:
    path_shapes = []
    for XY in path_XYs:
        XY = np.array(XY)
        shape = identify_shape(XY)
        path_shapes.append(shape)
    identified_shapes.append(path_shapes)

# Print shape counts
shape_counts = Counter([shape[0] for shapes in identified_shapes for shape in shapes])
for shape, count in shape_counts.items():
    print(f"Detected {count} {shape} shape(s)")

# Visualize and save the shapes in a PNG file
visualize_shapes_on_png(paths_XYs, identified_shapes, png_path)
import cv2
import numpy as np
from PIL import Image, ImageOps

def find_edges(image):
    gray = np.array(image.convert("L"))
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 50, 150) > 0

def sketch_scores(control, image, tol=4):
    sketch = np.array(control.convert("L")) > 127
    edges = find_edges(image)
    if not edges.any():
        return 0.0, 0.0

    dist_to_edge = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_sketch = cv2.distanceTransform((~sketch).astype(np.uint8), cv2.DIST_L2, 3)

    recall = (dist_to_edge[sketch] <= tol).mean()
    precision = (dist_to_sketch[edges] <= tol).mean()
    return float(recall), float(precision)


if __name__ == "__main__":
    control = Image.open("outputs/alien_1/control.png")
    for scale in [0.0, 0.5, 1.0, 1.3]:
        image = Image.open(f"outputs/alien_1/scale_{scale}.png")
        recall, precision = sketch_scores(control, image)
        print(f"scale {scale}: recall={recall:.2f}  precision={precision:.2f}")
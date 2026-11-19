import cv2
import numpy as np
from PIL import Image, ImageOps

def preprocess_sketch(path, size=512):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (size,size), interpolation=cv2.INTER_AREA)

    gray = cv2.GaussianBlur(gray, (3,3), 0)

    lines = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 12
    )

    kernel = np.ones((2,2), np.uint8)
    lines = cv2.dilate(lines, kernel)

    return Image.fromarray(lines).convert("RGB")

if __name__ == "__main__":
    control = preprocess_sketch("sketches/alien_1.jpeg")
    control.save("outputs/alien_1_control_v2.png")
    print("saved!")

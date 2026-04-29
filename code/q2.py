import cv2
import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMG_PATH = os.path.join(BASE_DIR, "images", "earrings.jpg")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q2")
os.makedirs(OUT_DIR, exist_ok=True)

# Load image
img = cv2.imread(IMG_PATH)

if img is None:
    raise FileNotFoundError(f"Image not found at {IMG_PATH}")

# ---------------- Camera parameters ----------------
focal_length_mm = 8
pixel_size_mm = 2.2 / 1000   # 2.2 µm → 0.0022 mm
distance_mm = 720

# Scale factor
mm_per_pixel = (pixel_size_mm * distance_mm) / focal_length_mm

print(f"Scale factor = {mm_per_pixel:.3f} mm/pixel\n")

# ---------------- Image processing ----------------
# Convert to HSV (better for color segmentation)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Gold/yellow color range (tuned for this image)
lower = np.array([10, 50, 50])
upper = np.array([40, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

# Clean noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# ---------------- Find earrings ----------------
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = img.copy()
earring_count = 1

for cnt in contours:
    area = cv2.contourArea(cnt)

    # Ignore noise
    if area > 1000:
        x, y, w, h = cv2.boundingRect(cnt)

        width_mm = w * mm_per_pixel
        height_mm = h * mm_per_pixel

        print(f"Earring {earring_count}")
        print(f"Pixel size: {w} px × {h} px")
        print(f"Real size : {width_mm:.2f} mm × {height_mm:.2f} mm\n")

        # Draw bounding box
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Put size text
        cv2.putText(
            result,
            f"{width_mm:.1f}mm x {height_mm:.1f}mm",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        earring_count += 1

# ---------------- Save outputs ----------------
cv2.imwrite(os.path.join(OUT_DIR, "q2_mask.png"), mask)
cv2.imwrite(os.path.join(OUT_DIR, "q2_result.png"), result)

print("Outputs saved in outputs/q2/")
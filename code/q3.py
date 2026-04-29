import cv2 as cv
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMG1_PATH = os.path.join(BASE_DIR, "images", "c1.jpg")
IMG2_PATH = os.path.join(BASE_DIR, "images", "c2.jpg")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q3")
os.makedirs(OUT_DIR, exist_ok=True)

img1 = cv.imread(IMG1_PATH)
img2 = cv.imread(IMG2_PATH)

if img1 is None:
    raise FileNotFoundError(f"Image not found: {IMG1_PATH}")
if img2 is None:
    raise FileNotFoundError(f"Image not found: {IMG2_PATH}")


# --------------------------------------------------
# Helper: resize for display only
# --------------------------------------------------
def resize_for_display(img, width=700):
    h, w = img.shape[:2]
    scale = width / w
    return cv.resize(img, (width, int(h * scale))), scale


# --------------------------------------------------
# Part (a): manual point selection
# --------------------------------------------------
def get_points(image, window_name, n=6):
    display_img, scale = resize_for_display(image)
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < n:
            original_x = x / scale
            original_y = y / scale
            points.append([original_x, original_y])
            cv.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            cv.putText(display_img, str(len(points)), (x + 8, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_callback)

    while True:
        cv.imshow(window_name, display_img)
        key = cv.waitKey(20) & 0xFF

        if len(points) == n:
            break
        if key == 27:
            break

    cv.destroyWindow(window_name)
    return np.array(points, dtype=np.float32)


print("Click 6 matching points on image c1.")
pts1 = get_points(img1, "Select 6 points in c1", 6)

print("Click the SAME 6 matching points on image c2 in the same order.")
pts2 = get_points(img2, "Select 6 points in c2", 6)

print("Manual points in c1:")
print(pts1)
print("Manual points in c2:")
print(pts2)

# Homography from c1 to c2
H_manual, mask_manual = cv.findHomography(pts1, pts2, cv.RANSAC)

manual_warped = cv.warpPerspective(img1, H_manual, (img2.shape[1], img2.shape[0]))

cv.imwrite(os.path.join(OUT_DIR, "q3a_manual_warped.png"), manual_warped)

# --------------------------------------------------
# Part (b): difference image for manual homography
# --------------------------------------------------
manual_diff = cv.absdiff(manual_warped, img2)
manual_diff_gray = cv.cvtColor(manual_diff, cv.COLOR_BGR2GRAY)
_, manual_diff_thresh = cv.threshold(manual_diff_gray, 30, 255, cv.THRESH_BINARY)

cv.imwrite(os.path.join(OUT_DIR, "q3b_manual_difference_gray.png"), manual_diff_gray)
cv.imwrite(os.path.join(OUT_DIR, "q3b_manual_difference_threshold.png"), manual_diff_thresh)


# --------------------------------------------------
# Part (c): ORB keypoints and matches
# --------------------------------------------------
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

orb = cv.ORB_create(nfeatures=3000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

good_matches = matches[:80]

match_img = cv.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv.imwrite(os.path.join(OUT_DIR, "q3c_orb_matches.png"), match_img)

print(f"Number of ORB keypoints in c1: {len(kp1)}")
print(f"Number of ORB keypoints in c2: {len(kp2)}")
print(f"Number of matches used: {len(good_matches)}")


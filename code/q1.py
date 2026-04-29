import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "lines.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "q1")
os.makedirs(OUT_DIR, exist_ok=True)

D = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1)

def fit_tls(x, y):
    points = np.column_stack((x, y))
    centroid = np.mean(points, axis=0)

    A = points - centroid
    _, _, Vt = np.linalg.svd(A)

    direction = Vt[0]
    normal = np.array([-direction[1], direction[0]])
    normal = normal / np.linalg.norm(normal)

    a, b = normal
    c = -(a * centroid[0] + b * centroid[1])

    if b < 0:
        a, b, c = -a, -b, -c

    m = -a / b
    k = -c / b

    return a, b, c, m, k


# ---------------- Q1(a): TLS for first line ----------------
x1 = D[:, 0]
y1 = D[:, 3]

a, b, c, m, k = fit_tls(x1, y1)

print("Q1(a): TLS line for first scatter")
print(f"ax + by + c = 0: a={a:.4f}, b={b:.4f}, c={c:.4f}")
print(f"y = mx + k: m={m:.4f}, k={k:.4f}")


# Plot Q1(a)
plt.figure(figsize=(6, 5))
plt.scatter(x1, y1, label="First line data")

xx = np.linspace(min(x1), max(x1), 100)
yy = m * xx + k

plt.plot(xx, yy, label="TLS fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Q1(a): Total Least Squares Fit")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "q1a_tls_first_line.png"), dpi=300)
plt.close()


# ---------------- Q1(b): RANSAC for three lines ----------------
X_cols = D[:, :3]
Y_cols = D[:, 3:]

X_all = X_cols.flatten()
Y_all = Y_cols.flatten()

def line_from_two_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    norm = np.sqrt(a * a + b * b)
    if norm == 0:
        return None

    return np.array([a / norm, b / norm, c / norm])


def point_line_distance(line, x, y):
    a, b, c = line
    return np.abs(a * x + b * y + c)


def ransac_line(x, y, threshold=0.7, iterations=10000):
    best_inliers = []

    rng = np.random.default_rng(42)

    for _ in range(iterations):
        ids = rng.choice(len(x), 2, replace=False)

        p1 = (x[ids[0]], y[ids[0]])
        p2 = (x[ids[1]], y[ids[1]])

        line = line_from_two_points(p1, p2)

        if line is None:
            continue

        distances = point_line_distance(line, x, y)
        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    return best_inliers


remaining = np.arange(len(X_all))
detected_lines = []

for i in range(3):
    x_remaining = X_all[remaining]
    y_remaining = Y_all[remaining]

    inlier_ids_local = ransac_line(x_remaining, y_remaining)
    inlier_ids_global = remaining[inlier_ids_local]

    x_in = X_all[inlier_ids_global]
    y_in = Y_all[inlier_ids_global]

    a, b, c, m, k = fit_tls(x_in, y_in)

    detected_lines.append((a, b, c, m, k, inlier_ids_global))

    remaining = np.setdiff1d(remaining, inlier_ids_global)

print("\nQ1(b): RANSAC + TLS fitted lines")

for i, (a, b, c, m, k, inliers) in enumerate(detected_lines, start=1):
    print(f"\nLine {i}")
    print(f"Inliers: {len(inliers)}")
    print(f"ax + by + c = 0: a={a:.4f}, b={b:.4f}, c={c:.4f}")
    print(f"y = mx + k: m={m:.4f}, k={k:.4f}")


# Plot Q1(b)
plt.figure(figsize=(7, 6))
plt.scatter(X_all, Y_all, s=15, label="All points")

xx = np.linspace(min(X_all), max(X_all), 200)

for i, (a, b, c, m, k, inliers) in enumerate(detected_lines, start=1):
    yy = m * xx + k
    plt.plot(xx, yy, linewidth=2, label=f"RANSAC line {i}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Q1(b): Three Lines using RANSAC")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "q1b_ransac_three_lines.png"), dpi=300)
plt.close()
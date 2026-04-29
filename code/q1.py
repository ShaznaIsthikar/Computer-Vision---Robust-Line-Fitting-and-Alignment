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


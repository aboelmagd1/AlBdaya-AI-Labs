import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================
# إعداد الشكل العام
# ==========================
plt.style.use('dark_background')

# تصغير الخطوط
plt.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 9,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.titlesize': 16
})

fig = plt.figure(figsize=(22, 20))
fig.suptitle(
    "Mathematics & AI Functions Atlas",
    fontsize=16,
    fontweight='bold'
)

# المجالات المختلفة
x = np.linspace(-5, 5, 1000)
x_pos = np.linspace(0.01, 5, 1000)

# ==========================
# 1) Linear Function
# ==========================
ax = fig.add_subplot(4, 4, 1)
ax.plot(x, 2*x + 1, lw=2.5)
ax.set_title("Linear Function\n$f(x)=2x+1$")
ax.grid(alpha=0.3)

# ==========================
# 2) Polynomial Functions
# ==========================
ax = fig.add_subplot(4, 4, 2)

colors = ['cyan', 'lime', 'orange', 'red', 'magenta']

for n, c in zip(range(1, 6), colors):
    ax.plot(x, x**n, label=f"$x^{n}$", color=c)

ax.set_ylim(-150, 150)
ax.set_title("Polynomial Functions")
ax.legend()
ax.grid(alpha=0.3)

# ==========================
# 3) Exponential
# ==========================
ax = fig.add_subplot(4, 4, 3)

ax.plot(x, np.exp(x), color='gold', lw=2.5)
ax.set_ylim(0, 150)

ax.set_title("Exponential\n$f(x)=e^x$")
ax.grid(alpha=0.3)

# ==========================
# 4) Logarithm
# ==========================
ax = fig.add_subplot(4, 4, 4)

ax.plot(x_pos, np.log(x_pos), color='lime', lw=2.5)

ax.set_title("Natural Logarithm\n$f(x)=\\ln(x)$")
ax.grid(alpha=0.3)

# ==========================
# 5) Sigmoid
# ==========================
ax = fig.add_subplot(4, 4, 5)

sigmoid = 1 / (1 + np.exp(-x))

ax.plot(x, sigmoid, color='red', lw=2.5)

ax.set_title("Sigmoid")
ax.grid(alpha=0.3)

# ==========================
# 6) Tanh
# ==========================
ax = fig.add_subplot(4, 4, 6)

ax.plot(x, np.tanh(x), color='cyan', lw=2.5)

ax.set_title("Tanh")
ax.grid(alpha=0.3)

# ==========================
# 7) ReLU
# ==========================
ax = fig.add_subplot(4, 4, 7)

relu = np.maximum(0, x)

ax.plot(x, relu, color='orange', lw=2.5)

ax.set_title("ReLU")
ax.grid(alpha=0.3)

# ==========================
# 8) Leaky ReLU
# ==========================
ax = fig.add_subplot(4, 4, 8)

leaky = np.where(x > 0, x, 0.01*x)

ax.plot(x, leaky, color='yellow', lw=2.5)

ax.set_title("Leaky ReLU")
ax.grid(alpha=0.3)

# ==========================
# 9) GELU
# ==========================
ax = fig.add_subplot(4, 4, 9)

gelu = 0.5 * x * (
    1 + np.tanh(
        np.sqrt(2/np.pi) *
        (x + 0.044715*x**3)
    )
)

ax.plot(x, gelu, color='violet', lw=2.5)

ax.set_title("GELU (GPT/BERT)")
ax.grid(alpha=0.3)

# ==========================
# 10) Softmax
# ==========================
ax = fig.add_subplot(4, 4, 10)

z = np.linspace(-5, 5, 100)
softmax = np.exp(z) / np.sum(np.exp(z))

ax.plot(z, softmax, color='green', lw=2.5)

ax.set_title("Softmax Distribution")
ax.grid(alpha=0.3)

# ==========================
# 11) Gaussian Distribution
# ==========================
ax = fig.add_subplot(4, 4, 11)

gaussian = (
    1 / np.sqrt(2*np.pi)
) * np.exp(-x**2 / 2)

ax.plot(x, gaussian, color='deepskyblue', lw=2.5)

ax.set_title("Gaussian Distribution")
ax.grid(alpha=0.3)

# ==========================
# 12) Sign Function
# ==========================
ax = fig.add_subplot(4, 4, 12)

sign = np.sign(x)

ax.plot(x, sign, color='orange', lw=2.5)

ax.set_ylim(-1.5, 1.5)

ax.set_title("Sign Function")
ax.grid(alpha=0.3)

# ==========================
# 13) MSE Loss
# ==========================
ax = fig.add_subplot(4, 4, 13)

error = np.linspace(-5, 5, 1000)
mse = error**2

ax.plot(error, mse, color='red', lw=2.5)

ax.set_title("MSE Loss")
ax.grid(alpha=0.3)

# ==========================
# 14) MAE Loss
# ==========================
ax = fig.add_subplot(4, 4, 14)

mae = np.abs(error)

ax.plot(error, mae, color='lime', lw=2.5)

ax.set_title("MAE Loss")
ax.grid(alpha=0.3)

# ==========================
# 15) Gradient Descent Field
# ==========================
ax = fig.add_subplot(4, 4, 15)

Xg, Yg = np.meshgrid(
    np.linspace(-3, 3, 20),
    np.linspace(-3, 3, 20)
)

Zg = Xg**2 + Yg**2

dZdx = 2 * Xg
dZdy = 2 * Yg

ax.contour(Xg, Yg, Zg, 15)

ax.quiver(
    Xg, Yg,
    -dZdx, -dZdy,
    scale=60
)

ax.set_title("Gradient Descent Field")

# ==========================
# 16) 3D Loss Surface
# ==========================
ax3d = fig.add_subplot(
    4, 4, 16,
    projection='3d'
)

X, Y = np.meshgrid(
    np.linspace(-4, 4, 120),
    np.linspace(-4, 4, 120)
)

Z = (
    X**2 +
    0.5*Y**2 +
    np.sin(X)*np.cos(Y)
)

surf = ax3d.plot_surface(
    X,
    Y,
    Z,
    cmap='viridis',
    edgecolor='none',
    alpha=0.9
)

ax3d.set_title("3D Loss Surface", fontsize=9)

fig.colorbar(
    surf,
    ax=ax3d,
    shrink=0.55,
    aspect=12
)

# ==========================
# تحسين المظهر
# ==========================
for ax in fig.axes:
    try:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    except:
        pass

# plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.tight_layout()
plt.show()
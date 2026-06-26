import numpy as np
import matplotlib.pyplot as plt

# إعداد واجهة الرسم (3 صفوف وعمودين لعرض 6 دوال)
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Plot of the Functions from the Image', fontsize=16, fontweight='bold')

# لتفادي القسمة على صفر، نقسم النطاق إلى جزء موجب وجزء سالب للدوال الترددية
x_fine_neg = np.linspace(-1, -0.001, 1000)
x_fine_pos = np.linspace(0.001, 1, 1000)
x_fine = np.concatenate([x_fine_neg, x_fine_pos])

x_standard = np.linspace(-2, 2, 400)

# ---------------------------------------------------------
# 1. f(x) = x * sin(1/x)
# ---------------------------------------------------------
y1 = x_fine * np.sin(1 / x_fine)
axs[0, 0].plot(x_fine, y1, color='blue', label=r'$f(x) = x \sin(\frac{1}{x})$')
axs[0, 0].set_title(r'$f(x) = x \sin(\frac{1}{x})$')
axs[0, 0].grid(True, linestyle='--')
axs[0, 0].axhline(0, color='black', linewidth=0.5)
axs[0, 0].axvline(0, color='black', linewidth=0.5)

# ---------------------------------------------------------
# 2. f(x) = x^2 * sin(1/x)
# ---------------------------------------------------------
y2 = (x_fine**2) * np.sin(1 / x_fine)
axs[0, 1].plot(x_fine, y2, color='green', label=r'$f(x) = x^2 \sin(\frac{1}{x})$')
axs[0, 1].set_title(r'$f(x) = x^2 \sin(\frac{1}{x})$')
axs[0, 1].grid(True, linestyle='--')
axs[0, 1].axhline(0, color='black', linewidth=0.5)
axs[0, 1].axvline(0, color='black', linewidth=0.5)

# ---------------------------------------------------------
# 3. f(x) = sin(1/x)
# ---------------------------------------------------------
y3 = np.sin(1 / x_fine)
axs[1, 0].plot(x_fine, y3, color='red', label=r'$f(x) = \sin(\frac{1}{x})$')
axs[1, 0].set_title(r'$f(x) = \sin(\frac{1}{x})$')
axs[1, 0].grid(True, linestyle='--')
axs[1, 0].axhline(0, color='black', linewidth=0.5)
axs[1, 0].axvline(0, color='black', linewidth=0.5)

# ---------------------------------------------------------
# 4. f(x) = c (Constant function, assuming c = 2 as an example)
# ---------------------------------------------------------
c_val = 2
y4 = np.full_like(x_standard, c_val)
axs[1, 1].plot(x_standard, y4, color='purple', label=r'$f(x) = c$')
axs[1, 1].set_title(r'$f(x) = c$ (for $c=2$)')
axs[1, 1].grid(True, linestyle='--')
axs[1, 1].axhline(0, color='black', linewidth=0.5)
axs[1, 1].axvline(0, color='black', linewidth=0.5)
axs[1, 1].set_ylim(-1, c_val + 2)

# ---------------------------------------------------------
# 5. f(x) = { 1, x > 0 ; -1, x < 0 } (Signum function)
# ---------------------------------------------------------
x5_neg = np.linspace(-2, -0.01, 200)
x5_pos = np.linspace(0.01, 2, 200)
axs[2, 0].plot(x5_neg, np.full_like(x5_neg, -1), color='orange', linewidth=2)
axs[2, 0].plot(x5_pos, np.full_like(x5_pos, 1), color='orange', linewidth=2)
# رسم الدوائر المفتوحة عند الصفر لتوضيح عدم الاتصال
axs[2, 0].scatter([0, 0], [-1, 1], facecolors='none', edgecolors='orange', s=50, zorder=3)
axs[2, 0].set_title(r'$f(x) = \pm 1$ (Sign Function)')
axs[2, 0].grid(True, linestyle='--')
axs[2, 0].axhline(0, color='black', linewidth=0.5)
axs[2, 0].axvline(0, color='black', linewidth=0.5)
axs[2, 0].set_ylim(-2, 2)

# ---------------------------------------------------------
# 6. f(x) = x^2
# ---------------------------------------------------------
y6 = x_standard**2
axs[2, 1].plot(x_standard, y6, color='brown', label=r'$f(x) = x^2$')
axs[2, 1].set_title(r'$f(x) = x^2$')
axs[2, 1].grid(True, linestyle='--')
axs[2, 1].axhline(0, color='black', linewidth=0.5)
axs[2, 1].axvline(0, color='black', linewidth=0.5)

# تحسين المسافات بين الرسومات وعرض اللوحة
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
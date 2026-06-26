"""
Built-in functions, NumPy functions, and Matplotlib functions
used in week2_numpy_visualization_OOP.ipynb
"""

# ==================================================
# Python Built-in Functions
# ==================================================

print()
type(object)
len(object)
sum(iterable)
min(iterable)
max(iterable)
abs(number)


# ==================================================
# NumPy Functions
# ==================================================

import numpy as np

np.array(object)
np.zeros(shape)
np.ones(shape)
np.arange(start, stop, step)
np.linspace(start, stop, num)

np.mean(a)
np.median(a)
np.std(a)
np.var(a)
np.sum(a)
np.min(a)
np.max(a)

np.sqrt(a)
np.abs(a)
np.exp(a)
np.sin(a)
np.cos(a)

np.dot(a, b)

np.all(a)
np.allclose(a, b)
np.array_equal(a, b)

np.random.seed(seed)
np.random.randn(*shape)


# ==================================================
# NumPy Array Attributes
# ==================================================

arr = np.array([1, 2, 3])

arr.shape
arr.ndim
arr.size
arr.dtype

np.__version__


# ==================================================
# Matplotlib Functions
# ==================================================

import matplotlib.pyplot as plt

plt.figure()
plt.subplots()

plt.plot()
plt.scatter()
plt.hist()

plt.title("Title")
plt.xlabel("X")
plt.ylabel("Y")

plt.grid()
plt.legend()

plt.axhline()
plt.axvline()

plt.tight_layout()
plt.show()

plt.style.use("default")

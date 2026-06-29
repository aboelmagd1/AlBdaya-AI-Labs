import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
ar = np.arange(1 , 100 , 1)
print(ar)
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
np.random.seed(5)
arr = np.linspace(1, 10, 1000) #من 1 الى 10 انت عايز 1000 رقم يعني الـlen = 1000
# print(arr)
# print(len(arr)) #1000
#create liner data 
x = np.arange(1 , 100 )
print(x)
y = 2*x + 20
y = 2 * np.random.normal(0, 1.5) * x + 5 + np.random.normal(0, 1.5, size=x.shape)
print(y)

plt.figure(figsize=(16, 9), dpi=100)
plt.subplot(2, 2, 1) 
plt.plot(x, y, 'oy', markersize=2.5, label='Data points')
plt.grid(True, alpha=0.3)



plt.legend()
plt.tight_layout()
plt.show()
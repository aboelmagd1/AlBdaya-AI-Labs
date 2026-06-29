import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# توليد بيانات أكثر واقعية وشبيهة بالحقيقية
np.random.seed(42)
x = np.linspace(1, 10, 150)
y = 2.0 * x + 5.0 + np.random.normal(0, 1.5, size=x.shape)

plt.figure(figsize=(20, 9), dpi=100)
plt.subplot(1, 2, 1) # for divid figure
plt.plot(x, y, 'oy', markersize=8, label='Data points')
# plt.plot(x, y, 'r-', linewidth=1)

np.random.seed(2)
w = np.random.rand(1)[0]
b = np.random.rand(1)[0]



eta = 0.01
epochs = 1000
costVal = np.zeros(epochs)

for i in range(epochs):
    yhat = (w * x) + b #to get prodected y 
    w = w - eta * (yhat - y) @ x / len(x) # to get wieght الفكرة بتاعتها جاية من متوسط مربع الخطأ بعد ما عموضنا بالمعادلة الخاصة الخط واخدنا المشتقة الاولى وبعد كدا تلاعبنا شوبة  باننا حطينا reta وال2 الخاصة بالتبسيط والمشتقة طبعا منسبة للw لان المعادة اللي تحتت بالنسة ل b
    b = b - eta * (np.mean(yhat - y))

    costVal[i] = 0.5 * (np.mean((yhat-y)**2)) # دي معادلة MSE 
    if costVal[i] < 1e-5: # الخطأ المسموح به
        break
#to drow line
xvals = np.linspace(x.min() - 2, x.max() + 2, 100) 
yvals = (w * xvals) + b
plt.plot(xvals, yvals, 'r-', linewidth=2, label='Fitted line')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Linear Regression using Gradient Descent')
plt.grid(True, alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(range(1, i+2), costVal[:i+1])
plt.xlabel('Epochs')
plt.ylabel('Cost MSE')
plt.title('training cost vs epochs')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f'Final w: {w: 0.4f}')
print(f'Final b: {b: 0.4f}')
print(f'Final cost: {costVal[i]: 0.4f}')
print(f'Training stopped at epoch: {i+1}')
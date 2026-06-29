import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

class ClassificationPipeline:
    def __init__(self, eta=0.1, epochs=1000):
        self.eta = eta
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.cost_history = []
        
    def sigmoid(self, z):
        # دالة التنشيط لتحويل المخرجات للاحتمالات بين 0 و 1
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # 1. Synthetic 2D data generation (2 classes)
    def generate_2D_data(self, num_samples=200):
        np.random.seed(42)
        # توليد ميزات ثنائية الأبعاد (2D Features)
        X1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], num_samples // 2)
        X2 = np.random.multivariate_normal([-2, -2], [[1, 0.5], [0.5, 1]], num_samples // 2)
        
        self.X = np.vstack((X1, X2))
        # الفئات: 0 للفئة الأولى و 1 للفئة الثانية
        self.y = np.hstack((np.zeros(num_samples // 2), np.ones(num_samples // 2)))
        return self.X, self.y
    
    # 2. Feature normalization (z-score)
    def normalization(self):
        # تقييس الميزات لكل عمود على حدة ليصبح المتوسط=0 والانحراف=1
        self.X_norm = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)
        return self.X_norm

    # 3. Simple model class (Training Logic)
    def fit(self):
        # تهيئة الأوزان بناءً على عدد الميزات (2 ميزات)
        self.w = np.random.randn(self.X_norm.shape[1])
        self.b = np.random.randn()
        m = self.X_norm.shape[0]
        
        for i in range(self.epochs):
            # الحساب التوقعي (Z ثم احتمالية السيجويد)
            z = np.dot(self.X_norm, self.w) + self.b
            y_hat = self.sigmoid(z)
            
            # حساب دالة التكلفة (Binary Cross-Entropy Loss)
            cost = -(1/m) * np.sum(self.y * np.log(y_hat + 1e-15) + (1 - self.y) * np.log(1 - y_hat + 1e-15))
            self.cost_history.append(cost)
            
            # حساب الاشتقاق وتحديث الأوزان (Gradient Descent)
            dw = (1/m) * np.dot(self.X_norm.T, (y_hat - self.y))
            db = (1/m) * np.sum(y_hat - self.y)
            
            self.w -= self.eta * dw
            self.b -= self.eta * db

    # 4. Predictions and error calculation
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        probabilities = self.sigmoid(z)
        return (probabilities >= 0.5).astype(int)
    
    def calculate_error(self):
        predictions = self.predict(self.X_norm)
        # حساب نسبة الخطأ في التصنيف
        error_rate = np.mean(predictions != self.y)
        return error_rate

    # 5. Comprehensive visualization (2x2 subplots)
    def plot_results(self):
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # الرسمة 1: البيانات الأصلية قبل التقييس
        axs[0, 0].scatter(self.X[self.y == 0, 0], self.X[self.y == 0, 1], color='red', label='Class 0', alpha=0.7)
        axs[0, 0].scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], color='blue', label='Class 1', alpha=0.7)
        axs[0, 0].set_title('1. Original Raw Data (2D)')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.2)
        
        # الرسمة 2: البيانات بعد التقييس (Z-Score)
        axs[0, 1].scatter(self.X_norm[self.y == 0, 0], self.X_norm[self.y == 0, 1], color='red', label='Class 0', alpha=0.7)
        axs[0, 1].scatter(self.X_norm[self.y == 1, 0], self.X_norm[self.y == 1, 1], color='blue', label='Class 1', alpha=0.7)
        axs[0, 1].set_title('2. Normalized Data (Mean=0, Std=1)')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.2)
        
        # الرسمة 3: منحنى انخفاض التكلفة (Cost Convergence)
        axs[1, 0].plot(self.cost_history, color='lime')
        axs[1, 0].set_title('3. Loss Convergence (BCE Cost)')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Cost')
        axs[1, 0].grid(True, alpha=0.2)
        
        # الرسمة 4: حد القرار النهائي للنموذج (Decision Boundary)
        axs[1, 1].scatter(self.X_norm[self.y == 0, 0], self.X_norm[self.y == 0, 1], color='red', alpha=0.6)
        axs[1, 1].scatter(self.X_norm[self.y == 1, 0], self.X_norm[self.y == 1, 1], color='blue', alpha=0.6)
        
        # حساب ورسم الخط الفاصل (Decision Boundary Line)
        x_values = np.linspace(np.min(self.X_norm[:, 0]), np.max(self.X_norm[:, 0]), 100)
        y_values = -(self.w[0] * x_values + self.b) / self.w[1]
        axs[1, 1].plot(x_values, y_values, label='Decision Boundary', color='yellow', linewidth=2)
        
        error = self.calculate_error()
        axs[1, 1].set_title(f'4. Final Model (Error Rate: {error*100:.2f}%)')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.show()

# --- تشغيل خط الأنابيب بالخطوات المطلوبة كاملاً ---
pipeline = ClassificationPipeline(eta=0.1, epochs=500)
pipeline.generate_2D_data(200) # 1. توليد بيانات ثنائية الأبعاد لفئتين
pipeline.normalization()       # 2. عمل تقييس للبيانات
pipeline.fit()                 # 3. تدريب النموذج البسيط
pipeline.plot_results()        # 4 و 5. حساب الأخطاء وعرض الرسومات في شبكة 2x2
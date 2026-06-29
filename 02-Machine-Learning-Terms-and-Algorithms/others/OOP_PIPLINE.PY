# Create a classification pipeline with:

# 1. Synthetic 2D data generation (2 classes)

# 2. Feature normalization (z-score)

# 3. Simple model class

# 4. Predictions and error calculation

# 5. Comprehensive visualization (2x2 subplots)


# YOUR CODE HERE
# class LinearRegPipeline :
#     def __int__(self , learningrate = .01 , epochs=1000):
#         self.lr = learningrate
#         self.epochs = epochs
#         self.w = np.random.randn()
#         self.b = np.random.randn()


import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

class LinearRegPipeline:
    def __init__(self, eta=0.01, epochs=1000):
        self.eta = eta
        self.epochs = epochs
        self.w = np.random.randn()
        self.b = np.random.randn()
        self.costVal = np.zeros(self.epochs)
    


    def generate_2D_data(self, num_samples=500 , minvalue= 1, maxvalue = 10):
        np.random.seed(42) 
        self.x = np.linspace(minvalue , maxvalue , num_samples)
        self.y = 3 * self.x + 5 * np.random.normal(0, .6, size=num_samples) #np.random.randn(num_samples)
        self.n = len(self.x)
        return self.x , self.y

    def fit(self):
        for i in range(self.epochs):
            self.yhat = (self.w * self.x) + self.b
            self.w = self.w - self.eta * (self.yhat - self.y) @ self.x / self.n
            self.b = self.b - self.eta * (np.mean(self.yhat - self.y))

            self.costVal[i] = 0.5 * (np.mean((self.yhat-self.y)**2))
            if self.costVal[i] < 1e-5:
                break

    def plot_results(self):
        # Matplotlib visualization
        plt.scatter(self.x , self.y, color='blue', label='Data Points')
        plt.plot(self.x, (self.w * self.x + self.b), color='red', linewidth=2, label='Regression Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Final Model: y = {self.w:.2f}x + {self.b:.2f}')
        plt.legend()
        plt.grid(True , alpha = .3)
        plt.show()

# --- How to run the pipeline ---
pipeline = LinearRegPipeline(eta=0.001, epochs=2000)
pipeline.generate_2D_data(100 , 1 ,20)
pipeline.fit()
pipeline.plot_results()
# raise NotImplementedError()
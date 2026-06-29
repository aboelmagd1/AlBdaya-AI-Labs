import numpy as np

# # Create a 1D array from a list
# arr_1d = np.array([1, 2, 3, 4])
# print(arr_1d)
# print (5 * arr_1d)
# # Create a 2D array (matrix)
# arr_2d = np.array([[1, 2], [3, 4]])
# print (5 * arr_2d)

def create_polynomial_features(x, degree):
    """Create polynomial features"""
    return np.column_stack([x**i for i in range(degree + 1)])

x = np.arange(2 , 10 , 1)
print(x , 10**1)
degree = [range(1,6)]
print(create_polynomial_features (x , 3))

x = create_polynomial_features (x , 3)
def standardize_features(X):
    """Standardize features to have mean=0, std=1"""
    X_std = X.copy()
    # Don't standardize the bias term (first column of ones)
    X_std[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
    return X_std


print(standardize_features(x))
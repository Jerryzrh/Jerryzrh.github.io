import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import numpy as np

# Load the diabetes dataset
diabetes = load_diabetes()

# Create a Pandas DataFrame from the data
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
X = diabetes.data
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize variables to store results
best_k = None
min_rmse = float('inf')
best_model = None

# Iterate through values of k from 1 to 10
for k in range(1, 11):
    # Initialize kNN regressor with current k value
    knn_regressor = KNeighborsRegressor(n_neighbors=k)

    # Fit kNN regressor to training data
    knn_regressor.fit(X_train, y_train)

    # Predict on test set
    y_pred = knn_regressor.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Check if current RMSE is lower than the minimum RMSE found so far
    if rmse < min_rmse:
        min_rmse = rmse
        best_k = k
        best_model = 'kNN Regression'

# Initialize linear regression model
linear_regressor = LinearRegression()

# Fit linear regression to training data
linear_regressor.fit(X_train, y_train)

# Predict on test set
y_pred_linear = linear_regressor.predict(X_test)

# Calculate RMSE for linear regression
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))

# Check if RMSE for linear regression is lower than the current minimum RMSE
if rmse_linear < min_rmse:
    min_rmse = rmse_linear
    best_model = 'Linear Regression'

# Initialize support vector regression model
svr_regressor = SVR()

# Fit support vector regression to training data
svr_regressor.fit(X_train, y_train)

# Predict on test set
y_pred_svr = svr_regressor.predict(X_test)

# Calculate RMSE for support vector regression
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))

# Check if RMSE for support vector regression is lower than the current minimum RMSE
if rmse_svr < min_rmse:
    min_rmse = rmse_svr
    best_model = 'Support Vector Regression'

# Output results
print("Best value of k for kNN Regression:", best_k)
print("RMSE for the best kNN Regression model:", min_rmse)
print("Best model:", best_model)

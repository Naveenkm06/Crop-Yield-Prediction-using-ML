#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset (replace 'crop_data.csv' with your dataset)
data = pd.read_csv('augumented_data_set.csv')

# Select features and target variable
X = data[['AREA', 'N','P','K']]  # Features
y = data['YIELD']  # Target variable

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Fit the model
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gbr.fit(X_normalized, y)

# Function to predict yield for user input
def predict_yield():
    AREA = float(input("Enter area value in Hectare: "))
    
    N = float(input("Enter nitrogen usage in percentage: "))
    P = float(input("Enter pottasium usage percentage: "))
    K = float(input("Enter phosphate usage percentage: "))
    
    # Normalize user input
    new_data_point = np.array([[AREA, N,P,K]])
    new_data_point_normalized = scaler.transform(new_data_point)
    
    # Predict yield
    predicted_yield = gbr.predict(new_data_point_normalized)[0]
    print("Predicted yield:", predicted_yield)
    
    # Calculate R2 score and MSE
    y_pred = gbr.predict(X_normalized)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print("R-squared:", r2)
    print("Mean Squared Error:", mse)
    
    # Calculate accuracy
    check_accuracy = input("Do you want to check accuracy? (yes/no): ")
    if check_accuracy.lower() == 'yes':
        actual_yield = float(input("Enter actual yield: "))
        accuracy = 100 * (1 - abs(actual_yield - predicted_yield) / actual_yield)
        print("Accuracy:", accuracy)
    
    # Visualize predicted vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='blue', label='Predictions')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Predictions')
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Actual vs Predicted Yield')
    plt.legend()
    plt.show()

# User interface loop
while True:
    print("\nEnter input values to predict yield (type 'exit' to quit):")
    predict_yield_input = input("Enter 'yes' to predict yield or 'exit' to quit: ")
    if predict_yield_input.lower() == 'exit':
        print("THANK YOU !")
        break
    elif predict_yield_input.lower() == 'yes':
        predict_yield()
    else:
        print("Invalid input. Please enter 'yes' or 'exit'.")


# In[ ]:





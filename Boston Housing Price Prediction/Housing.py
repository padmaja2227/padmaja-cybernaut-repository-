# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages

# Ignore warnings
warnings.filterwarnings('ignore')

# 1. Data Preprocessing

# Load the dataset
df = pd.read_excel("Boston Dataset.xlsx")

# Drop 'Unnamed: 0' if it exists or any irrelevant column
df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

# Handle missing data (check for missing values)
print(f"Missing values in each column: {df.isnull().sum()}")

# Normalize or standardize features using StandardScaler
scaler = StandardScaler()
X = df.drop(columns=['medv'])  # Features
y = df['medv']  # Target variable (median house price)

X_scaled = scaler.fit_transform(X)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Model Building

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Ridge Regression Model with hyperparameter tuning (alpha)
ridge_model = Ridge()

# Hyperparameter tuning using GridSearchCV to find the best alpha value for Ridge
param_grid = {'alpha': np.logspace(-4, 4, 100)}
grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best alpha value for Ridge Regression
best_alpha = grid_search.best_params_['alpha']
print(f"Best alpha for Ridge Regression: {best_alpha}")

# Train Ridge model with the best alpha
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train, y_train)

# 3. Model Evaluation

# Predictions using both models
y_pred_lr = lr_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)

# Mean Squared Error and R-squared for Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Mean Squared Error and R-squared for Ridge Regression
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# 4. Data Visualization

# Create PDF to save outputs
with PdfPages('model_evaluation_output.pdf') as pdf:

    # Output evaluation results to PDF
    plt.figure(figsize=(8, 6))
    plt.text(0.1, 0.8, f"Linear Regression - MSE: {mse_lr:.2f}\nR-squared: {r2_lr:.2f}", fontsize=12)
    plt.text(0.1, 0.6, f"Ridge Regression - MSE: {mse_ridge:.2f}\nR-squared: {r2_ridge:.2f}", fontsize=12)
    plt.axis('off')
    plt.title('Model Evaluation Results', fontsize=14)
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Plot predictions vs actual values for both models
    plt.figure(figsize=(12, 6))

    # Linear Regression
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_lr, color='blue', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title("Linear Regression: Predicted vs Actual")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")

    # Ridge Regression
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_ridge, color='green', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title("Ridge Regression: Predicted vs Actual")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")

    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

print("PDF saved as 'model_evaluation_output.pdf'")


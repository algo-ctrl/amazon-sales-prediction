# =============================================
# MINI PROJECT: AMAZON SALES PREDICTION MODEL
# =============================================

# ----------
# 1. PROBLEM DEFINITION
# ----------
"""
PROBLEM STATEMENT:
Predict total sales amount for Amazon orders based on product features, 
customer information, and transaction details.

BUSINESS VALUE:
Accurate sales prediction helps in:
- Inventory planning and management
- Revenue forecasting
- Identifying high-value product categories
- Optimizing marketing strategies

ML TASK TYPE: Regression (predicting continuous value - Total Sales)
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)
import joblib

# ----------
# 2. DATA PREPARATION
# ----------
print("\n=== DATA PREPARATION ===")

# Load dataset with relative path
file_path = "amazon_sales_data_8000.csv"
df = pd.read_csv(file_path)
print(f"\nInitial data shape: {df.shape}")
print("\nSample data:")
print(df.head(3))

# Feature engineering
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.weekday

# Drop non-predictive columns
df = df.drop(['Order ID', 'Customer Name', 'Date'], axis=1)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Product', 'Category', 'Customer Location', 'Payment Method', 'Status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Handle missing values if any
initial_rows = df.shape[0]
df = df.dropna()
print(f"\nRemoved {initial_rows - df.shape[0]} rows with missing values")
print(f"\nFinal dataset size: {df.shape[0]} records")

# ----------
# 3. EXPLORATORY DATA ANALYSIS
# ----------
print("\n=== EXPLORATORY DATA ANALYSIS ===")

# Correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 10))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap (8000 records dataset)')
plt.tight_layout()
plt.show()

# Sales distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Total Sales'], bins=30, kde=True)
plt.title('Distribution of Total Sales (8000 records)')
plt.show()

# ----------
# 4. MODEL BUILDING
# ----------
print("\n=== MODEL BUILDING ===")

# Prepare data
X = df.drop('Total Sales', axis=1)
y = df['Total Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train model
model = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    max_depth=12,
    min_samples_split=5,
    n_jobs=-1
)
print("\nTraining model on larger dataset...")
model.fit(X_train_scaled, y_train)

# ----------
# 5. MODEL EVALUATION
# ----------
print("\n=== MODEL EVALUATION ===")

# Predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
metrics = {
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'MAE': mean_absolute_error(y_test, y_pred),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred),
    'R²': r2_score(y_test, y_pred),
    'Explained Variance': explained_variance_score(y_test, y_pred)
}

# Display metrics
print("\nModel Performance Metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Visual evaluation
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Actual vs Predicted
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax[0])
ax[0].plot([y.min(), y.max()], [y.min(), y.max()], '--r')
ax[0].set_title('Actual vs Predicted Sales (8000 records)')
ax[0].set_xlabel('Actual Sales ($)')
ax[0].set_ylabel('Predicted Sales ($)')

# Residual plot
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, ax=ax[1])
ax[1].set_title('Distribution of Residuals (8000 records)')
ax[1].set_xlabel('Prediction Error ($)')

plt.tight_layout()
plt.show()

# Feature importance
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances)
plt.title('Feature Importances (8000 records)')
plt.tight_layout()
plt.show()

# ----------
# 6. RESULTS INTERPRETATION
# ----------
print("\n=== RESULTS INTERPRETATION ===")

print("\nKey Findings:")
print("1. The model achieves good predictive performance with R² of {:.2f}".format(metrics['R²']))
print("2. Most significant features influencing sales:")
for i, row in importances.head(5).iterrows():
    print(f"   - {row['Feature']} (importance: {row['Importance']:.3f})")
print("3. Average prediction error: ${:.2f} (MAE)".format(metrics['MAE']))
print("4. The model explains {:.1f}% of sales variance".format(metrics['Explained Variance']*100))

print("\nBusiness Recommendations:")
print("1. Focus inventory planning on high-impact product categories identified")
print("2. Optimize pricing strategy based on price-sales relationships")
print("3. Tailor marketing efforts to customer locations with highest sales potential")
print("4. Monitor payment methods that correlate with higher sales")

# Save model artifacts (relative paths)
joblib.dump(model, 'amazon_sales_predictor_8000.pkl')
joblib.dump(scaler, 'scaler_8000.pkl')
joblib.dump(label_encoders, 'label_encoders_8000.pkl')

print("\nModel artifacts saved for production use.")

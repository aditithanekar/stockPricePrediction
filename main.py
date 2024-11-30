import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
import time
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('/Users/keerthi/CS160/stockPricePredictors/all_stocks_5yr.csv')

# Exploratory Data Analysis
plt.figure(figsize=(15, 5))
plt.plot(df['close'])
plt.title('S&P 500', fontsize=15)
plt.ylabel('Price in dollars')
plt.show()

# Feature Engineering
df['open-close'] = df['open'] - df['close']
df['low-high'] = df['low'] - df['high']
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

# Check for class balance
plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
plt.title('Target Distribution')
plt.show()

# Select features and target
features = df[['open-close', 'low-high', 'volume']].fillna(0)  # Replace NaN values
target = df['target']

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)

# Debugging: Use a subset of data for faster testing
X_train_sample = X_train[:1000]
y_train_sample = y_train[:1000]
X_valid_sample = X_valid[:1000]
y_valid_sample = y_valid[:1000]

# XGBoost model with multi-threading
xgb = XGBClassifier(
    n_jobs=-1,  # Use all CPU cores
    max_depth=6,  # Limit depth of trees to avoid overfitting
    learning_rate=0.1,  # Step size for optimization
    n_estimators=100,  # Number of trees
    subsample=0.8,  # Use 80% of the data for each tree
    colsample_bytree=0.8,  # Use 80% of features for each tree
    objective='binary:logistic',  # Binary classification
    eval_metric='auc'  # Evaluation metric
)

# Train and evaluate models
startTime = time.time()

xgb.fit(X_train_sample, y_train_sample)
train_accuracy = metrics.roc_auc_score(y_train_sample, xgb.predict_proba(X_train_sample)[:, 1])
valid_accuracy = metrics.roc_auc_score(y_valid_sample, xgb.predict_proba(X_valid_sample)[:, 1])

endTime = time.time()

print(f"Training Accuracy (AUC): {train_accuracy}")
print(f"Validation Accuracy (AUC): {valid_accuracy}")
print(f"Elapsed time: {endTime - startTime:.5f} seconds")

# Plot confusion matrix for XGBoost
ConfusionMatrixDisplay.from_estimator(xgb, X_valid_sample, y_valid_sample)
plt.show()

# Display data head
print(df.head())

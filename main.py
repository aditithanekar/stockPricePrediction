# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
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
features = df[['open-close', 'low-high', 'volume']]
target = df['target']

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)

# Assuming X_train and y_train have been defined earlier in the script
# Select the first 5 samples for debugging
X_train_sample = X_train[:1000]
y_train_sample = y_train[:1000]
X_valid_sample = X_valid[:1000]
y_valid_sample = y_valid[:1000]

# Train and evaluate models
models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True),
    XGBClassifier()
]
X_train_sample = np.nan_to_num(X_train_sample, nan=0)

startTime = time.time()

for model in models:
    model.fit(X_train_sample, y_train_sample)
    train_accuracy = metrics.roc_auc_score(y_train_sample, model.predict_proba(X_train_sample)[:, 1])
    valid_accuracy = metrics.roc_auc_score(y_valid_sample, model.predict_proba(X_valid_sample)[:, 1])
    print(f"{model}:\nTraining Accuracy: {train_accuracy}\nValidation Accuracy: {valid_accuracy}\n")

endTime = time.time()
elapseTime = endTime-startTime
print(f"Elapsed time: {elapseTime:.5f} seconds")

# Plot confusion matrix for the best model (example: Logistic Regression)
from sklearn.metrics import ConfusionMatrixDisplay
best_model = models[0]  # Replace with the best-performing model if needed
ConfusionMatrixDisplay.from_estimator(best_model, X_valid_sample, y_valid_sample)
plt.show()

print(df.head())


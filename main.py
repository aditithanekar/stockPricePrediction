# Import necessary GPU-accelerated libraries
import cupy as cp
import cudf
from cuml.linear_model import LogisticRegression
from cuml.svm import SVC
from xgboost import XGBClassifier
from cuml.preprocessing import StandardScaler
from cuml import metrics
from cuml.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')

# Load dataset using cuDF
df = cudf.read_csv('all_stocks_5yr.csv')

# Exploratory Data Analysis (visualization remains on CPU)
plt.figure(figsize=(15, 5))
plt.plot(df['close'].to_pandas())
plt.title('S&P 500', fontsize=15)
plt.ylabel('Price in dollars')
plt.show()

# Feature Engineering
df['open-close'] = df['open'] - df['close']
df['low-high'] = df['low'] - df['high']
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Check for class balance (still visualized on CPU)
plt.pie(df['target'].value_counts().to_pandas().values, labels=[0, 1], autopct='%1.1f%%')
plt.title('Target Distribution')
plt.show()

# Select features and target
features = df[['open-close', 'low-high', 'volume']]
target = df['target']

# Normalize features using GPU-accelerated StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convert features and target to CuPy arrays for processing
features = cp.array(features)
target = cp.array(target.to_pandas())

# Train-test split (done on GPU)
from cuml.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)

# Select a smaller sample for debugging
X_train_sample = X_train[:1000]
y_train_sample = y_train[:1000]
X_valid_sample = X_valid[:1000]
y_valid_sample = y_valid[:1000]

# Train and evaluate models
models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True),
    XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
]

start_time = time.time()

for model in models:
    model.fit(X_train_sample, y_train_sample)
    train_accuracy = roc_auc_score(y_train_sample, model.predict_proba(X_train_sample)[:, 1])
    valid_accuracy = roc_auc_score(y_valid_sample, model.predict_proba(X_valid_sample)[:, 1])
    print(f"{model}:\nTraining Accuracy: {train_accuracy}\nValidation Accuracy: {valid_accuracy}\n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.5f} seconds")

# Plot confusion matrix for the best model (example: Logistic Regression)
from sklearn.metrics import ConfusionMatrixDisplay
best_model = models[0]  # Replace with the best-performing model if needed
ConfusionMatrixDisplay.from_estimator(best_model, cp.asnumpy(X_valid_sample), cp.asnumpy(y_valid_sample))
plt.show()

print(df.head().to_pandas())

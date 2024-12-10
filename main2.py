
#when running this on only one of the kinds of models(Logistic regression, it is much faster) 
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
from multiprocessing import Pool
import warnings
import time
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    df = pd.read_csv('/Users/aditithanekar/Documents/GitHub/stockPricePredictor/stockPricePrediction/archive (1)/all_stocks_5yr.csv')
    df['open-close'] = df['open'] - df['close']
    df['low-high'] = df['low'] - df['high']
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    return df

def feature_target_split(df):
    features = ['open-close', 'low-high', 'volume']
    target = 'target'
    return df[features], df[target]

def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def train_test_split_data(features, target):
    X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)
    return X_train, X_valid, y_train, y_valid

def process_model(X_train, y_train, X_valid, y_valid):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    train_accuracy = metrics.roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    valid_accuracy = metrics.roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    return f"LogisticRegression:\nTraining Accuracy: {train_accuracy}\nValidation Accuracy: {valid_accuracy}\n"

def main():
    df = load_and_preprocess_data()
    features, target = feature_target_split(df)
    features = normalize_features(features)
    X_train, X_valid, y_train, y_valid = train_test_split_data(features, target)
    #we used only 100 rows of data because it was taking too long for us to actually test things
    X_train_sample = X_train[:1000]
    y_train_sample = y_train[:1000]
    X_valid_sample = X_valid[:1000]
    y_valid_sample = y_valid[:1000]

    startTime = time.time()

    def wrapper():
        return process_model(X_train_sample, y_train_sample, X_valid_sample, y_valid_sample)

    with ProcessPoolExecutor(max_workers=8) as executor: # can change the number of workers here (threads)
        results = list(executor.map(wrapper, []))

    endTime = time.time()
    elapseTime = endTime - startTime
    print(f"Elapsed time: {elapseTime:.5f} seconds")

    print('\n'.join(results))

if __name__ == '__main__':
    main()

#HERE is where I created slowdown: because I usede 3 models in a for loop
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn import metrics
# from concurrent.futures import ProcessPoolExecutor
# import warnings
# import time
# import os

# warnings.filterwarnings('ignore')

# def load_and_preprocess_data():
#     df = pd.read_csv('/Users/aditithanekar/Documents/GitHub/stockPricePredictor/stockPricePrediction/archive (1)/all_stocks_5yr.csv')
#     df['open-close'] = df['open'] - df['close']
#     df['low-high'] = df['low'] - df['high']
#     df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
#     return df

# def feature_target_split(df):
#     features = ['open-close', 'low-high', 'volume']
#     target = 'target'
#     return df[features], df[target]

# def normalize_features(features):
#     scaler = StandardScaler()
#     return scaler.fit_transform(features)

# def train_test_split_data(features, target):
#     X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)
#     return X_train, X_valid, y_train, y_valid

# def process_model(model, X_train, y_train, X_valid, y_valid):
#     model.fit(X_train, y_train)
#     train_accuracy = metrics.roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
#     valid_accuracy = metrics.roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
#     return f"{type(model).__name__}:\nTraining Accuracy: {train_accuracy}\nValidation Accuracy: {valid_accuracy}\n"

# def main():
#     # print(os.cpu_count()) # check how many cores I have
#     df = load_and_preprocess_data()
#     features, target = feature_target_split(df)
#     features = normalize_features(features)
#     X_train, X_valid, y_train, y_valid = train_test_split_data(features, target)

#     X_train_sample = X_train[:1000]
#     y_train_sample = y_train[:1000]
#     X_valid_sample = X_valid[:1000]
#     y_valid_sample = y_valid[:1000]

#     models = [
#         LogisticRegression(),
#         SVC(kernel='poly', probability=True),
#         XGBClassifier()
#     ]
#     startTime = time.time()
#     #I had slow down with more workers... 1 thread = 1.98675 s, 3 threads 2.09083
#     with ProcessPoolExecutor(max_workers=2) as executor:
#         results = [executor.submit(process_model, model, X_train_sample, y_train_sample, X_valid_sample, y_valid_sample) 
#                    for model in models]
        
#         for future in results:
#             print(future.result())
        
#     endTime = time.time()
#     elapseTime = endTime - startTime
#     print(f"Elapsed time: {elapseTime:.5f} seconds")


# if __name__ == '__main__':
#     main()


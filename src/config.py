import os
import numpy as np

#Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

#Dataset configurations
DATA_PATH = os.path.join(DATA_DIR, 'customer_churn_data.csv')

#Model configurations
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2

#Hyperparameter grids for different models
MODEL_PARAMETERS = {
    'Random_Forest': {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    'Logistic_Regression': {
        'C': np.logspace(-4, 4, 10),
        'penalty': ['l2'],
        'solver': ['liblinear', 'saga', 'lbfgs']
    },
    'Gradient_Boosting': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 10],
        'subsample': [0.6, 0.8, 1.0]
    }
}
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE

def get_model(model_type='Random_Forest', params=None):
    """
    Initialize and return a model for predicting customer churn.
    Args:
        model_type (str): Type of model to initialize. Options are 'Random_Forest', 'Logistic_Regression', 'Gradient_Boosting'.
        params (dict): Hyperparameters for the model.
    """
    if model_type == 'Random_Forest':
        return RandomForestClassifier(**params, class_weight='balanced',
                                      random_state=RANDOM_STATE)
    elif model_type == 'Logistic_Regression':
        return LogisticRegression(**params, class_weight='balanced',
                                  random_state=RANDOM_STATE, max_iter=1000)
    elif model_type == 'Gradient_Boosting':
        return GradientBoostingClassifier(**params, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")
    

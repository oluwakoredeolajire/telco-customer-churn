import joblib
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import get_model
from src.config import MODEL_PARAMETERS, DATA_PATH, RANDOM_STATE
from src.features import create_features

def train_model(model_type='Random_Forest'):
    # Load and preprocess data
    df = load_data(DATA_PATH)
    label_encoders, X_train, X_test, y_train, y_test = preprocess_data(df)

    # Create features
    X_train = create_features(X_train, label_encoders)
    X_test = create_features(X_test, label_encoders)

    # Get hyperparameter grid
    parameters = MODEL_PARAMETERS[model_type]

    # Initialize model
    model = get_model(model_type=model_type, params={})

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(param_distributions=parameters,
                                        estimator=model,
                                        n_iter=80,
                                        cv=5,
                                        random_state=RANDOM_STATE,
                                        n_jobs=-1)
    
    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Best model from random search
    best_model = random_search.best_estimator_
    print(f"Best hyperparameters for {model_type}: {random_search.best_params_}")

    # Save the trained model
    joblib.dump(best_model, f'results/{model_type}_model.pkl')
    joblib.dump(label_encoders, 'results/label_encoders.pkl')
    print(f"{model_type} model trained and saved successfully.")


from src.train import train_model
from src.evaluate import evaluate_model

def run_experiment(model_type='Random_Forest'):
    """Run training and evaluation for the specified model type.
    Args:
        model_type (str): Type of model to run experiment on. Options are 'Random_Forest', 'Logistic_Regression', 'Gradient_Boosting'.
    """
    if model_type == 'Random_Forest':
        print("Training Random Forest model...")
        train_model(model_type='Random_Forest')
        print("Evaluating Random Forest model...")
        evaluate_model(model_type='Random_Forest')

    elif model_type == 'Logistic_Regression':
        print("Training Logistic Regression model...")
        train_model(model_type='Logistic_Regression')
        print("Evaluating Logistic Regression model...")
        evaluate_model(model_type='Logistic_Regression')
    
    elif model_type == 'Gradient_Boosting':
        print("Training Gradient Boosting model...")
        train_model(model_type='Gradient_Boosting')
        print("Evaluating Gradient Boosting model...")
        evaluate_model(model_type='Gradient_Boosting')
    
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")

if __name__ == "__main__":
    # Example: Run experiment for Random Forest model
    run_experiment(model_type='Random_Forest')

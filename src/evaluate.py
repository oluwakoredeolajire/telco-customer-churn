import joblib
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.features import create_features
from src.config import DATA_PATH
from src.utils import save_evaluation_metrics, plot_roc_curve, plot_confusion_matrix

def evaluate_model(model_type='Random_Forest'):
    # Load and preprocess data
    df = load_data(DATA_PATH)
    label_encoders, X_train, X_test, y_train, y_test = preprocess_data(df)

    # Create features
    X_test = create_features(X_test, label_encoders)

    # Load the trained model
    model = joblib.load(f'results/{model_type}_model.pkl')

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Prepare metrics dictionary
    metrics = {
        'model_type': model_type,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'roc_auc': roc_auc
    }

    # Save evaluation metrics
    save_evaluation_metrics(metrics, 'results/metrics.json')

    # Print evaluation results
    print(f"{model_type} Accuracy: {accuracy:.4f}")
    print(f"{model_type} Confusion Matrix:")
    print(cm)
    print(f"{model_type} Classification Report:")
    print(report)
    print(f"{model_type} ROC AUC Score: {roc_auc:.4f}")

    # Plot and save ROC curve
    plot_roc_curve(y_test, y_prob, model_type, f'results/figures/{model_type}_roc_curve.png')
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, model_type, f'results/figures/{model_type}_confusion_matrix.png')


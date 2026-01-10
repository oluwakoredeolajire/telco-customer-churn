# Experiment Results
## Model Evaluation Overview
In this experiment, three machine learning models for predicting customer churn were evaluated: **Random Forest Classifier, Logistic Regression, and Gradient Boosting Classifier.**

The evaluation was based on several metrics:
- Accuracy
- ROC AUC
- Classification Report
- Confusion Matrix


### 1. Random Forest Classifier
- Best hyperparameters for Random_Forest: {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None, 'bootstrap': True}
- Accuracy: 0.7913
- ROC AUC: 0.8364
- Precision and Recall:
    - Precision for class 0: 0.82, Precision for class 1: 0.67
    - Recall for class 0: 0.92, Recall for class 1: 0.42
- Confusion Matrix:
    - True Negatives (TN): 958
    - False Positives (FP): 78
    - False Negatives (FN): 216
    - True Positives (TP): 157
- **Analysis:**
    - Random Forest performs reasonably well with an accuracy of 79.13%. It has a strong recall for predicting non-churned customers (92%) but struggles with predicting churned customers (42%). This is reflected in the relatively low recall for class 1 (churned). The ROC AUC score is decent (0.8364), but there is room for improvement in predicting the minority class (churned customers).


### 2. Logistic Regression
- Best hyperparameters for Logistic_Regression: {'solver': 'saga', 'penalty': 'l2', 'C': np.float64(0.005994842503189409)}
- Accuracy: 0.7559
- ROC AUC: 0.8595
- Precision and Recall:
    - Precision for class 0: 0.92, Precision for class 1: 0.52
    - Recall for class 0: 0.73, Recall for class 1: 0.83
- Confusion Matrix:
    - True Negatives (TN): 757
    - False Positives (FP): 279
    - False Negatives (FN): 65
    - True Positives (TP): 308
- **Analysis:**
    -Logistic Regression has a slightly lower accuracy (75.59%) but outperforms the Random Forest model in terms of recall for churned customers (83%). However, it has lower precision for class 1 (churned customers), meaning it is more prone to false positives. Its ROC AUC is the highest of the three models (0.8595), indicating a strong overall performance in distinguishing between the classes.


### 3. Gradient Boosting Classifier
- Best hyperparameters for Gradient_Boosting: {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05}
- Accuracy: 0.8084
- ROC AUC: 0.8578
- Precision and Recall:
    - Precision for class 0: 0.84, Precision for class 1: 0.69
    - Recall for class 0: 0.92, Recall for class 1: 0.51
- Confusion Matrix:
    - True Negatives (TN): 950
    - False Positives (FP): 86
    - False Negatives (FN): 184
    - True Positives (TP): 189
- **Analysis:**
    - Graident Boosting Classifier performs the best in terms of accuracy (80.84%), but similar to Random Forest, it has a relatively low recall for churned customers (51%). It has a strong precision for non-churned customers (84%), but the recall for the minority class (churned) could be improved. Its ROC AUC score is close to that of Logistic Regression, indicating it also does well at distinguishing between churned and non-churned customers.


## Conclusion
- **Best Performing Model:** Gradient Boosting Classifier with an accuracy of 80.84% and ROC AUC of 0.8578. It strikes a good balance between predicting both classes, with strong precision for non-churned customers and reasonable accuracy overall.
- **Second Best:** Logistic Regression with an accuracy of 75.59% and the highest ROC AUC of 0.8595. Despite its slightly lower accuracy, its higher recall for churned customers (83%) makes it the second-best model, particularly when prioritizing the identification of churned customers.
- **Third Best:** Random Forest Classifier with an accuracy of 79.13% and ROC AUC of 0.8364. It performs well on non-churned customers but struggles to identify churned customers, making it the least effective model overall for this particular dataset.

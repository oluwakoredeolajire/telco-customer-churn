# Telco Customer Churn Prediction

## Problem Statement
Predicting customer churn is crucial for businesses to improve retention strategies. This project focuses on predicting which customers are most likely to churn in a telecommunications company.

## Dataset Description
The dataset contains information about customers, including contract types, tenure, payment methods, and services used. The target variable is whether a customer has churned or not (`Churn`).

-----

## Setup
1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Place the dataset CSV in `data/`.

-----

## Usage
Run the full pipeline with:
python experiments/run_experiment.py

**The script will:**
- Load and preprocess the data.
- Create features.
- Train models.
- Evaluate the models and save metrics.

**Results are saved in results/:**
- metrics.json → accuracy, confusion matrix, classification report
- figures/ → confusion matrix and roc curve plots

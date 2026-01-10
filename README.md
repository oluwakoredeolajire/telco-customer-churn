# Telco Customer Churn Prediction

## Problem Statement
Customer churn is a significant challenge for businesses in the telecommunications industry, as retaining existing customers is often more cost-effective than acquiring new ones. High churn rates can lead to reduced revenue and increased marketing costs. The goal of this project is to build a machine learning model that accurately predicts whether a customer will churn (i.e., stop using the service) based on historical customer data.

## Dataset Description
The dataset contains information about customers, including contract types, tenure, payment methods, and services used. The target variable is whether a customer has churned or not (`Churn`).

-----

## Setup
1. Clone the repository:
    - git clone https://github.com/oluwakoredeolajire/telco-customer-churn.git
    - cd telco-customer-churn

2. Create a virtual environment and activate it:
    - python3 -m venv env
    - env\Scripts\activate [On Windows]

3. Install dependencies:
    - pip install -r requirements.txt

4. Place the dataset CSV in `data/`.

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
- metrics.json → accuracy, confusion matrix, classification report, roc_auc
- figures/ → confusion matrix and roc curve plots
- For a detailed analysis, see [Experiment Results](results.md).

-----

## License
This project is licensed under the MIT License. See the LICENSE file for details.
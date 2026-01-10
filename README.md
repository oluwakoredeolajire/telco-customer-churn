# Telco Customer Churn Prediction

## Problem Statement
Predicting customer churn is crucial for businesses to improve retention strategies. This project focuses on predicting which customers are most likely to churn in a telecommunications company.

## Dataset Description
The dataset contains information about customers, including contract types, tenure, payment methods, and services used. The target variable is whether a customer has churned or not (`Churn`).

-----

## Project Structure
telco-customer-churn/
├── README.md
├── requirements.txt
├── results.md
├── data/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── experiments/
│   ├── __init__.py
│   ├── eda.ipynb
│   └── run_experiment.py
└── results/
    ├── metrics.json
    └── figures/

-----

## Setup
1. Clone the repository:
    git clone https://github.com/oluwakoredeolajire/telco-customer-churn.git
    cd telco-customer-churn

2. Create a virtual environment and activate it:
    python3 -m venv env
    env\Scripts\activate [On Windows]

3. Install dependencies:
    pip install -r requirements.txt

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
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_features(df, label_encoders=None):
    """
    Create new features for the customer churn dataset based on insights from EDA.
    This function adds features such as tenure groups, high monthly charges indicator,
    and electronic check payment method indicator. It also standardizes numerical features.

    Parameters:
    -----------
    df (pd.DataFrame): The preprocessed customer churn dataframe.
    label_encoders (dict, optional): Dictionary of label encoders from preprocessing.
    """

    # Tenure-related features
    df['tenure_group'] = pd.cut(df['tenure'], 
                               bins=[0, 12, 24, 36, 48, 60, 72], 
                               labels=[0, 1, 2, 3, 4, 5], # Changed to 0-4 for 5 bins
                               include_lowest=True).astype(int)
    df['tenure_years'] = df['tenure'] / 12
    
    # Charge-related features
    df['high_monthly_charges'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

    # Electronic check payment method feature
    if label_encoders is not None and 'PaymentMethod' in df.columns:
        payment_le = label_encoders['PaymentMethod'] # Get the label encoder for PaymentMethod
        try:
            electronic_code = list(payment_le.classes_).index('Electronic check')
            df['electronic_check'] = (df['PaymentMethod'] == electronic_code).astype(int)
        except ValueError:
            df['electronic_check'] = 0

    # Standardize numerical features
    numeric_columns = df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

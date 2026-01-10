import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.config import TARGET_COLUMN, RANDOM_STATE, TEST_SIZE

def preprocess_data(df):
    """
    Preprocess the customer churn dataset.
    This function handles missing values, encodes categorical variables, and drops unnecessary columns.
    It also splits the data into training and testing sets
    """

    # Handle missing values in 'TotalCharges'. Convert to numeric and fill NaNs with 0 because they correspond to customers with no tenure.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Encode target variable
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        if col != 'customerID':  # Exclude customerID from encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
    #Drop customerID column
    df.drop(columns=['customerID'], inplace=True)
    
    # Separate features and target variable
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN] # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    return label_encoders, X_train, X_test, y_train, y_test


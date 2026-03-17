import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

def df_basic_cleaning_and_split(data = fetch_california_housing(as_frame=True).frame,
    target_col="MedHouseVal",
    test_size=0.3,
    random_state=42):
    
    df = data.copy() 

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train test split first (prevents leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def array_standardise_data(X_train, X_test, y_train, y_test):
    # Turning the data into a numpy array 
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    # Standardisation
    standardisation = StandardScaler()
    standardisation.fit(X_train)
    X_train = standardisation.transform(X_train)

    # Parameters from the X_train used on X_test to prevent leakage.
    X_test = standardisation.transform(X_test)

    return X_train, X_test, y_train, y_test


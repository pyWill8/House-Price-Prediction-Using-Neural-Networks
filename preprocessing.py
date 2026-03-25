import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

def df_basic_cleaning_and_split(data = fetch_california_housing(as_frame=True).frame,
    target_col="MedHouseVal",
    test_size=0.3,
    random_state=42,
    additional_features = "clusters_feature.csv",
    include_split = True,
    include_additional_features = True):
    
    df = data.copy() 
    # Adding additional features
    if include_additional_features == True:
        additional_features_df = pd.read_csv(additional_features)
        df = pd.concat([df, additional_features_df], axis=1)

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train test split first (prevents leakage)
    if include_split == False: 
        return X
    
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

def df_feature_constructions(X_train, X_test):

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Ratio Features
    X_train["rooms_per_household"] = X_train["AveRooms"] / X_train["AveOccup"]
    X_test["rooms_per_household"] = X_test["AveRooms"] / X_test["AveOccup"]

    X_train["bedroom_ratio"] = X_train["AveBedrms"] / X_train["AveRooms"]
    X_test["bedroom_ratio"] = X_test["AveBedrms"] / X_test["AveRooms"]

    # Density feature
    X_train["population_density"] = X_train["Population"] / X_train["AveOccup"]
    X_test["population_density"] = X_test["Population"] / X_test["AveOccup"]

    return X_train, X_test

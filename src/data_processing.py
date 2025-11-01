"""
data processing module for ibadan house price prediction
handles data loading, cleaning, and basic exploratory data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

def load_dataset(file_path):
    """
    load the ibadan housing dataset from csv file
    
    parameters:
    file_path (str): path to the csv file
    
    returns:
    pd.DataFrame: loaded dataset
    """
    try:
        data = pd.read_csv(file_path)
        print(f"dataset loaded successfully with {len(data)} records")
        return data
    except FileNotFoundError:
        print(f"error: file {file_path} not found")
        return None
    except Exception as e:
        print(f"error loading dataset: {str(e)}")
        return None

def basic_data_info(data):
    """
    display basic information about the dataset
    
    parameters:
    data (pd.DataFrame): input dataset
    """
    print("\n=== dataset overview ===")
    print(f"shape: {data.shape}")
    print(f"columns: {list(data.columns)}")
    
    print("\n=== data types ===")
    print(data.dtypes)
    
    print("\n=== missing values ===")
    missing_vals = data.isnull().sum()
    if missing_vals.sum() > 0:
        print(missing_vals[missing_vals > 0])
    else:
        print("no missing values found")
    
    print("\n=== basic statistics ===")
    print(data.describe())

def handle_missing_values(data):
    """
    handle missing values in the dataset using appropriate strategies
    
    parameters:
    data (pd.DataFrame): input dataset with potential missing values
    
    returns:
    pd.DataFrame: dataset with missing values handled
    """
    data_cleaned = data.copy()
    
    # check for missing values
    missing_summary = data_cleaned.isnull().sum()
    
    if missing_summary.sum() == 0:
        print("no missing values to handle")
        return data_cleaned
    
    print("handling missing values...")
    
    # handle numeric columns with median imputation
    numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if data_cleaned[col].isnull().sum() > 0:
            median_val = data_cleaned[col].median()
            data_cleaned[col].fillna(median_val, inplace=True)
            print(f"filled {col} missing values with median: {median_val}")
    
    # handle categorical columns with mode imputation
    categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if data_cleaned[col].isnull().sum() > 0:
            mode_val = data_cleaned[col].mode()[0]
            data_cleaned[col].fillna(mode_val, inplace=True)
            print(f"filled {col} missing values with mode: {mode_val}")
    
    print("missing values handled successfully")
    return data_cleaned

def perform_basic_eda(data):
    """
    perform basic exploratory data analysis
    
    parameters:
    data (pd.DataFrame): cleaned dataset
    """
    print("\n=== exploratory data analysis ===")
    
    # price distribution analysis
    print(f"\nprice statistics:")
    print(f"mean price: ₦{data['price_naira'].mean():,.0f}")
    print(f"median price: ₦{data['price_naira'].median():,.0f}")
    print(f"price range: ₦{data['price_naira'].min():,.0f} - ₦{data['price_naira'].max():,.0f}")
    
    # location analysis
    print(f"\nlocation distribution:")
    location_stats = data.groupby('location').agg({
        'price_naira': ['count', 'mean', 'median']
    }).round(0)
    print(location_stats)
    
    # house type analysis
    print(f"\nhouse type distribution:")
    house_type_stats = data['house_type'].value_counts()
    print(house_type_stats)
    
    # correlation analysis for numeric features
    numeric_features = data.select_dtypes(include=[np.number]).columns
    correlation_with_price = data[numeric_features].corr()['price_naira'].sort_values(ascending=False)
    print(f"\ncorrelation with price:")
    print(correlation_with_price)

def create_train_test_split(data, test_size=0.2, random_state=42):
    """
    split the dataset into training and testing sets
    
    parameters:
    data (pd.DataFrame): processed dataset
    test_size (float): proportion of data for testing
    random_state (int): random seed for reproducibility
    
    returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    # separate features and target
    X = data.drop(['price_naira', 'id', 'is_outlier', 'outlier_reason'], axis=1, errors='ignore')  # remove target and id columns
    y = data['price_naira']
    
    # create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    print(f"\ntrain-test split completed:")
    print(f"training set: {X_train.shape[0]} samples")
    print(f"test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, output_dir='data'):
    """
    save processed training and testing data
    
    parameters:
    X_train, X_test, y_train, y_test: split datasets
    output_dir (str): directory to save processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # save training data
    train_data = X_train.copy()
    train_data['price_naira'] = y_train
    train_data.to_csv(f'{output_dir}/train_data.csv', index=False)
    
    # save test data
    test_data = X_test.copy()
    test_data['price_naira'] = y_test
    test_data.to_csv(f'{output_dir}/test_data.csv', index=False)
    
    print(f"processed data saved to {output_dir}/")

def main():
    """
    main function to execute data processing pipeline
    """
    print("starting data processing pipeline...")
    
    # load dataset
    data = load_dataset('data/ibadan_housing_prices.csv')
    if data is None:
        return
    
    # basic data information
    basic_data_info(data)
    
    # handle missing values
    data_cleaned = handle_missing_values(data)
    
    # perform eda
    perform_basic_eda(data_cleaned)
    
    # create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(data_cleaned)
    
    # save processed data
    save_processed_data(X_train, X_test, y_train, y_test)
    
    print("\ndata processing completed successfully!")

if __name__ == "__main__":
    main()
"""
feature engineering module for ibadan house price prediction
handles encoding of categorical features and scaling of numeric features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

class TargetEncoder:
    """
    custom target encoder for categorical variables
    """
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.target_means = {}
        self.global_mean = 0
        
    def fit(self, X, y):
        """fit the target encoder"""
        self.global_mean = y.mean()
        
        for column in X.columns:
            # calculate mean target for each category
            category_stats = pd.DataFrame({'category': X[column], 'target': y})
            category_means = category_stats.groupby('category')['target'].agg(['mean', 'count'])
            
            # apply smoothing
            smoothed_means = (
                (category_means['mean'] * category_means['count'] + 
                 self.global_mean * self.smoothing) / 
                (category_means['count'] + self.smoothing)
            )
            
            self.target_means[column] = smoothed_means.to_dict()
        
        return self
    
    def transform(self, X):
        """transform categorical features using target encoding"""
        X_encoded = X.copy()
        
        for column in X.columns:
            if column in self.target_means:
                X_encoded[column] = X[column].map(self.target_means[column])
                # handle unseen categories with global mean
                X_encoded[column].fillna(self.global_mean, inplace=True)
        
        return X_encoded
    
    def fit_transform(self, X, y):
        """fit and transform in one step"""
        return self.fit(X, y).transform(X)

def identify_feature_types(data):
    """
    identify numeric and categorical features in the dataset
    
    parameters:
    data (pd.DataFrame): input dataset
    
    returns:
    tuple: (numeric_features, categorical_features)
    """
    # exclude target and id columns
    features = data.drop(['price_naira', 'id', 'is_outlier', 'outlier_reason'], axis=1, errors='ignore')
    
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(include=['object']).columns.tolist()
    
    print(f"numeric features: {numeric_features}")
    print(f"categorical features: {categorical_features}")
    
    return numeric_features, categorical_features

def create_ordinal_mappings():
    """
    create ordinal mappings for categorical features with natural ordering
    
    returns:
    dict: mapping of features to their ordinal categories
    """
    ordinal_mappings = {
        'house_condition': ['Old', 'Renovated', 'Newly Built'],
        'furnishing': ['Unfurnished', 'Semi-Furnished', 'Furnished'],
        'water_supply': ['Poor', 'Irregular', 'Good'],
        'neighborhood_prestige': ['Low', 'Medium', 'High']
    }
    
    return ordinal_mappings

def create_interaction_features(data):
    """
    create meaningful interaction features
    
    parameters:
    data (pd.DataFrame): input dataset
    
    returns:
    pd.DataFrame: dataset with interaction features
    """
    data_with_interactions = data.copy()
    
    # bedrooms to bathrooms ratio
    data_with_interactions['bedroom_bathroom_ratio'] = (
        data_with_interactions['bedrooms'] / 
        (data_with_interactions['bathrooms'] + 0.1)  # avoid division by zero
    )
    
    # total rooms
    data_with_interactions['total_rooms'] = (
        data_with_interactions['bedrooms'] + data_with_interactions['bathrooms']
    )
    
    # quality score (combination of security and infrastructure)
    data_with_interactions['quality_score'] = (
        data_with_interactions['security_rating'] + 
        data_with_interactions['infrastructure_quality']
    ) / 2oladimeji@ubuntu:~/Desktop/ibadan-house-price-prediction$ conda activate base
(base) oladimeji@ubuntu:~/Desktop/ibadan-house-price-prediction$ 





    
    # location convenience score
    data_with_interactions['convenience_score'] = (
        1 / (data_with_interactions['proximity_to_main_road_km'] + 0.1) +
        1 / (data_with_interactions['distance_to_city_center_km'] + 0.1)
    )
    
    print("interaction features created successfully")
    return data_with_interactions

def build_feature_transformer(X_train, y_train):
    """
    build feature transformation pipeline
    
    parameters:
    X_train (pd.DataFrame): training features
    y_train (pd.Series): training target
    
    returns:
    ColumnTransformer: fitted feature transformer
    """
    # identify feature types
    numeric_features, categorical_features = identify_feature_types(X_train)
    
    # create ordinal mappings
    ordinal_mappings = create_ordinal_mappings()
    
    # separate ordinal and nominal categorical features
    ordinal_features = [feat for feat in categorical_features if feat in ordinal_mappings]
    nominal_features = [feat for feat in categorical_features if feat not in ordinal_mappings]
    
    print(f"ordinal features: {ordinal_features}")
    print(f"nominal features: {nominal_featureoladimeji@ubuntu:~/Desktop/ibadan-house-price-prediction$ conda activate base
(base) oladimeji@ubuntu:~/Desktop/ibadan-house-price-prediction$ 




s}")
    
    # create transformers
    transformers = []
    
    # numeric features - standardization
    if numeric_features:
        numeric_transformer = StandardScaler()
        transformers.append(('numeric', numeric_transformer, numeric_features))
    
    # ordinal features - ordinal encoding
    if ordinal_features:
        ordinal_transformer = OrdinalEncoder(
            categories=[ordinal_mappings[feat] for feat in ordinal_features],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        transformers.append(('ordinal', ordinal_transformer, ordinal_features))
    
    # nominal features - target encoding
    if nominal_features:
       
        nominal_transformer = Pipeline([
            ('label_encoder', LabelEncoder())
        ])
        
        #  LabelEncoder doesn't work well with ColumnTransformer
        # for multiple columns
    
    # create column transformer
    if len(transformers) > 0:
        feature_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
    else:
        feature_transformer = StandardScaler()  # fallback
    
    return feature_transformer

from sklearn.base import BaseEstimator, TransformerMixin

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """
    custom label encoder that works with ColumnTransformer
    """
    def __init__(self):
        self.encoders = {}
        
    def fit(self, X, y=None):
        # handle both numpy arrays and pandas dataframes
        if hasattr(X, 'columns'):
            # dataframe case
            for col in X.columns:
                encoder = LabelEncoder()
                encoder.fit(X[col].astype(str))
                self.encoders[col] = encoder
        else:
            # single column case
            encoder = LabelEncoder()
            if hasattr(X, 'values'):
                X_values = X.values.ravel()
            else:
                X_values = X.ravel()
            encoder.fit(X_values.astype(str))
            self.encoders[0] = encoder
        return self
        
    def transform(self, X):
        # handle both numpy arrays and pandas dataframes
        if hasattr(X, 'columns'):
            # dataframe case
            result = X.copy()
            for col in X.columns:
                if col in self.encoders:
                    result[col] = self.encoders[col].transform(X[col].astype(str))
            return result.values
        else:
            # single column case
            if hasattr(X, 'values'):
                X_values = X.values.ravel()
            else:
                X_values = X.ravel()
            return self.encoders[0].transform(X_values.astype(str)).reshape(-1, 1)
        
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

def create_complete_pipeline(X_train, y_train):
    """
    create complete feature engineering pipeline
    
    parameters:
    X_train (pd.DataFrame): training features
    y_train (pd.Series): training target
    
    returns:
    ColumnTransformer: complete feature engineering pipeline
    """
    # add interaction features to training data
    X_train_enhanced = create_interaction_features(X_train)
    
    # identify all features after enhancement
    numeric_features, categorical_features = identify_feature_types(X_train_enhanced)
    
    # handle categorical features with target encoding for nominal and ordinal encoding for ordinal
    ordinal_mappings = create_ordinal_mappings()
    
    # create separate transformers
    transformers = []
    
    # numeric features (including interaction features)
    all_numeric = [col for col in X_train_enhanced.columns if col not in categorical_features]
    numeric_transformer = StandardScaler()
    transformers.append(('numeric', numeric_transformer, all_numeric))
    
    # ordinal categorical features
    ordinal_features = [feat for feat in categorical_features if feat in ordinal_mappings]
    if ordinal_features:
        ordinal_transformer = OrdinalEncoder(
            categories=[ordinal_mappings[feat] for feat in ordinal_features],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        transformers.append(('ordinal', ordinal_transformer, ordinal_features))
    
    # nominal categorical features - use one hot encoding
    nominal_features = [feat for feat in categorical_features if feat not in ordinal_mappings]
    if nominal_features:
        nominal_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        transformers.append(('nominal', nominal_transformer, nominal_features))
    
    # create column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    return preprocessor

class InteractionFeatureTransformer:
    """
    custom transformer to add interaction features
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return create_interaction_features(X)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

def prepare_features_for_modeling(train_file='data/train_data.csv', test_file='data/test_data.csv'):
    """
    prepare features for modeling by applying transformations
    
    parameters:
    train_file (str): path to training data
    test_file (str): path to test data
    
    returns:
    tuple: (X_train_transformed, X_test_transformed, y_train, y_test, feature_transformer)
    """
    # load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # separate features and targets
    X_train = train_data.drop(['price_naira', 'is_outlier', 'outlier_reason'], axis=1, errors='ignore')
    y_train = train_data['price_naira']
    X_test = test_data.drop(['price_naira', 'is_outlier', 'outlier_reason'], axis=1, errors='ignore')
    y_test = test_data['price_naira']
    
    print(f"loaded training data: {X_train.shape}")
    print(f"loaded test data: {X_test.shape}")
    
    # add interaction features to both datasets
    X_train_enhanced = create_interaction_features(X_train)
    X_test_enhanced = create_interaction_features(X_test)
    
    # create and fit feature transformer
    feature_transformer = create_complete_pipeline(X_train, y_train)
    
    # fit on training data and transform both
    X_train_transformed = feature_transformer.fit_transform(X_train_enhanced)
    X_test_transformed = feature_transformer.transform(X_test_enhanced)
    
    print(f"transformed training data shape: {X_train_transformed.shape}")
    print(f"transformed test data shape: {X_test_transformed.shape}")
    
    return X_train_transformed, X_test_transformed, y_train, y_test, feature_transformer

def save_feature_transformer(transformer, filepath='models/feature_transformer.pkl'):
    """
    save the fitted feature transformer
    
    parameters:
    transformer: fitted transformer object
    filepath (str): path to save the transformer
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(transformer, filepath)
    print(f"feature transformer saved to {filepath}")

def main():
    """
    main function to execute feature engineering pipeline
    """
    print("starting feature engineering pipeline...")
    
    # prepare features
    X_train_transformed, X_test_transformed, y_train, y_test, feature_transformer = prepare_features_for_modeling()
    
    # save transformer
    save_feature_transformer(feature_transformer)
    
    # save transformed data
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train_transformed.npy', X_train_transformed)
    np.save('data/X_test_transformed.npy', X_test_transformed)
    np.save('data/y_train.npy', y_train.values)
    np.save('data/y_test.npy', y_test.values)
    
    print("feature engineering completed successfully!")
    print(f"final feature dimensions: {X_train_transformed.shape[1]} features")

if __name__ == "__main__":
    main()
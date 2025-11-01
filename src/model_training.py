"""
model training module for ibadan house price prediction
trains multiple models and selects the best performing one
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_transformed_data():
    """
    load the transformed training and test data
    
    returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        X_train = np.load('data/X_train_transformed.npy')
        X_test = np.load('data/X_test_transformed.npy')
        y_train = np.load('data/y_train.npy')
        y_test = np.load('data/y_test.npy')
        
        print(f"loaded transformed data successfully")
        print(f"training set: {X_train.shape}")
        print(f"test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    except FileNotFoundError:
        print("error: transformed data files not found. run feature engineering first.")
        return None, None, None, None

def calculate_metrics(y_true, y_pred):
    """
    calculate regression metrics
    
    parameters:
    y_true: actual values
    y_pred: predicted values
    
    returns:
    dict: dictionary containing various metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # calculate mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def perform_cross_validation(model, X, y, cv_folds=5):
    """
    perform cross validation for a model
    
    parameters:
    model: sklearn model object
    X: feature matrix
    y: target vector
    cv_folds (int): number of cross validation folds
    
    returns:
    dict: cross validation scores
    """
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # calculate cross validation scores
    cv_rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error'))
    cv_mae_scores = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    cv_r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    
    return {
        'cv_rmse_mean': cv_rmse_scores.mean(),
        'cv_rmse_std': cv_rmse_scores.std(),
        'cv_mae_mean': cv_mae_scores.mean(),
        'cv_mae_std': cv_mae_scores.std(),
        'cv_r2_mean': cv_r2_scores.mean(),
        'cv_r2_std': cv_r2_scores.std()
    }

def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    train linear regression model
    
    parameters:
    X_train, y_train: training data
    X_test, y_test: test data
    
    returns:
    tuple: (model, train_metrics, test_metrics, cv_metrics)
    """
    print("training linear regression...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    cv_metrics = perform_cross_validation(model, X_train, y_train)
    
    return model, train_metrics, test_metrics, cv_metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    train random forest model
    
    parameters:
    X_train, y_train: training data
    X_test, y_test: test data
    
    returns:
    tuple: (model, train_metrics, test_metrics, cv_metrics)
    """
    print("training random forest...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    cv_metrics = perform_cross_validation(model, X_train, y_train)
    
    return model, train_metrics, test_metrics, cv_metrics

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    train xgboost model
    
    parameters:
    X_train, y_train: training data
    X_test, y_test: test data
    
    returns:
    tuple: (model, train_metrics, test_metrics, cv_metrics)
    """
    print("training xgboost...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    cv_metrics = perform_cross_validation(model, X_train, y_train)
    
    return model, train_metrics, test_metrics, cv_metrics

def train_lightgbm(X_train, y_train, X_test, y_test):
    """
    train lightgbm model
    
    parameters:
    X_train, y_train: training data
    X_test, y_test: test data
    
    returns:
    tuple: (model, train_metrics, test_metrics, cv_metrics)
    """
    print("training lightgbm...")
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    cv_metrics = perform_cross_validation(model, X_train, y_train)
    
    return model, train_metrics, test_metrics, cv_metrics

def compare_models(model_results):
    """
    compare model performance and select the best one
    
    parameters:
    model_results (dict): dictionary containing model results
    
    returns:
    str: name of the best model
    """
    print("\n=== model comparison ===")
    
    comparison_df = pd.DataFrame()
    
    for model_name, (model, train_metrics, test_metrics, cv_metrics) in model_results.items():
        row = {
            'model': model_name,
            'train_r2': train_metrics['r2'],
            'test_r2': test_metrics['r2'],
            'cv_r2_mean': cv_metrics['cv_r2_mean'],
            'cv_r2_std': cv_metrics['cv_r2_std'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_mape': test_metrics['mape']
        }
        comparison_df = pd.concat([comparison_df, pd.DataFrame([row])], ignore_index=True)
    
    # sort by cross validation r2 score
    comparison_df = comparison_df.sort_values('cv_r2_mean', ascending=False)
    
    print(comparison_df.round(4))
    
    # select best model based on cv r2 score
    best_model_name = comparison_df.iloc[0]['model']
    print(f"\nbest model: {best_model_name}")
    
    return best_model_name, comparison_df

def save_model_and_results(best_model_name, model_results, comparison_df):
    """
    save the best model and results
    
    parameters:
    best_model_name (str): name of the best model
    model_results (dict): all model results
    comparison_df (pd.DataFrame): model comparison results
    """
    os.makedirs('models', exist_ok=True)
    
    # save best model
    best_model = model_results[best_model_name][0]
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"best model ({best_model_name}) saved to models/best_model.pkl")
    
    # save model comparison results
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    print("model comparison results saved to models/model_comparison.csv")
    
    # save detailed results
    results_summary = {}
    for model_name, (model, train_metrics, test_metrics, cv_metrics) in model_results.items():
        results_summary[model_name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_metrics': cv_metrics
        }
    
    joblib.dump(results_summary, 'models/detailed_results.pkl')
    print("detailed results saved to models/detailed_results.pkl")

def get_feature_importance(best_model_name, model_results):
    """
    extract feature importance from the best model
    
    parameters:
    best_model_name (str): name of the best model
    model_results (dict): all model results
    
    returns:
    np.array: feature importance scores
    """
    best_model = model_results[best_model_name][0]
    
    if hasattr(best_model, 'feature_importances_'):
        return best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        return np.abs(best_model.coef_)
    else:
        return None

def main():
    """
    main function to execute model training pipeline
    """
    print("starting model training pipeline...")
    
    # load transformed data
    X_train, X_test, y_train, y_test = load_transformed_data()
    if X_train is None:
        return
    
    # train all models
    model_results = {}
    
    # linear regression
    lr_model, lr_train, lr_test, lr_cv = train_linear_regression(X_train, y_train, X_test, y_test)
    model_results['linear_regression'] = (lr_model, lr_train, lr_test, lr_cv)
    
    # random forest
    rf_model, rf_train, rf_test, rf_cv = train_random_forest(X_train, y_train, X_test, y_test)
    model_results['random_forest'] = (rf_model, rf_train, rf_test, rf_cv)
    
    # xgboost
    xgb_model, xgb_train, xgb_test, xgb_cv = train_xgboost(X_train, y_train, X_test, y_test)
    model_results['xgboost'] = (xgb_model, xgb_train, xgb_test, xgb_cv)
    
    # lightgbm
    lgb_model, lgb_train, lgb_test, lgb_cv = train_lightgbm(X_train, y_train, X_test, y_test)
    model_results['lightgbm'] = (lgb_model, lgb_train, lgb_test, lgb_cv)
    
    # compare models and select best
    best_model_name, comparison_df = compare_models(model_results)
    
    # save results
    save_model_and_results(best_model_name, model_results, comparison_df)
    
    # save feature importance
    feature_importance = get_feature_importance(best_model_name, model_results)
    if feature_importance is not None:
        np.save('models/feature_importance.npy', feature_importance)
        print("feature importance saved to models/feature_importance.npy")
    
    print("\nmodel training completed successfully!")
    
    # print final results
    best_results = model_results[best_model_name]
    print(f"\nfinal model performance ({best_model_name}):")
    print(f"test r2 score: {best_results[2]['r2']:.4f}")
    print(f"test rmse: ₦{best_results[2]['rmse']:,.0f}")
    print(f"test mae: ₦{best_results[2]['mae']:,.0f}")

if __name__ == "__main__":
    main()
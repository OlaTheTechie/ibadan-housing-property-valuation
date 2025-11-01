"""
predictor module for ibadan house price prediction
loads trained model and provides prediction functionality
"""

import numpy as np
import pandas as pd
import joblib
import os
from scipy import stats
# import sys
# sys.path.append('src')
# from feature_engineering import CustomLabelEncoder

class HousePricePredictor:
    """
    house price predictor class that loads trained model and transformer
    """
    
    def __init__(self, model_path='models/best_model.pkl', 
                 transformer_path='models/feature_transformer.pkl'):
        """
        initialize the predictor
        
        parameters:
        model_path (str): path to saved model
        transformer_path (str): path to saved feature transformer
        """
        self.model = None
        self.transformer = None
        self.feature_importance = None
        self.model_loaded = False
        
        self.load_model(model_path)
        self.load_transformer(transformer_path)
        self.load_feature_importance()
    
    def load_model(self, model_path):
        """
        load the trained model
        
        parameters:
        model_path (str): path to saved model
        """
        try:
            self.model = joblib.load(model_path)
            print(f"model loaded successfully from {model_path}")
            self.model_loaded = True
        except FileNotFoundError:
            print(f"error: model file {model_path} not found")
            self.model_loaded = False
        except Exception as e:
            print(f"error loading model: {str(e)}")
            self.model_loaded = False
    
    def load_transformer(self, transformer_path):
        """
        load the feature transformer
        
        parameters:
        transformer_path (str): path to saved transformer
        """
        try:
            self.transformer = joblib.load(transformer_path)
            print(f"transformer loaded successfully from {transformer_path}")
        except FileNotFoundError:
            print(f"error: transformer file {transformer_path} not found")
        except Exception as e:
            print(f"error loading transformer: {str(e)}")
    
    def load_feature_importance(self):
        """
        load feature importance scores
        """
        try:
            self.feature_importance = np.load('models/feature_importance.npy')
            print("feature importance loaded successfully")
        except FileNotFoundError:
            print("feature importance file not found")
        except Exception as e:
            print(f"error loading feature importance: {str(e)}")
    
    def prepare_input_data(self, input_dict):
        """
        prepare input data for prediction
        
        parameters:
        input_dict (dict): dictionary containing input features
        
        returns:
        pd.DataFrame: prepared input dataframe with interaction features
        """
        # create dataframe from input
        input_df = pd.DataFrame([input_dict])
        
        # ensure all required columns are present with default values if missing
        required_columns = [
            'location', 'latitude', 'longitude', 'area_sqm', 'bedrooms',
            'bathrooms', 'toilets', 'stories', 'house_type', 'furnishing',
            'condition', 'parking_spaces', 'distance_to_city_center_km', 
            'proximity_to_main_road_km', 'security_rating',
            'infrastructure_quality', 'electricity_stability', 'water_supply',
            'neighborhood_prestige', 'desirability_score'
        ]
        
        for col in required_columns:
            if col not in input_df.columns:
                # set reasonable defaults
                if col in ['bedrooms', 'bathrooms', 'toilets']:
                    input_df[col] = 3
                elif col in ['parking_spaces']:
                    input_df[col] = 1
                elif col == 'stories':
                    input_df[col] = 1
                elif col in ['security_rating', 'infrastructure_quality', 'electricity_stability', 'water_supply']:
                    input_df[col] = 7.0
                elif col in ['proximity_to_main_road_km', 'distance_to_city_center_km']:
                    input_df[col] = 2.0
                elif col == 'area_sqm':
                    input_df[col] = 200  # default area
                elif col == 'furnishing':
                    input_df[col] = 'Semi-Furnished'
                elif col == 'condition':
                    input_df[col] = 'Renovated'
                elif col == 'neighborhood_prestige':
                    input_df[col] = 4
                elif col == 'desirability_score':
                    input_df[col] = 4.0
                else:
                    input_df[col] = 0
        
        # add interaction features
        input_df = self.create_interaction_features(input_df)
        
        return input_df
    
    def create_interaction_features(self, data):
        """
        create interaction features for input data
        
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
        ) / 2
        
        # location convenience score
        data_with_interactions['convenience_score'] = (
            1 / (data_with_interactions['proximity_to_main_road_km'] + 0.1) +
            1 / (data_with_interactions['distance_to_city_center_km'] + 0.1)
        )
        
        return data_with_interactions
    
    def predict_price(self, input_dict):
        """
        predict house price for given input
        
        parameters:
        input_dict (dict): dictionary containing house features
        
        returns:
        dict: prediction results including price and confidence interval
        """
        if not self.model_loaded or self.transformer is None:
            return {
                'error': 'model or transformer not loaded properly',
                'predicted_price': None,
                'confidence_interval': None
            }
        
        try:
            # prepare input data
            input_df = self.prepare_input_data(input_dict)
            
            # transform features
            input_transformed = self.transformer.transform(input_df)
            
            # make prediction
            predicted_price = self.model.predict(input_transformed)[0]
            
            # calculate confidence interval (approximate)
            confidence_interval = self.calculate_confidence_interval(
                input_transformed, predicted_price
            )
            
            return {
                'predicted_price': max(0, predicted_price),  # ensure non-negative
                'confidence_interval': confidence_interval,
                'error': None
            }
            
        except Exception as e:
            return {
                'error': f'prediction error: {str(e)}',
                'predicted_price': None,
                'confidence_interval': None
            }
    
    def calculate_confidence_interval(self, input_transformed, predicted_price, confidence=0.95):
        """
        calculate approximate confidence interval for prediction
        
        parameters:
        input_transformed: transformed input features
        predicted_price: predicted price
        confidence (float): confidence level
        
        returns:
        tuple: (lower_bound, upper_bound)
        """
        try:
            # simple approach: use a percentage of the predicted price
            # in production, you might want to use more sophisticated methods
            uncertainty_factor = 0.15  # 15% uncertainty
            
            margin = predicted_price * uncertainty_factor
            
            # calculate bounds
            alpha = 1 - confidence
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower_bound = predicted_price - z_score * margin
            upper_bound = predicted_price + z_score * margin
            
            return (max(0, lower_bound), upper_bound)
            
        except Exception:
            # fallback to simple percentage bounds
            margin = predicted_price * 0.2
            return (max(0, predicted_price - margin), predicted_price + margin)
    
    def get_feature_importance_dict(self):
        """
        get feature importance as dictionary
        
        returns:
        dict: feature names and their importance scores
        """
        if self.feature_importance is None:
            return None
        
        # note: this is a simplified version
        # in practice, you'd need to map transformed feature indices back to original names
        feature_names = [
            'bedrooms', 'bathrooms', 'toilets', 'parking_space',
            'proximity_to_main_road_km', 'distance_to_city_center_km',
            'security_rating', 'infrastructure_quality', 'electricity_stability',
            'noise_level', 'bedroom_bathroom_ratio', 'total_rooms',
            'quality_score', 'convenience_score'
        ]
        
        # ensure we don't exceed available importance scores
        n_features = min(len(feature_names), len(self.feature_importance))
        
        importance_dict = {}
        for i in range(n_features):
            importance_dict[feature_names[i]] = self.feature_importance[i]
        
        # sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def predict_batch(self, input_list):
        """
        predict prices for multiple houses
        
        parameters:
        input_list (list): list of input dictionaries
        
        returns:
        list: list of prediction results
        """
        results = []
        for input_dict in input_list:
            result = self.predict_price(input_dict)
            results.append(result)
        
        return results

def load_model_performance():
    """
    load model performance metrics
    
    returns:
    dict: model performance metrics
    """
    try:
        comparison_df = pd.read_csv('models/model_comparison.csv')
        best_model_row = comparison_df.iloc[0]
        
        performance = {
            'model_name': best_model_row['model'],
            'r2_score': best_model_row['test_r2'],
            'rmse': best_model_row['test_rmse'],
            'mae': best_model_row['test_mae'],
            'mape': best_model_row['test_mape']
        }
        
        return performance
        
    except FileNotFoundError:
        return {
            'model_name': 'unknown',
            'r2_score': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'mape': 0.0
        }

def create_sample_prediction():
    """
    create a sample prediction for testing
    
    returns:
    dict: sample prediction result
    """
    predictor = HousePricePredictor()
    
    sample_input = {
        'location': 'Bodija',
        'latitude': 7.4352,
        'longitude': 3.9133,
        'area_sqm': 350,
        'bedrooms': 4,
        'bathrooms': 3,
        'toilets': 4,
        'stories': 2,
        'house_type': 'Duplex',
        'furnishing': 'Furnished',
        'condition': 'New',
        'parking_spaces': 2,
        'distance_to_city_center_km': 8.5,
        'proximity_to_main_road_km': 1.5,
        'security_rating': 8.5,
        'infrastructure_quality': 9.0,
        'electricity_stability': 8.0,
        'water_supply': 9.0,
        'neighborhood_prestige': 5,
        'desirability_score': 5.0
    }
    
    result = predictor.predict_price(sample_input)
    return result

def main():
    """
    main function for testing the predictor
    """
    print("testing house price predictor...")
    
    # create sample prediction
    result = create_sample_prediction()
    
    if result['error'] is None:
        print(f"predicted price: ₦{result['predicted_price']:,.0f}")
        print(f"confidence interval: ₦{result['confidence_interval'][0]:,.0f} - ₦{result['confidence_interval'][1]:,.0f}")
    else:
        print(f"prediction failed: {result['error']}")
    
    # load performance metrics
    performance = load_model_performance()
    print(f"\nmodel performance:")
    print(f"model: {performance['model_name']}")
    print(f"r2 score: {performance['r2_score']:.4f}")
    print(f"rmse: ₦{performance['rmse']:,.0f}")

if __name__ == "__main__":
    main()
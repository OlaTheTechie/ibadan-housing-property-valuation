# Ibadan Property Sale Price Prediction System

A comprehensive machine learning system for predicting property sale prices in Ibadan, Nigeria, built with Python and Streamlit.

## Project Overview

This system uses advanced machine learning algorithms to predict property sale prices based on comprehensive features including location, property type, size, condition, and neighborhood characteristics. The model is trained on realistic synthetic data representing current Ibadan real estate market dynamics.

## Dataset Description

The dataset contains 2,000 property listings from 10 major neighborhoods in Ibadan with realistic market-based pricing:

### Neighborhood Tiers & Price Ranges
- **High-End**: Agodi GRA (₦80M-₦220M), Iyaganku GRA (₦70M-₦200M)
- **Upper**: Jericho (₦60M-₦180M), Bodija (₦40M-₦150M)
- **Mid-Upper**: Alalubosa (₦30M-₦100M)
- **Mid**: Oluyole Estate (₦25M-₦90M)
- **Mid-Low**: Akobo (₦20M-₦70M), Samonda (₦18M-₦65M)
- **Low**: Challenge (₦10M-₦40M), Apete (₦6M-₦30M)

### Features
- **Property**: area_sqm, bedrooms, bathrooms, toilets, stories, house_type, condition, furnishing, parking_spaces
- **Location**: latitude, longitude, distance_to_city_center_km, proximity_to_main_road_km, neighborhood_prestige, desirability_score
- **Quality**: security_rating, infrastructure_quality, electricity_stability, water_supply
- **Target**: price_naira (Property sale price in Nigerian Naira)

## Model Performance

The system trains and compares multiple algorithms:
- **LightGBM** (Best Model)
- Random Forest
- XGBoost
- Linear Regression

**Achieved Performance**:
- **R² Score**: 0.9447 (Excellent)
- **RMSE**: ₦17,041,494
- **MAE**: ₦11,199,422

## Installation and Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ibadan-house-price-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data file exists**
   - Place `ibadan_houses.csv` in the `data/` directory

## How to Run

### Option 1: Complete Pipeline (Recommended)
```bash
python run_complete_system.py
```
This runs the entire pipeline automatically:
1. Generates realistic dataset with market benchmarks
2. Processes and cleans the data
3. Engineers features and creates transformations
4. Trains and selects the best model
5. Tests the prediction system

### Option 2: Manual Steps
Execute the following commands in order:

```bash
# 1. Generate Dataset
python generate_sale_price_dataset.py

# 2. Data Processing
python src/data_processing.py

# 3. Feature Engineering
python src/feature_engineering.py

# 4. Model Training
python src/model_training.py

# 5. Test Predictor
python src/predictor.py
```

### Launch Web Application
```bash
streamlit run app.py
```
- Interactive web interface for price predictions
- Real-time predictions with confidence intervals
- Model performance metrics and feature importance
- Professional UI for property valuation

## Project Structure

```
ibadan-house-price-prediction/
├── data/
│   ├── ibadan_housing_prices.csv     # realistic property dataset
│   ├── sale_price_qc_report.json     # quality control report
│   ├── train_data.csv                # processed training data
│   ├── test_data.csv                 # processed test data
│   └── *.npy                         # transformed feature arrays
├── src/
│   ├── data_processing.py            # data loading and EDA
│   ├── feature_engineering.py       # feature transformation
│   ├── model_training.py             # model training and selection
│   └── predictor.py                  # prediction functionality
├── models/
│   ├── best_model.pkl                # trained LightGBM model
│   ├── feature_transformer.pkl       # feature pipeline
│   ├── model_comparison.csv          # performance comparison
│   └── feature_importance.npy        # feature importance scores
├── generate_sale_price_dataset.py   # realistic dataset generator
├── run_complete_system.py           # complete pipeline runner
├── app.py                           # streamlit web application
├── requirements.txt                 # python dependencies
└── README.md                        # project documentation
```

## Usage

### Web Interface
1. Launch the Streamlit app using `streamlit run app.py`
2. Use the sidebar to input house features:
   - Select location from dropdown
   - Adjust bedrooms, bathrooms, and other amenities
   - Set quality ratings for security, infrastructure, etc.
3. Click "Predict Price" to get price estimate
4. View confidence intervals and model performance metrics

### Programmatic Usage
```python
from src.predictor import HousePricePredictor

# initialize predictor
predictor = HousePricePredictor()

# prepare property input
property_features = {
    'location': 'Bodija',
    'area_sqm': 350,
    'house_type': 'Duplex',
    'bedrooms': 4,
    'bathrooms': 3,
    'condition': 'New',
    'furnishing': 'Furnished',
    'stories': 2,
    'parking_spaces': 2,
    'desirability_score': 5.0,
    # ... other features
}

# get prediction
result = predictor.predict_price(property_features)
print(f"Predicted price: ₦{result['predicted_price']:,.0f}")
print(f"Confidence interval: ₦{result['confidence_interval'][0]:,.0f} - ₦{result['confidence_interval'][1]:,.0f}")
```

## Key Features

- **Multiple ML Models**: Compares Linear Regression, Random Forest, XGBoost, and LightGBM
- **Feature Engineering**: Creates interaction features and applies appropriate encoding
- **Cross Validation**: Uses 5-fold CV for robust model evaluation
- **Confidence Intervals**: Provides prediction uncertainty estimates
- **Interactive UI**: Clean, professional Streamlit interface
- **Model Interpretability**: Feature importance visualization
- **Error Handling**: Comprehensive error handling throughout the pipeline

## Technical Details

- **Target Encoding**: Used for high-cardinality categorical features
- **Ordinal Encoding**: Applied to features with natural ordering
- **Standard Scaling**: Applied to all numeric features
- **Interaction Features**: Created meaningful feature combinations
- **Model Selection**: Based on cross-validation R² scores

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact the development team.
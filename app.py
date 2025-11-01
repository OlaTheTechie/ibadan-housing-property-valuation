"""
streamlit app for ibadan house price prediction
provides user interface for house price prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# add src to path for imports
sys.path.append('src')
from predictor import HousePricePredictor, load_model_performance

# page configuration
st.set_page_config(
    page_title="Ibadan Property Sale Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom css for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .price-display {
        font-size: 3rem;
        font-weight: bold;
        color: #2e8b57;
        text-align: center;
        padding: 1rem;
        border: 2px solid #2e8b57;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-interval {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """
    load the trained model and predictor
    """
    return HousePricePredictor()

@st.cache_data
def load_performance_metrics():
    """
    load model performance metrics
    """
    return load_model_performance()

def create_sidebar_inputs():
    """
    create sidebar input widgets for property features
    
    returns:
    dict: dictionary containing user inputs
    """
    st.sidebar.header("property features")
    
    # location selection
    locations = [
        'Bodija', 'Jericho', 'Agodi GRA', 'Iyaganku GRA', 'Alalubosa',
        'Oluyole Estate', 'Challenge', 'Akobo', 'Samonda', 'Apete'
    ]
    location = st.sidebar.selectbox("location", locations)
    
    # coordinate and desirability mapping for locations
    location_data = {
        'Bodija': {'coords': (7.4352, 3.9133), 'desirability': 5.0, 'prestige': 5, 'distance': 8.5},
        'Jericho': {'coords': (7.4030, 3.8850), 'desirability': 4.8, 'prestige': 4, 'distance': 6.2},
        'Agodi GRA': {'coords': (7.4069, 3.8993), 'desirability': 4.7, 'prestige': 4, 'distance': 5.8},
        'Iyaganku GRA': {'coords': (7.3925, 3.8681), 'desirability': 4.6, 'prestige': 4, 'distance': 4.5},
        'Alalubosa': {'coords': (7.3839, 3.8617), 'desirability': 4.3, 'prestige': 4, 'distance': 3.2},
        'Oluyole Estate': {'coords': (7.3628, 3.8562), 'desirability': 4.2, 'prestige': 4, 'distance': 7.8},
        'Challenge': {'coords': (7.3383, 3.8773), 'desirability': 3.8, 'prestige': 3, 'distance': 12.4},
        'Akobo': {'coords': (7.3964, 3.9167), 'desirability': 3.5, 'prestige': 3, 'distance': 11.2},
        'Samonda': {'coords': (7.4306, 3.9081), 'desirability': 3.2, 'prestige': 3, 'distance': 9.8},
        'Apete': {'coords': (7.4492, 3.8722), 'desirability': 2.8, 'prestige': 2, 'distance': 15.7}
    }
    
    loc_data = location_data[location]
    latitude, longitude = loc_data['coords']
    
    # house type
    house_types = [
        'Detached House', 'Duplex', 'Bungalow', 'Terraced House', 'Flat', 'Mini Flat'
    ]
    house_type = st.sidebar.selectbox("house type", house_types)
    
    # basic features
    area_sqm = st.sidebar.slider("area (sqm)", 50, 600, 250)
    bedrooms = st.sidebar.slider("bedrooms", 1, 6, 3)
    bathrooms = st.sidebar.slider("bathrooms", 1, 5, 2)
    toilets = st.sidebar.slider("toilets", bathrooms, bathrooms + 2, bathrooms)
    stories = st.sidebar.slider("stories", 1, 3, 1)
    parking_spaces = st.sidebar.slider("parking spaces", 0, 4, 1)
    
    # categorical features
    furnishing_options = ['Unfurnished', 'Semi-Furnished', 'Furnished']
    furnishing = st.sidebar.selectbox("furnishing", furnishing_options)
    
    condition_options = ['Old', 'Renovated', 'New']
    condition = st.sidebar.selectbox("condition", condition_options)
    
    # location details
    st.sidebar.subheader("location details")
    proximity_to_main_road_km = st.sidebar.slider(
        "proximity to main road (km)", 0.1, 3.0, 1.5, 0.1
    )
    distance_to_city_center_km = st.sidebar.slider(
        "distance to city center (km)", 2.0, 20.0, loc_data['distance'], 0.5
    )
    
    # quality ratings
    st.sidebar.subheader("quality ratings")
    security_rating = st.sidebar.slider("security rating", 4.0, 10.0, 7.0, 0.1)
    infrastructure_quality = st.sidebar.slider("infrastructure quality", 4.0, 10.0, 7.0, 0.1)
    electricity_stability = st.sidebar.slider("electricity stability", 3.0, 9.0, 6.0, 0.1)
    water_supply = st.sidebar.slider("water supply quality", 4.0, 10.0, 7.0, 0.1)
    
    # compile inputs
    inputs = {
        'location': location,
        'latitude': latitude,
        'longitude': longitude,
        'area_sqm': area_sqm,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'toilets': toilets,
        'stories': stories,
        'house_type': house_type,
        'furnishing': furnishing,
        'condition': condition,
        'parking_spaces': parking_spaces,
        'distance_to_city_center_km': distance_to_city_center_km,
        'proximity_to_main_road_km': proximity_to_main_road_km,
        'security_rating': security_rating,
        'infrastructure_quality': infrastructure_quality,
        'electricity_stability': electricity_stability,
        'water_supply': water_supply,
        'neighborhood_prestige': loc_data['prestige'],
        'desirability_score': loc_data['desirability']
    }
    
    return inputs

def display_prediction_results(prediction_result):
    """
    display prediction results in the main area
    
    parameters:
    prediction_result (dict): prediction results from the model
    """
    if prediction_result['error'] is not None:
        st.error(f"prediction error: {prediction_result['error']}")
        return
    
    predicted_price = prediction_result['predicted_price']
    confidence_interval = prediction_result['confidence_interval']
    
    # main price display
    st.markdown(
        f'<div class="price-display">₦{predicted_price:,.0f}</div>',
        unsafe_allow_html=True
    )
    
    # confidence interval
    st.markdown(
        f'<div class="confidence-interval">'
        f'confidence interval: ₦{confidence_interval[0]:,.0f} - ₦{confidence_interval[1]:,.0f}'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # additional information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "lower bound",
            f"₦{confidence_interval[0]:,.0f}",
            f"{((confidence_interval[0] - predicted_price) / predicted_price * 100):+.1f}%"
        )
    
    with col2:
        st.metric(
            "predicted price",
            f"₦{predicted_price:,.0f}"
        )
    
    with col3:
        st.metric(
            "upper bound",
            f"₦{confidence_interval[1]:,.0f}",
            f"{((confidence_interval[1] - predicted_price) / predicted_price * 100):+.1f}%"
        )

def display_model_performance():
    """
    display model performance metrics
    """
    st.subheader("model performance")
    
    performance = load_performance_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("model type", performance['model_name'].replace('_', ' ').title())
    
    with col2:
        st.metric("r² score", f"{performance['r2_score']:.3f}")
    
    with col3:
        st.metric("rmse", f"₦{performance['rmse']:,.0f}")
    
    with col4:
        st.metric("mae", f"₦{performance['mae']:,.0f}")

def create_feature_importance_plot(predictor):
    """
    create feature importance plot
    
    parameters:
    predictor: trained predictor object
    """
    importance_dict = predictor.get_feature_importance_dict()
    
    if importance_dict is None:
        st.warning("feature importance data not available")
        return
    
    # create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(importance_dict.keys())[:10]  # top 10 features
    importances = list(importance_dict.values())[:10]
    
    bars = ax.barh(features, importances)
    ax.set_xlabel('importance score')
    ax.set_title('top 10 feature importance')
    
    # color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    st.pyplot(fig)

def main():
    """
    main streamlit app function
    """
    # app header
    st.markdown('<h1 class="main-header">ibadan property sale price predictor</h1>', unsafe_allow_html=True)
    
    # check if model files exist
    if not os.path.exists('models/best_model.pkl'):
        st.error("model files not found. please run the training pipeline first.")
        st.info("run the following commands in order:")
        st.code("""
        python src/data_processing.py
        python src/feature_engineering.py
        python src/model_training.py
        """)
        return
    
    # load predictor
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"failed to load predictor: {str(e)}")
        return
    
    # get user inputs
    user_inputs = create_sidebar_inputs()
    
    # create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("predicted sale price")
        
        # predict button
        if st.button("predict price", type="primary"):
            with st.spinner("calculating prediction..."):
                prediction_result = predictor.predict_price(user_inputs)
                display_prediction_results(prediction_result)
        
        # display sample prediction on load
        else:
            st.info("adjust the property features in the sidebar and click 'predict price' to get an estimate")
    
    with col2:
        st.subheader("property summary")
        
        # display current inputs summary
        summary_data = {
            'feature': ['location', 'house type', 'area (sqm)', 'bedrooms', 'bathrooms', 'stories', 'condition'],
            'value': [
                user_inputs['location'],
                user_inputs['house_type'],
                user_inputs['area_sqm'],
                user_inputs['bedrooms'],
                user_inputs['bathrooms'],
                user_inputs['stories'],
                user_inputs['condition']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True)
    
    # model performance section
    st.divider()
    display_model_performance()
    
    # feature importance section
    st.divider()
    st.subheader("feature importance")
    
    show_importance = st.checkbox("show feature importance plot")
    if show_importance:
        create_feature_importance_plot(predictor)
    
    # footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        ibadan property sale price prediction system | built with streamlit and scikit-learn
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
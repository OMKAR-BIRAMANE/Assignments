import streamlit as st
import requests

st.title('House Price Predictor')

st.header('Enter House Features')

# Input fields for house features
OverallQual = st.number_input('Overall Quality', min_value=1, max_value=10)
GrLivArea = st.number_input('Ground Living Area (sqft)', min_value=1)
GarageCars = st.number_input('Garage Cars', min_value=0, max_value=4)
GarageArea = st.number_input('Garage Area (sqft)', min_value=1)
TotalBsmtSF = st.number_input('Total Basement Area (sqft)', min_value=1)
FullBath = st.number_input('Full Bathrooms', min_value=0, max_value=4)
YearBuilt = st.number_input('Year Built', min_value=1800, max_value=2024)

# Prediction button
if st.button('Predict Price'):
    # API request to get predicted price
    response = requests.post(
        'http://localhost:8000/predict',
        json={
            'OverallQual': OverallQual,
            'GrLivArea': GrLivArea,
            'GarageCars': GarageCars,
            'GarageArea': GarageArea,
            'TotalBsmtSF': TotalBsmtSF,
            'FullBath': FullBath,
            'YearBuilt': YearBuilt
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        st.success(f'Predicted Price: ${result["predicted_price"]:.2f}')
    else:
        st.error('Error in prediction. Please try again.')

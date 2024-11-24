from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pydantic import BaseModel, validator
from typing import Dict, Optional
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HouseFeatures(BaseModel):
    LotArea: float
    YearBuilt: int
    TotalBsmtSF: float
    FirstFlrSF: float
    SecondFlrSF: float
    FullBath: int
    BedroomAbvGr: int
    GarageArea: float
    OverallQual: int
    GrLivArea: float
    KitchenQual: str
    GarageType: Optional[str] = None
    
    @validator('YearBuilt')
    def validate_year(cls, v):
        if v < 1800 or v > 2024:
            raise ValueError('Year must be between 1800 and 2024')
        return v
    
    @validator('OverallQual')
    def validate_quality(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Overall Quality must be between 1 and 10')
        return v

class HousePricePredictor:
    def __init__(self, data_file: str):
        """Initialize the predictor with training data"""
        try:
            self.df = pd.read_csv(data_file)
            self.label_encoders = {}
            self.model_path = 'house_price_model.joblib'
            self.scaler = StandardScaler()
            
            if os.path.exists(self.model_path):
                logger.info("Loading existing model...")
                self.load_model()
            else:
                logger.info("Training new model...")
                self.X, self.y = self.preprocess_data()
                self.train_and_evaluate_model()
                
            logger.info("Model ready for predictions!")
            
        except Exception as e:
            logger.error(f"Error initializing predictor: {str(e)}")
            raise

    def preprocess_data(self):
        """Enhanced data preprocessing"""
        try:
            # Select important numerical and categorical features
            numerical_features = [
                'LotArea', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', 
                '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'GarageArea',
                'OverallQual', 'GrLivArea'
            ]
            
            categorical_features = ['KitchenQual', 'GarageType']
            
            # Handle missing values for numerical features
            for feature in numerical_features:
                self.df[feature] = self.df[feature].fillna(self.df[feature].mean())
            
            # Handle missing values and encode categorical features
            for feature in categorical_features:
                self.df[feature] = self.df[feature].fillna('Missing')
                self.label_encoders[feature] = LabelEncoder()
                self.df[feature] = self.label_encoders[feature].fit_transform(self.df[feature])
            
            # Combine features
            features = numerical_features + categorical_features
            X = self.df[features]
            
            # Scale numerical features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=features)
            
            # Target variable
            y = self.df['SalePrice']
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train_and_evaluate_model(self):
        """Train and evaluate the model with cross-validation"""
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            # Initialize and train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            logger.info(f"Model Performance Metrics:")
            logger.info(f"RMSE: ${rmse:,.2f}")
            logger.info(f"MAE: ${mae:,.2f}")
            logger.info(f"R² Score: {r2:.4f}")
            
            # Save the model
            self.save_model()
            
        except Exception as e:
            logger.error(f"Error in training model: {str(e)}")
            raise

    def predict_price(self, features: Dict):
        """Predict house price based on input features"""
        try:
            # Convert input features to DataFrame
            input_df = pd.DataFrame([features])
            
            # Encode categorical features
            for feature, encoder in self.label_encoders.items():
                if feature in input_df.columns:
                    input_df[feature] = encoder.transform(input_df[feature])
            
            # Scale the input features
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)
            
            # Get feature importances
            feature_importance = dict(zip(input_df.columns, 
                                       self.model.feature_importances_))
            
            return {
                'predicted_price': float(prediction[0]),
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    def save_model(self):
        """Save the model and preprocessors to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_data, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the model and preprocessors from disk"""
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        logger.info("Model loaded successfully")

# Create FastAPI instance
app = FastAPI(
    title="House Price Prediction API",
    description="An API for predicting house prices using machine learning",
    version="2.0.0"
)

# Initialize the predictor
predictor = HousePricePredictor('train.csv')

@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to the House Price Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "/predict": "POST request to predict house price",
            "/model-info": "GET request for model information"
        }
    }

@app.get("/model-info")
def get_model_info():
    """Get information about the model"""
    feature_importance = dict(zip(predictor.X.columns, 
                                predictor.model.feature_importances_))
    return {
        "model_type": "Random Forest Regressor",
        "features_used": list(predictor.X.columns),
        "feature_importance": feature_importance,
        "model_parameters": predictor.model.get_params()
    }

@app.post("/predict")
def predict_house_price(features: HouseFeatures):
    """Predict house price based on input features"""
    try:
        # Convert Pydantic model to dict
        features_dict = {
            'LotArea': features.LotArea,
            'YearBuilt': features.YearBuilt,
            'TotalBsmtSF': features.TotalBsmtSF,
            '1stFlrSF': features.FirstFlrSF,
            '2ndFlrSF': features.SecondFlrSF,
            'FullBath': features.FullBath,
            'BedroomAbvGr': features.BedroomAbvGr,
            'GarageArea': features.GarageArea,
            'OverallQual': features.OverallQual,
            'GrLivArea': features.GrLivArea,
            'KitchenQual': features.KitchenQual,
            'GarageType': features.GarageType or 'Missing'
        }
        
        # Get prediction and feature importance
        result = predictor.predict_price(features_dict)
        
        return {
            "predicted_price": result['predicted_price'],
            "features_used": features_dict,
            "feature_importance": result['feature_importance']
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
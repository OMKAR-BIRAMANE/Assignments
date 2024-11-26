# predictor.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.train_model()

    def train_model(self):
        # Load the dataset
        df = pd.read_csv("house_prices.csv")
        # Features and target
        features = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "FullBath", "YearBuilt"]
        X = df[features]
        y = df["SalePrice"]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, features):
        df = pd.DataFrame([features])
        prediction = self.model.predict(df)
        return prediction[0]

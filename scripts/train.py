import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import json

def train_delivery_model(data_path: str = "../data/berlin_delivery_data.csv"):
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows")
    except FileNotFoundError:
        print("File not found")
        return None

    # weather map
    weather_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Snow": 3}
    df["weather_code"] = df["weather"].map(weather_map)

    # traffic map
    traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    df["traffic_code"] = df["traffic_level"].map(traffic_map)

    # exp map
    exp_map = {"Junior": 0, "Senior": 1, "Expert": 2}
    df["exp_code"] = df["driver_experience"].map(exp_map)
    
    df = pd.get_dummies(df, columns=["vehicle_type"], drop_first=True)
    feature_cols = [
            "distance_km", 
            "weather_code", 
            "traffic_code", 
            "exp_code", 
            "vehicle_type_Scooter", 
            "vehicle_type_Van"
        ]
    X = df[feature_cols]
    y = df["delivery_duration_mins"]
    
    print(f"Data Processed. Features: {list(X.columns)}")
    
    print("Training starts")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Testing starts")
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    print(f"Average Error: ±{error:.2f} mins")

    joblib.dump(model, "../models/delivery_model.pkl")
    print("Model saved")

    stats = {
        "mean_duration": float(y.mean()),
        "std_duration": float(y.std()),
        "description": "Baseline delivery stats from training data"
    }

    with open("../models/training_stats.json", "w") as f:
        json.dump(stats, f)
    print("Baseline stats saved")

    return model




if __name__ == "__main__":
    # Test loading
    model = train_delivery_model()
    if model is not None:
        print(model)
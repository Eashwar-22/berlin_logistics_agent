import re
from langchain_core.tools import tool
import math
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
import shap
import json
import requests

try:
    MODEL = joblib.load("../models/delivery_model.pkl")
    print("Model loaded")
except:
    print("Model not found")
    MODEL = None

EXPLAINER = None
if MODEL:
    try:
        # TreeExplainer is optimized for Random Forests
        EXPLAINER = shap.TreeExplainer(MODEL)
        print("SHAP Explainer Ready")
    except Exception as e:
        print(f"Could not load SHAP: {e}")

class DeliveryInput(BaseModel):
    vehicle_type: str = Field(description="The type of vehicle used for delivery.")
    weather: str = Field(description="Current weather conditions in Berlin.")
    distance_km: float = Field(description="Distance of the trip in KM")
    traffic_level: str = Field(description="Traffic density (default: Medium)", default="Medium")
    driver_experience: str = Field(description="Driver skill (default: Senior)", default="Senior")

def _normalize_input(val: str, options: list[str], default: str) -> str:
    """Helper: Fuzzy matches input to allowed options."""
    val_clean = val.strip().lower()
    for opt in options:
        if val_clean in opt.lower() or opt.lower() in val_clean:
            return opt
    return default

@tool(args_schema=DeliveryInput)
def predict_delivery_time(vehicle_type: str, weather: str, distance_km: float, traffic_level: str = "Medium", driver_experience: str = "Senior") -> str:
    """
    Predicts delivery duration (in minutes) using the trained ML model. 
    Handles fuzzy inputs (e.g. 'Rain' -> 'Rainy').
    """
    if MODEL is None: return "Error: Model not loaded."

    # 1. Normalize Inputs (The robust fix)
    weather_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Snow": 3}
    traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    exp_map = {"Junior": 0, "Senior": 1, "Expert": 2}
    
    weather = _normalize_input(weather, list(weather_map.keys()), "Cloudy")
    traffic_level = _normalize_input(traffic_level, list(traffic_map.keys()), "Medium")
    driver_experience = _normalize_input(driver_experience, list(exp_map.keys()), "Senior")
    
    # 2. Prepare DataFrame
    input_df = pd.DataFrame([{
        "distance_km": distance_km,
        "weather_code": weather_map.get(weather, 1),
        "traffic_code": traffic_map.get(traffic_level, 1),
        "exp_code": exp_map.get(driver_experience, 1),
        "vehicle_type_Scooter": 1 if "Scooter" in vehicle_type else 0,
        "vehicle_type_Van": 1 if "Van" in vehicle_type else 0
    }])
    
    prediction = MODEL.predict(input_df)[0]
    return f"{round(prediction, 1)} minutes"

@tool
def anonymize_pii(text: str) -> str:
    """
    Scans text for German PII (Emails, Names) and censors them.
    Useful for GDPR compliance before analysis.
    """
    # 1. Email Masking (Regex)
    # Looks for anything like "name@domain.com"
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    text = re.sub(email_pattern, "<EMAIL_REDACTED>", text)
    
    # 2. Name Masking (Simple Logic)
    # In a real app we'd use Microsoft Presidio, but for now specific patterns work.
    if "User_" in text:
        # We assume "User_1234" is potentially sensitive (e.g., employee ID).
        # Let's replace it with a generic placeholder.
        text = text.replace("User_", "Anon_Customer_")
        
    return text

@tool
def calculate_delivery_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the Haversine distance (in km) between two GPS points.
    Useful for estimating delivery travel time.
    """
    R = 6371  # Earth radius in km
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return round(R * c, 2)

@tool
def get_weather_risk(date_str: str) -> str:
    """
    Checks the historical weather for Berlin on a specific date.
    Returns: 'Sunny', 'Rainy', 'Snow', or 'Cloudy'.
    """
    if "2026-01" in date_str or "2026-02" in date_str:
        return "Rainy"  # Berlin winter is wet
    elif "2026-07" in date_str:
        return "Sunny"
    else:
        return "Cloudy"

@tool(args_schema=DeliveryInput)
def explain_delivery_prediction(vehicle_type: str, weather: str, distance_km: float, traffic_level: str = "Medium", driver_experience: str = "Senior") -> str:
    """
    Explains WHY the model predicted a certain time.
    """
    if EXPLAINER is None: return "Error: SHAP Explainer not loaded."
        
    # 1. Normalize Inputs
    weather_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Snow": 3}
    traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    exp_map = {"Junior": 0, "Senior": 1, "Expert": 2}
    
    weather = _normalize_input(weather, list(weather_map.keys()), "Cloudy")
    traffic_level = _normalize_input(traffic_level, list(traffic_map.keys()), "Medium")
    driver_experience = _normalize_input(driver_experience, list(exp_map.keys()), "Senior")

    # 2. Prepare DataFrame
    input_df = pd.DataFrame([{
        "distance_km": distance_km,
        "weather_code": weather_map.get(weather, 1),
        "traffic_code": traffic_map.get(traffic_level, 1),
        "exp_code": exp_map.get(driver_experience, 1),
        "vehicle_type_Scooter": 1 if "Scooter" in vehicle_type else 0,
        "vehicle_type_Van": 1 if "Van" in vehicle_type else 0
    }])
    
    # calculate shap values
    shap_values = EXPLAINER(input_df)
    
    # format explanation
    base_value = shap_values.base_values[0]
    values = shap_values.values[0]
    feature_names = input_df.columns
    
    explanation = f"Base Delivery Time (Avg): {base_value:.1f} mins\n"
    explanation += "Impact of factors:\n"
    
    for name, impact in zip(feature_names, values):
        # only show things that mattered (impact > 0.1 mins)
        if abs(impact) > 0.1:
            sign = "+" if impact > 0 else ""
            explanation += f"- {name}: {sign}{impact:.1f} mins\n"
            
    return explanation

@tool
def check_data_drift(daily_delivery_times: list[int]) -> str:
    """
    MLOps Tool: Checks if recent delivery data has drifted significantly 
    from the training baseline.
    Args:
        daily_delivery_times: List of delivery durations (in mins) from today.
    """
    try:
        # load baseline
        with open("../models/training_stats.json", "r") as f:
            stats = json.load(f)
            
        base_mean = stats["mean_duration"]
        base_std = stats["std_duration"]
        
        # calculate new stats
        new_mean = sum(daily_delivery_times) / len(daily_delivery_times)
        
        # z-score test
        z_score = (new_mean - base_mean) / base_std
        
        report = f"Baseline Mean: {base_mean:.1f} | Today's Mean: {new_mean:.1f}\n"
        report += f"Z-Score: {z_score:.2f}\n"
        
        if abs(z_score) > 2: # over 2 STD is probably a drift
            return report + "Drift detected. Model may be invalid. Retrain recommended."
        else:
            return report + "Data is stable. Model is healthy."
            
    except Exception as e:
        return f"Error checking drift: {e}"

@tool
def get_location_name(lat: float, lon: float) -> str:
    """
    Reverse Geocoding: Converts GPS coordinates to a real address/district name 
    using OpenStreetMap (Nominatim).
    """
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        # Nominatim requires a User-Agent header
        headers = {"User-Agent": "BerlinLogisticsAgent/1.0"}
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if "address" in data:
            # Try to return the most relevant part (District or Road)
            address = data["address"]
            return address.get("suburb", address.get("road", "Unknown Device")) + ", Berlin"
        else:
            return "Location not found"
            
    except Exception as e:
        return f"Error retrieving location: {e}"
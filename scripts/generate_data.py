import pandas as pd
import numpy as np
import random
import math
from datetime import datetime, timedelta

# Constants
NUM_ROWS = 50000
DISTRICTS = {
    "Mitte": (52.5200, 13.4050),
    "Kreuzberg": (52.4981, 13.3918),
    "Prenzlauer Berg": (52.5423, 13.4140),
    "Charlottenburg": (52.5167, 13.3000),
    "Schoneberg": (52.4822, 13.3571),
    "Friedrichshain": (52.5117, 13.4333),
    "Neukölln": (52.4800, 13.4333),
    "Pankow": (52.5667, 13.4000),
    "Lichtenberg": (52.5333, 13.5000),
    "Spandau": (52.5333, 13.1975),
    "Tempelhof": (52.4667, 13.3833),
    "Steglitz": (52.4492, 13.3217)
}
VEHICLES = ["Bike", "Scooter", "Van"]
WEATHER = ["Sunny", "Cloudy", "Rainy", "Snow"]
TRAFFIC = ["Low", "Medium", "High"]
DRIVER_EXP = ["Junior", "Senior", "Expert"]

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    R = 6371  # radius of earth in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

print(f"Generating {NUM_ROWS} rows of RICH dynamic Berlin data...")

data = []

for i in range(NUM_ROWS):
    # 1. Random Start/End Districts
    start_dist = random.choice(list(DISTRICTS.keys()))
    end_dist = random.choice(list(DISTRICTS.keys()))
    
    # 2. Add GPS Noise (Jitter)
    # +/- 0.03 degrees (approx 3km radius around center)
    start_lat = DISTRICTS[start_dist][0] + random.uniform(-0.03, 0.03)
    start_lon = DISTRICTS[start_dist][1] + random.uniform(-0.03, 0.03)
    
    end_lat = DISTRICTS[end_dist][0] + random.uniform(-0.03, 0.03)
    end_lon = DISTRICTS[end_dist][1] + random.uniform(-0.03, 0.03)
    
    # 3. Calculate Real Distance
    distance_km = haversine(start_lat, start_lon, end_lat, end_lon)
    
    # 4. Features: Time, Weather, Vehicle
    weather = random.choice(WEATHER)
    vehicle = random.choice(VEHICLES)
    
    # 5. NEW FEATURES: Traffic & Driver
    # Traffic is correlated with time of day
    random_hour = random.randint(8, 22)
    is_rush_hour = (8 <= random_hour <= 9) or (17 <= random_hour <= 19)
    
    if is_rush_hour:
        traffic = np.random.choice(["High", "Medium"], p=[0.7, 0.3])
    else:
        traffic = np.random.choice(["Low", "Medium", "High"], p=[0.5, 0.4, 0.1])
        
    driver_exp = random.choice(DRIVER_EXP)

    # 6. Calculate Duration Logic
    # Base Speed (km/h)
    if vehicle == "Bike": speed = 15
    elif vehicle == "Scooter": speed = 25
    else: speed = 30 # Van
    
    base_duration = (distance_km / speed) * 60
    
    # Apply Multipliers
    multipliers = 1.0
    
    # Weather Impact
    if weather == "Rainy": multipliers *= 1.2
    if weather == "Snow": multipliers *= 1.5
    
    # Traffic Impact
    if traffic == "Medium": multipliers *= 1.2
    if traffic == "High": multipliers *= 1.5
    
    # Driver Impact (Experts are faster)
    if driver_exp == "Junior": multipliers *= 1.1
    if driver_exp == "Expert": multipliers *= 0.9
    
    final_duration = base_duration * multipliers
    
    # Add randomness (stop lights, parking, etc)
    final_duration += random.uniform(-2, 10)
    
    # Create Row
    name = f"User_{random.randint(1000, 9999)}"
    timestamp = datetime.now() - timedelta(days=random.randint(0,30))
    timestamp = timestamp.replace(hour=random_hour, minute=random.randint(0,59))

    data.append({
        "order_id": f"ORD-{1000+i}",
        "pickup_district": start_dist,
        "pickup_lat": round(start_lat, 4),
        "pickup_lon": round(start_lon, 4),
        "dropoff_district": end_dist,
        "dropoff_lat": round(end_lat, 4),
        "dropoff_lon": round(end_lon, 4),
        "distance_km": round(distance_km, 2),
        "vehicle_type": vehicle,
        "weather": weather,
        "traffic_level": traffic,  
        "driver_experience": driver_exp, 
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "delivery_duration_mins": int(max(5, final_duration)) 
    })

df = pd.DataFrame(data)
df.to_csv("../data/berlin_delivery_data.csv", index=False)
print(f"Generated {NUM_ROWS} rows with GPS, Traffic, and Distance features.")
print(df.head())

import numpy as np
import pandas as pd
import joblib
import math
import warnings
import os
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances

# Suppress sklearn warnings
warnings.filterwarnings("ignore")

# Get the directory where this file is located
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def energy_to_magnitude(energy_joules):
    if energy_joules <= 0:
        raise ValueError("Energy must be positive")
    
    log_energy = math.log10(energy_joules)
    magnitude = (log_energy - 4.8) / 1.5
    return magnitude

def calculate_kinetic_energy(mass_kg, velocity_ms):
    return 0.5 * mass_kg * velocity_ms**2

def generate_seismic_features(magnitude, latitude, longitude):
    features = {}
    
    # Basic location and magnitude
    features['mag'] = magnitude
    features['latitude'] = latitude
    features['longitude'] = longitude
    
    # Time (current timestamp in milliseconds since epoch)
    features['time'] = int(datetime.now().timestamp() * 1000)
    
    # Depth estimation (typical asteroid impact depth is shallow)
    # Most impacts are surface or very shallow
    features['depth'] = np.random.uniform(0, 5)  # 0-5 km for impact
    
    # CDI (Community Determined Intensity) based on magnitude
    if magnitude >= 7.0:
        features['cdi'] = 6.0
    elif magnitude >= 6.0:
        features['cdi'] = 4.0
    elif magnitude >= 5.0:
        features['cdi'] = 3.0
    else:
        features['cdi'] = 2.0
    
    # MMI (Modified Mercalli Intensity) based on magnitude
    if magnitude >= 7.0:
        features['mmi'] = 7.0
    elif magnitude >= 6.0:
        features['mmi'] = 5.0
    elif magnitude >= 5.0:
        features['mmi'] = 4.0
    else:
        features['mmi'] = 3.0
    
    # Felt reports estimation (based on magnitude and populated areas)
    # Higher magnitude = more people feel it
    base_felt = max(0, int((magnitude - 3) * 1000))
    features['felt'] = base_felt
    
    # Significance calculation (empirical formula based on magnitude)
    features['sig'] = int(magnitude * 100 + 100)
    
    # Alert level based on advanced imputation rules from prepare_data
    if (magnitude >= 7.0 or features['sig'] >= 1000 or 
        features['mmi'] >= 8.0 or features['cdi'] >= 7.0):
        features['alert'] = 4  # Red alert
    elif (magnitude >= 6.5 or features['sig'] >= 700 or 
          features['mmi'] >= 6.5 or features['cdi'] >= 5.5 or
          (magnitude >= 6.0 and features['depth'] <= 10)):
        features['alert'] = 3  # Orange alert
    elif (magnitude >= 6.0 or features['sig'] >= 400 or 
          features['mmi'] >= 5.0 or features['cdi'] >= 4.0 or
          (magnitude >= 5.5 and features['depth'] <= 20)):
        features['alert'] = 2  # Yellow alert
    elif (magnitude >= 5.5 or features['sig'] >= 200 or 
          features['mmi'] >= 4.0 or features['cdi'] >= 3.0 or
          (magnitude >= 5.0 and features['depth'] <= 30)):
        features['alert'] = 1  # Green alert
    else:
        features['alert'] = 0  # No alert
    
    # Tsunami (impacts can potentially generate tsunamis if in ocean)
    # For simplicity, assume no tsunami for terrestrial impacts
    features['tsunami'] = 0
    
    # MagType (most common type for asteroid impacts would be body wave)
    features['magType'] = 24  # Most common value from training data
    
    return features

def denormalize_features_with_scaler(normalized_values, feature_names, scaler):
    # Create array with normalized values
    normalized_array = np.array(normalized_values).reshape(1, -1)
    
    # Use the scaler to inverse transform
    denormalized_array = scaler.inverse_transform(normalized_array)[0]
    
    # Create dictionary with denormalized values
    denormalized = {}
    for i, feature in enumerate(feature_names):
        denormalized[feature] = denormalized_array[i]
    
    return denormalized

def find_closest_in_cluster(cluster_id, synthetic_features, clustered_data, original_data):
    """Find the closest real earthquake example in the predicted cluster"""
    cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
    
    if len(cluster_data) == 0:
        return None, None, None
    
    # Features to compare (normalized features used in training)
    comparison_features = ['mag', 'depth', 'latitude', 'longitude', 'sig', 'cdi', 'mmi']
    
    # Extract synthetic features for comparison
    synthetic_array = np.array([[synthetic_features[f] for f in comparison_features]])
    
    # Extract cluster data for comparison
    cluster_array = cluster_data[comparison_features].values
    
    # Calculate distances
    distances = euclidean_distances(synthetic_array, cluster_array)[0]
    
    # Find closest example
    closest_idx = np.argmin(distances)
    closest_example_normalized = cluster_data.iloc[closest_idx]
    
    # Get the corresponding original data (same index)
    closest_example_original = original_data.iloc[closest_example_normalized.name]
    
    return closest_example_normalized, closest_example_original, distances[closest_idx]

def predict_seismic_impact(mass_kg, velocity_ms, latitude, longitude):
    # Step 1: Calculate energy and magnitude
    energy = calculate_kinetic_energy(mass_kg, velocity_ms)
    magnitude = energy_to_magnitude(energy)
    
    # Step 2: Generate synthetic seismic features
    synthetic_features = generate_seismic_features(magnitude, latitude, longitude)
    
    # Step 3: Load trained models and data
    try:
        kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        model_info = joblib.load(os.path.join(MODEL_DIR, "model_info.pkl"))
        clustered_data = pd.read_csv(os.path.join(MODEL_DIR, "clustered_earthquakes.csv"))
        original_data = pd.read_csv(os.path.join(MODEL_DIR, "dataset.csv"))
    except Exception as e:
        print(f"Error loading models: {e}")
        return None
    
    # Step 4: Prepare ALL features for prediction
    all_features = ['mag', 'time', 'felt', 'cdi', 'mmi', 'alert', 'sig', 'tsunami', 'magType', 'longitude', 'latitude', 'depth']
    features_to_normalize = model_info["features_to normalize"]
    
    # Normalize only the specified features
    feature_values_to_normalize = [synthetic_features[f] for f in features_to_normalize]
    feature_array_to_normalize = np.array(feature_values_to_normalize).reshape(1, -1)
    normalized_subset = scaler.transform(feature_array_to_normalize)[0]
    
    # Create full feature array with normalized values in correct positions
    full_feature_array = []
    normalize_idx = 0
    
    for feature in all_features:
        if feature in features_to_normalize:
            full_feature_array.append(normalized_subset[normalize_idx])
            normalize_idx += 1
        else:
            full_feature_array.append(synthetic_features[feature])
    
    full_feature_array = np.array(full_feature_array).reshape(1, -1)
    
    # Step 5: Predict cluster
    predicted_cluster = kmeans.predict(full_feature_array)[0]
    
    # Step 6: Find closest real example in cluster
    closest_normalized, closest_original, distance = find_closest_in_cluster(
        predicted_cluster, 
        {f: normalized_subset[features_to_normalize.index(f)] for f in features_to_normalize},
        clustered_data,
        original_data
    )
    
    if closest_normalized is not None and closest_original is not None:
        # Create refined prediction using closest example
        refined_features = synthetic_features.copy()
        
        # Replace generated features with real example (except magnitude and position)
        keep_original = ['mag', 'latitude', 'longitude']
        
        for column in closest_original.index:
            if column not in keep_original:
                refined_features[column] = closest_original[column]
        
        return refined_features, energy, predicted_cluster
    
    else:
        return synthetic_features, energy, predicted_cluster

if __name__ == "__main__":
    # Test scenarios with different asteroid sizes and locations
    test_scenarios = [
        {
            "name": "SMALL ASTEROID - Urban Impact",
            "description": "20-meter diameter stony asteroid (Chelyabinsk-like event)",
            "diameter": 20,  # meters
            "density": 3000,  # kg/m³ (stony meteorite)
            "velocity": 18000,  # m/s
            "location": "Tokyo, Japan",
            "latitude": 35.6762,
            "longitude": 139.6503
        },
        {
            "name": "MEDIUM ASTEROID - Rural Impact", 
            "description": "50-meter diameter rocky asteroid (Tunguska-like event)",
            "diameter": 50,  # meters
            "density": 2500,  # kg/m³ (rocky asteroid)
            "velocity": 20000,  # m/s
            "location": "Rural Siberia, Russia",
            "latitude": 60.8867,
            "longitude": 101.8431
        },
        {
            "name": "LARGE ASTEROID - Coastal Impact",
            "description": "100-meter diameter iron-rich asteroid (potential tsunami)",
            "diameter": 100,  # meters
            "density": 7800,  # kg/m³ (iron meteorite)
            "velocity": 25000,  # m/s (high velocity)
            "location": "Pacific Ocean near California",
            "latitude": 34.0522,
            "longitude": -120.2437
        },
        {
            "name": "CATASTROPHIC ASTEROID - Continental Impact",
            "description": "200-meter diameter asteroid (regional devastation)",
            "diameter": 200,  # meters
            "density": 3500,  # kg/m³ (mixed composition)
            "velocity": 30000,  # m/s (very high velocity)
            "location": "Australian Outback",
            "latitude": -25.2744,
            "longitude": 133.7751
        }
    ]
    
    print("ASTEROID IMPACT SEISMIC PREDICTION RESULTS")
    print("=" * 50)
    
    for i, scenario in enumerate(test_scenarios, 1):
        # Calculate mass from diameter and density
        radius = scenario['diameter'] / 2
        volume = (4/3) * math.pi * (radius ** 3)
        mass = volume * scenario['density']
        
        result, energy, cluster = predict_seismic_impact(
            mass_kg=mass,
            velocity_ms=scenario['velocity'], 
            latitude=scenario['latitude'],
            longitude=scenario['longitude']
        )
        
        if result:
            alert_names = {0: "No Alert", 1: "Green", 2: "Yellow", 3: "Orange", 4: "Red"}
            
            print(f"\nSCENARIO {i}: {scenario['name']}")
            print(f"Location: {scenario['location']}")
            print(f"Asteroid: {scenario['diameter']}m diameter, {scenario['velocity']:,} m/s")
            print(f"Energy: {energy:.2e} J")
            print(f"Magnitude: {result['mag']:.2f}")
            print(f"Alert Level: {alert_names.get(int(result['alert']), 'Unknown')}")
            print(f"Intensity (MMI): {result['mmi']:.1f}")
            print(f"Significance: {result['sig']:.0f}")
            print(f"Cluster: {cluster}")
            if result['tsunami'] == 1:
                print("⚠️  TSUNAMI WARNING")
            print("-" * 30)
    
    print("\nAll predictions completed.")
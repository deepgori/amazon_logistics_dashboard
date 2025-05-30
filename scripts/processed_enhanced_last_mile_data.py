import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta

RAW_META_DATA_DIR = 'data/last_mile_raw/almrrc2021-data-training'
RAW_EVAL_DATA_DIR = 'data/last_mile_raw/almrrc2021-data-evaluation'
PROCESSED_META_DATA_DIR = 'data/processed_last_mile_meta'

def load_json_file(filepath):
    """Loads a JSON file and handles potential errors."""
    if not os.path.exists(filepath):
        return {} # Return empty dict if file not found
    try:
        with open(filepath, 'r') as f:
            content = json.load(f)
            if not content: # Check for genuinely empty JSONs
                return {} # Return empty dict if JSON is empty
            return content
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return {} # Return empty dict on decode error
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}: {e}")
        return {} # Return empty dict on other errors

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    This function remains, but its output will likely be 0.0 due to data limitations.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0 

    R = 6371 
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def process_single_route_data(route_id, route_details, package_details_for_route, actual_sequences_for_route):
    """
    Processes details for a single route_id from combined JSON data.
    This version correctly extracts city, origin, packages, calculates volume, and sums duration from 'stops'.
    Distance is still 0.0 due to data limitation.
    """
    # Initialize all values to None or 0.0 for safety
    city = 'Unknown'
    route_date = None
    station_code = None
    route_score = None
    origin_lat, origin_lon = None, None
    vehicle_capacity = None
    num_deliveries = 0
    total_calculated_volume_cm3 = 0.0
    actual_route_duration_hours = 0.0
    actual_route_distance_km = 0.0

    if route_details: # Ensure route_details is not empty
        route_date = route_details.get('date_YYYY_MM_DD') or route_details.get('date')
        city = route_details.get('city', 'Unknown') 
        station_code = route_details.get('station_code')
        route_score = route_details.get('route_score')
        vehicle_capacity = route_details.get('executor_capacity_cm3') or route_details.get('vehicleCapacity')

        if 'origin' in route_details and isinstance(route_details.get('origin'), dict):
            origin_lat = route_details['origin'].get('latitude')
            origin_lon = route_details['origin'].get('longitude')
        
        stops_data_from_route_json = route_details.get('stops', {}) 
        
        if stops_data_from_route_json and isinstance(stops_data_from_route_json, dict):
            total_travel_time_seconds = 0
            total_planned_service_time_seconds = 0
            
            for stop_id, stop_detail in stops_data_from_route_json.items():
                if isinstance(stop_detail, dict): # Ensure stop_detail is a dictionary
                    total_travel_time_seconds += stop_detail.get('travel_time_to_next_stop_in_seconds', 0)
                    total_planned_service_time_seconds += stop_detail.get('planned_service_time_seconds', 0)
            
            actual_route_duration_hours = (total_travel_time_seconds + total_planned_service_time_seconds) / 3600 
            if actual_route_duration_hours < 0: actual_route_duration_hours = 0.0 # Ensure non-negative

    if package_details_for_route: # Ensure package_details is not empty
        inner_packages_dict = package_details_for_route.get('AD', {}) 
        if not inner_packages_dict and 'packages' in package_details_for_route and isinstance(package_details_for_route['packages'], dict):
            inner_packages_dict = package_details_for_route['packages']
        
        packages_list_values = list(inner_packages_dict.values()) 
        num_deliveries = len(packages_list_values) 
        
        for pkg in packages_list_values:
            if 'dimensions' in pkg and isinstance(pkg.get('dimensions'), dict):
                depth = pkg['dimensions'].get('depth_cm', 0)
                height = pkg['dimensions'].get('height_cm', 0)
                width = pkg['dimensions'].get('width_cm', 0)
                total_calculated_volume_cm3 += (depth * height * width)
    
    actual_route_distance_km = 0.0    

    return {
        'route_id': route_id,
        'city': city,
        'route_date': route_date,
        'station_code': station_code,
        'route_score': route_score,
        'origin_latitude': origin_lat,
        'origin_longitude': origin_lon,
        'vehicle_capacity_cm3': vehicle_capacity,
        'num_deliveries': num_deliveries, 
        'total_calculated_volume_cm3': total_calculated_volume_cm3, 
        'actual_route_duration_hours': actual_route_duration_hours,
        'actual_route_distance_km': actual_route_distance_km        # Remains 0.0
    }

# --- Main Processing Logic ---
if __name__ == "__main__":
    print("Starting processing of available ALMRRC meta-data JSONs (final attempt to maximize output)...")
    
    os.makedirs(PROCESSED_META_DATA_DIR, exist_ok=True)
    all_processed_routes = []

    # --- Load all relevant JSONs from model_build_inputs ---
    print(f"\n--- Loading from {RAW_META_DATA_DIR}/model_build_inputs/ ---")
    route_data_build = load_json_file(os.path.join(RAW_META_DATA_DIR, 'model_build_inputs', 'route_data.json'))
    package_data_build = load_json_file(os.path.join(RAW_META_DATA_DIR, 'model_build_inputs', 'package_data.json'))
    actual_sequences_build = load_json_file(os.path.join(RAW_META_DATA_DIR, 'model_build_inputs', 'actual_sequences.json'))
    print(f"DEBUG: route_data_build has {len(route_data_build)} entries.")
    print(f"DEBUG: package_data_build has {len(package_data_build)} entries.")
    print(f"DEBUG: actual_sequences_build has {len(actual_sequences_build)} entries.")


    print(f"\n--- Loading from {RAW_META_DATA_DIR}/model_apply_inputs/ ---")
    route_data_apply = load_json_file(os.path.join(RAW_META_DATA_DIR, 'model_apply_inputs', 'new_route_data.json'))
    package_data_apply = load_json_file(os.path.join(RAW_META_DATA_DIR, 'model_apply_inputs', 'new_package_data.json'))
    actual_sequences_apply = load_json_file(os.path.join(RAW_META_DATA_DIR, 'model_score_inputs', 'new_actual_sequences.json')) # Corrected path
    print(f"DEBUG: route_data_apply entries: {len(route_data_apply)}.")
    print(f"DEBUG: package_data_apply entries: {len(package_data_apply)}.")
    print(f"DEBUG: actual_sequences_apply entries: {len(actual_sequences_apply)}.")
    
    print(f"\n--- Loading from {RAW_EVAL_DATA_DIR}/model_apply_inputs/ and other eval folders ---")
    route_data_eval = load_json_file(os.path.join(RAW_EVAL_DATA_DIR, 'model_apply_inputs', 'eval_route_data.json'))
    package_data_eval = load_json_file(os.path.join(RAW_EVAL_DATA_DIR, 'model_apply_inputs', 'eval_package_data.json'))
    actual_sequences_eval = load_json_file(os.path.join(RAW_EVAL_DATA_DIR, 'model_score_inputs', 'eval_actual_sequences.json')) # Assuming eval_actual_sequences.json is here
    print(f"DEBUG: route_data_eval entries: {len(route_data_eval)}.")
    print(f"DEBUG: package_data_eval entries: {len(package_data_eval)}.")
    print(f"DEBUG: actual_sequences_eval entries: {len(actual_sequences_eval)}.")
    
    all_route_ids_to_process = set()
    if route_data_build: all_route_ids_to_process.update(route_data_build.keys())
    if route_data_apply: all_route_ids_to_process.update(route_data_apply.keys())
    if route_data_eval: all_route_ids_to_process.update(route_data_eval.keys())

    if not all_route_ids_to_process:
        print("\nNo route IDs found in any of the specified meta-data JSONs. Ensure files are correctly downloaded and not empty.")
        exit()

    processed_count = 0
    print(f"\n--- Starting processing of {len(all_route_ids_to_process)} unique route IDs ---")
    processed_first_few_debug = 0 
    for i, route_id in enumerate(all_route_ids_to_process):
        route_details = route_data_build.get(route_id) or route_data_apply.get(route_id) or route_data_eval.get(route_id)
        package_details = package_data_build.get(route_id) or package_data_apply.get(route_id) or package_data_eval.get(route_id)
        actual_sequences = actual_sequences_build.get(route_id) or actual_sequences_apply.get(route_id) or actual_sequences_eval.get(route_id)
        
        if processed_first_few_debug < 5:
            print(f"\nDEBUG Route ID: {route_id}")
            print(f"  route_details found: {bool(route_details)} (len: {len(route_details) if route_details else 0})")
            print(f"  package_details found: {bool(package_details)} (len: {len(package_details) if package_details else 0})")
            print(f"  actual_sequences found: {bool(actual_sequences)} (len: {len(actual_sequences) if actual_sequences else 0})")
            if not route_details: print(f"  --> Missing route_details for {route_id}")
            if not package_details: print(f"  --> Missing package_details for {route_id}")
            if not actual_sequences: print(f"  --> Missing actual_sequences for {route_id}")
            
        if not all([route_details, package_details]): # Removed actual_sequences from this check
            # print(f"DEBUG: Skipping route {route_id}: Missing core data from meta files (route_details or package_details).")
            continue 

        processed_metrics = process_single_route_data(route_id, route_details, package_details, actual_sequences)
        
        if processed_metrics:
            all_processed_routes.append(processed_metrics)
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} routes...")
            if processed_first_few_debug < 5: # Use separate counter for these prints
                pass


    if not all_processed_routes:
        print("\nNo valid routes processed after filtering for complete data. Ensure JSON files contain matching RouteIDs and relevant data.")
        exit()

    df_enhanced_meta_routes = pd.DataFrame(all_processed_routes)
    
    df_enhanced_meta_routes['route_date'] = df_enhanced_meta_routes['route_date'].astype(str).str.strip()
    df_enhanced_meta_routes['route_date'] = pd.to_datetime(df_enhanced_meta_routes['route_date'], errors='coerce')

    output_csv_path = os.path.join(PROCESSED_META_DATA_DIR, 'processed_enhanced_meta_routes.csv') 
    os.makedirs(PROCESSED_META_DATA_DIR, exist_ok=True)
    df_enhanced_meta_routes.to_csv(output_csv_path, index=False)
    
    print(f"\nEnhanced meta-data processing complete. Processed {len(df_enhanced_meta_routes)} routes.")
    print(f"Data saved to '{output_csv_path}'")
    print("\nFirst 5 rows of processed data:")
    print(df_enhanced_meta_routes.head())
    print("\nDescriptive statistics for processed data:")
    print(df_enhanced_meta_routes.describe())
    
    print("\nRoute counts by city (from enhanced meta-data):")
    print(df_enhanced_meta_routes['city'].value_counts())

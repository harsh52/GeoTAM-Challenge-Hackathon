import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
from rtree import index
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import concurrent.futures



def parallel(business_data_gdf_part, file_number):
    # Step 2: Ensure CRS Consistency
    print("Reprojecting GeoDataFrame to EPSG:27700...")
    business_data_gdf = business_data_gdf_part.to_crs(epsg=27700)

    # Step 3: Replace OSM Data with `.gpkg` Data
    business_locations = business_data_gdf.copy()
    print(f"Using {len(business_locations)} business locations from GeoPackage.")

    # Step 4: Download Public Transit Stations from OpenStreetMap
    place_name = "Manchester, UK"
    print(f"Downloading public transit stations for {place_name}...")
    transit_stations = ox.geometries_from_place(place_name, tags={'public_transport': True})
    transit_stations = transit_stations.to_crs(epsg=27700)
    print(f"Downloaded {len(transit_stations)} transit stations.")

    # Step 5: Calculate Distance to Nearest Transit Station
    print("Calculating distance to the nearest transit station...")


    def calculate_nearest_station_distance(geometry, stations):
        return stations.distance(geometry).min()


    business_locations['nearest_station_distance'] = business_locations['geometry'].apply(
        calculate_nearest_station_distance, stations=transit_stations
    )
    print("Distance calculation completed.")

    # Step 6: Calculate POI Density as a Proxy for Foot Traffic
    print("Calculating POI density for each business location...")


    def calculate_poi_density_optimized(geometry, spatial_index, pois, radius):
        buffer = geometry.buffer(radius)
        possible_matches_index = list(spatial_index.intersection(buffer.bounds))
        possible_matches = pois.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(buffer)]
        return len(precise_matches)


    radius = 1000 #meters
    print("Building spatial index for POI density calculation...")
    spatial_index = index.Index()
    for idx, geom in enumerate(business_locations['geometry']):
        spatial_index.insert(idx, geom.bounds)

    business_locations['poi_density'] = business_locations['geometry'].apply(
        lambda geom: calculate_poi_density_optimized(geom, spatial_index, business_locations, radius)
    )
    print("POI density calculation completed.")

    # Step 7: Calculate Foot Traffic Index
    print("Calculating Foot Traffic Index...")
    max_density = business_locations['poi_density'].max()
    min_density = business_locations['poi_density'].min()
    business_locations['Foot_Traffic_Index'] = (
            (business_locations['poi_density'] - min_density) / (max_density - min_density)
    )
    print("Foot Traffic Index calculation completed.")

    # Step 8: Adding Competitor Counts
    print("Calculating competitor counts...")
    spatial_index = index.Index()
    for idx, geom in enumerate(business_locations['geometry']):
        spatial_index.insert(idx, geom.bounds)


    def count_competitors(geometry, spatial_index, competitors, radius, category_col, business_category):
        buffer = geometry.buffer(radius)
        possible_matches_index = list(spatial_index.intersection(buffer.bounds))
        possible_matches = competitors.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(buffer)]
        precise_matches = precise_matches[precise_matches[category_col] == business_category]
        return len(precise_matches)


    category_col = 'voasubcategory'
    business_locations['competitor_count'] = business_locations.apply(
        lambda row: count_competitors(
            row.geometry, spatial_index, business_locations, radius,
            category_col=category_col, business_category=row[category_col]
        ), axis=1
    )
    print("Competitor count calculation completed.")

    # Step 9: Add Parking Availability
    print("Checking parking availability...")
    parking_data = ox.geometries_from_place(place_name, tags={'amenity': 'parking'})
    parking_data = parking_data.to_crs(epsg=27700)
    business_locations['parking_available'] = business_locations['geometry'].apply(
        lambda geom: any(parking_data.intersects(geom.buffer(5)))
    )
    print("Parking availability check completed.")

    # Rate Relief Adjustments
    business_locations['rateable_value'] = pd.to_numeric(business_locations['voarateablevalue'], errors='coerce').fillna(0)

    relief_factor = 0.8  # Businesses receiving reliefs get reduced turnover proxy

    business_locations['adjusted_rateable_value'] = business_locations['rateable_value'] * (
            business_locations['laratesreliefsamount'].isnull().astype(int) + (1 - relief_factor)
    )

    # Step 10: Advanced Clustering (DBSCAN and GMM)
    print("Applying advanced clustering methods...")

    # Data cleaning
    for col in ['poi_density', 'adjusted_rateable_value', 'competitor_count',
                'Foot_Traffic_Index', 'voafloorarea', 'laratespaid',
                'laratesreliefsamount']:
        business_locations[col] = pd.to_numeric(business_locations[col], errors='coerce')


    # Prepare features for clustering
    features_for_clustering = business_locations[[
        'poi_density', 'adjusted_rateable_value', 'competitor_count', 'Foot_Traffic_Index',
        'voafloorarea', 'laratespaid', 'laratesreliefsamount'
    ]].fillna(0)

    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features_for_clustering)

    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=10)  # Adjust parameters
    business_locations['dbscan_cluster'] = dbscan.fit_predict(normalized_features)

    # Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=3, random_state=42)  # Adjust components
    business_locations['gmm_cluster'] = gmm.fit_predict(normalized_features)
    print("Clustering completed.")

    # Step 11: Feature Importance-Based Weight Optimization
    print("Optimizing feature weights using Random Forest...")
    X = features_for_clustering
    y = business_locations['adjusted_rateable_value']  # Synthetic proxy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    feature_importances = rf.feature_importances_

    category_multipliers = {
        'RETAIL': 6.0,
        'LEISURE': 5.0,
        'OFFICE': 3.0,
        'INDUSTRIAL': 4.0,
        'OTHER': 4.0
    }

    subcategory_multipliers = {
        'RETAIL_HIGH_STREET': 6.5,
        'RETAIL_RESTAURANTS_AND_CAFES': 6.2,
        'LEISURE_LUXURY_HOTELS': 5.8,
        'OFFICE_GENERAL': 3.5,
        'INDUSTRIAL_LIGHT': 4.2,
    }
    business_locations['category_multiplier'] = business_locations['voacategory'].map(category_multipliers).fillna(4.0)

    business_locations['subcategory_multiplier'] = business_locations['voasubcategory'].map(
        subcategory_multipliers).fillna(4.0)
    business_locations['final_multiplier'] = (
            0.7 * business_locations['category_multiplier'] +
            0.3 * business_locations['subcategory_multiplier']
    )

    # Map multipliers to the business category
    business_locations['category_multiplier'] = business_locations['voacategory'].map(category_multipliers).fillna(4.0)

    business_locations['normalized_rateable_value'] = business_locations['adjusted_rateable_value'] / (
            business_locations['voafloorarea'] + 1
    )
    # Apply feature importances as weights
    business_locations['optimized_turnover_proxy'] = (
            1.5 * business_locations['adjusted_rateable_value'] * business_locations['final_multiplier'] +
            1.5 * np.log1p(business_locations['Foot_Traffic_Index'] + 1e-6) +
            1.2 * np.exp(-0.1 * business_locations['nearest_station_distance'].clip(upper=5000)) +
            1.5 * business_locations['parking_available'] +
            1.0 * (1 / (np.sqrt(business_locations['competitor_count']) + 1)) +
            0.5 * (business_locations['normalized_rateable_value'] / (business_locations['voafloorarea'] + 1)) +
            0.7 * business_locations['laratespaid']
    )

    # Step 12: Cluster-Specific Predictive Modeling
    print("Training predictive models for each cluster...")
    for cluster in business_locations['gmm_cluster'].unique():
        cluster_data = business_locations[business_locations['gmm_cluster'] == cluster]
        X_cluster = cluster_data[['poi_density', 'adjusted_rateable_value', 'competitor_count', 'Foot_Traffic_Index',
                                  'voafloorarea', 'laratespaid', 'laratesreliefsamount', 'category_multiplier']]
        y_cluster = cluster_data['optimized_turnover_proxy'].fillna(0)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_cluster, y_cluster)
        cluster_data['predicted_turnover'] = model.predict(X_cluster)
        business_locations.loc[cluster_data.index, 'predicted_turnover'] = cluster_data['predicted_turnover']

    # Step 13: Save Results
    output_gpkg_path = f"business_data_advanced_ml{file_number}.shp"
    business_locations.to_file(output_gpkg_path)
    print(f"Results saved to {output_gpkg_path}.")


if __name__ == "__main__":
    # Step 1: Load the GeoPackage File
    print("Loading GeoPackage data...")
    business_data_gdf = gpd.read_file("D_D/GeoTAM_Hackathon_OpenLocal.gpkg")  # Replace with the actual file path
    print(f"Loaded {len(business_data_gdf)} business locations.")

    # Step 2: Define fixed chunk size (20,000 rows per chunk)
    chunk_size = 20000
    num_chunks = (len(business_data_gdf) + chunk_size - 1) // chunk_size  # Calculate total chunks
    chunks = [business_data_gdf.iloc[i:i + chunk_size] for i in range(0, len(business_data_gdf), chunk_size)]

    print(f"Divided data into {len(chunks)} chunks with up to {chunk_size} rows each.")

    # Step 3: Use ProcessPoolExecutor for parallel processing
    num_threads = min(len(chunks), 6)  # Use up to 4 threads or less, depending on the number of chunks
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        print(f"Starting parallel processing with {num_threads} threads...")
        futures = {executor.submit(parallel, chunk.copy(), idx): idx for idx, chunk in enumerate(chunks)}

        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Partition {idx + 1}/{len(chunks)} processed successfully.")
            except Exception as e:
                print(f"[ERROR] Partition {idx + 1} failed with error: {e}")

    # Combine results into a single GeoDataFrame
    print("Combining processed chunks...")
    combined_data = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))
    print(f"Processing completed. Total records processed: {len(combined_data)}")
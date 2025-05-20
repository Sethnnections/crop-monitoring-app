from datetime import datetime, date, timedelta
from flask import Flask, render_template, jsonify, request
from config import Config
import json
import requests
import logging
import numpy as np
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    Geometry,
    BBox,
    CRS,
    bbox_to_dimensions,
)
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform
from functools import partial
import pyproj

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

API_BASE_URL = 'http://localhost:3000/api/farmers'

def get_farm_geojson(farmer_id):
    """Fetch farm data from API and convert to GeoJSON format"""
    try:
        url = f"{API_BASE_URL}/{farmer_id}/farms"
        logger.debug(f"Fetching farm data from: {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        logger.debug(f"API response: {data}")
        
        if not data.get('farms') or len(data['farms']) == 0:
            raise ValueError("No farms found for this farmer")
        
        # Convert the first farm's boundary to GeoJSON FeatureCollection format
        farm = data['farms'][0]
        boundary = farm.get('boundary')
        
        if not boundary:
            raise ValueError("Farm boundary data not found")
        
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": farm.get('name'),
                        "size": farm.get('size'),
                        "location": farm.get('location')
                    },
                    "geometry": boundary
                }
            ]
        }
        
        logger.debug(f"Converted GeoJSON: {geojson}")
        return geojson
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing farm data: {e}")
        raise

def validate_geojson(geojson_data):
    """Validate and normalize GeoJSON structure"""
    if not isinstance(geojson_data, dict):
        raise ValueError("GeoJSON must be a dictionary")
    
    if geojson_data.get('type') != 'FeatureCollection':
        raise ValueError("Only FeatureCollection type is supported")
    
    if not isinstance(geojson_data.get('features'), list) or len(geojson_data['features']) == 0:
        raise ValueError("FeatureCollection must contain at least one feature")
    
    for feature in geojson_data['features']:
        if feature.get('type') != 'Feature':
            raise ValueError("All elements in features must be of type Feature")
        
        geometry = feature.get('geometry')
        if not geometry:
            raise ValueError("Feature must have a geometry")
            
        if geometry.get('type') not in ['Polygon', 'MultiPolygon']:
            raise ValueError("Only Polygon and MultiPolygon geometries are supported")
            
        if not isinstance(geometry.get('coordinates'), list):
            raise ValueError("Geometry coordinates must be a list")
    
    return geojson_data

def get_geojson_centroid(geojson_data):
    """Get centroid of GeoJSON geometry, handling both Polygon and MultiPolygon"""
    try:
        geojson_data = validate_geojson(geojson_data)
        geom = shape(geojson_data['features'][0]['geometry'])

        if geom.geom_type == 'MultiPolygon':
            geom = geom.convex_hull

        project = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:4326'),
            pyproj.Proj(init='epsg:3857')
        )
        projected_geom = transform(project, geom)
        centroid = projected_geom.centroid

        project_back = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:3857'),
            pyproj.Proj(init='epsg:4326')
        )
        centroid_wgs84 = transform(project_back, centroid)

        logger.debug(f"Calculated centroid: {centroid_wgs84.y}, {centroid_wgs84.x}")
        return [centroid_wgs84.y, centroid_wgs84.x]
    except Exception as e:
        logger.error(f"Error calculating centroid: {e}")
        raise

def get_geojson_bbox(geojson_data):
    """Get bounding box of GeoJSON geometry"""
    try:
        geojson_data = validate_geojson(geojson_data)
        geom = shape(geojson_data['features'][0]['geometry'])
        minx, miny, maxx, maxy = geom.bounds
        logger.debug(f"GeoJSON BBox: {(minx, miny, maxx, maxy)}")
        return BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
    except Exception as e:
        logger.error(f"Error calculating bounding box: {e}")
        raise

def get_geometry_from_geojson(geojson_data):
    """Create SentinelHub Geometry from GeoJSON"""
    try:
        geojson_data = validate_geojson(geojson_data)
        return Geometry(geojson_data['features'][0]['geometry'], crs=CRS.WGS84)
    except Exception as e:
        logger.error(f"Error creating geometry: {e}")
        raise

@app.route('/monitoring/<int:farmer_id>')
def crop_monitoring(farmer_id):
    try:
        url = f"{API_BASE_URL}/{farmer_id}/farms"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('farms') or len(data['farms']) == 0:
            raise ValueError("No farms found for this farmer")
        
        first_farm = data['farms'][0]
        if not first_farm.get('boundary'):
            raise ValueError("Farm boundary data not found")
        
        initial_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": first_farm.get('name'),
                        "size": first_farm.get('size'),
                        "location": first_farm.get('location')
                    },
                    "geometry": first_farm['boundary']
                }
            ]
        }
        
        centroid = get_geojson_centroid(initial_geojson)
        
        return render_template(
            'crop_monitoring.html',
            center=centroid,
            geojson=json.dumps(initial_geojson),
            farmer=data.get('farmer', {}),
            farms=data['farms'],
            selected_farm=first_farm
        )
    except Exception as e:
        logger.error(f"Failed to render crop monitoring: {e}")
        return "Error loading crop monitoring data", 500

@app.route('/get-crop-index', methods=['POST'])
def get_crop_index():
    try:
        geojson_data = request.json
        logger.info("Received crop index request")
        logger.debug(f"Request data: {geojson_data}")

        # Extract parameters from request
        start_date = geojson_data.get('start_date')
        end_date = geojson_data.get('end_date')
        index_type = geojson_data.get('index_type', 'ndvi').lower()

        # Validate dates
        if not start_date or not end_date:
            today = datetime.utcnow()
            thirty_days_ago = today - timedelta(days=30)
            start_date = thirty_days_ago.strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')

        time_interval = (start_date, end_date)
        logger.info(f"Using time interval: {time_interval}")
        logger.info(f"Using index type: {index_type}")

        # Validate GeoJSON
        validate_geojson(geojson_data)
        geometry = get_geometry_from_geojson(geojson_data)
        bbox = get_geojson_bbox(geojson_data)

        # Set resolution (10m for Sentinel-2)
        resolution = 10
        size = bbox_to_dimensions(bbox, resolution=resolution)

        # Initialize SentinelHub configuration
        config = SHConfig()
        config.sh_client_id = app.config['SH_CLIENT_ID']
        config.sh_client_secret = app.config['SH_CLIENT_SECRET']
        config.instance_id = app.config['SH_INSTANCE_ID']

        # Generate evalscript based on index type
        evalscript = get_evalscript_for_index(index_type)

        # Create SentinelHub request
        request_sh = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order='leastCC'
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.PNG),
                SentinelHubRequest.output_response('index_values', MimeType.TIFF)
            ],
            geometry=geometry,
            size=size,
            config=config
        )

        # Get data from SentinelHub
        data = request_sh.get_data()
        if data:
            logger.info(f"{index_type.upper()} data retrieved successfully")
            
            # Extract image and index values
            if isinstance(data, list) and len(data) >= 2:
                image_data = data[0]
                index_values = data[1]
            elif isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                image_data = data[0].get('default.png')
                index_values = data[0].get('index_values.tif')
            else:
                logger.error(f"Unexpected data structure: {data}")
                return jsonify({'status': 'error', 'message': 'Invalid data structure from SentinelHub'})
                
            # Calculate statistics
            index_values = np.array(index_values)
            valid_values = index_values[index_values != 0]  # Filter out zeros (no data)
            
            if valid_values.size > 0:
                average_value = float(np.mean(valid_values))
                min_value = float(np.min(valid_values))
                max_value = float(np.max(valid_values))
            else:
                average_value = 0
                min_value = 0
                max_value = 0

            # Return the results
            return jsonify({
                'status': 'success',
                'image': image_data.tolist() if isinstance(image_data, np.ndarray) else image_data,
                'average_value': average_value,
                'min_value': min_value,
                'max_value': max_value,
                'index_type': index_type
            })
        
        else:
            logger.warning("No data available for the selected time range")
            return jsonify({'status': 'error', 'message': 'No data available for the selected time range'})

    except Exception as e:
        logger.exception(f"Error processing crop index request: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

def get_evalscript_for_index(index_type):
    """Generate evalscript based on index type"""
    # Base evalscript template
    evalscript_template = """
//VERSION=3
function setup() {
    return {
        input: [%INPUT_BANDS%],
        output: [
            { id: "default", bands: 4 },
            { id: "index_values", bands: 1, sampleType: "FLOAT32" }
        ]
    };
}

function evaluatePixel(samples) {
    // Calculate index
    %INDEX_CALCULATION%
    
    // Generate color based on index value
    let color;
    let output_value;
    if (%VALID_DATA_CHECK%) {
        // Color scale from red (low) to green (high)
        if (index_value < -0.2) {
            color = [0.7, 0, 0, 1]; // Deep red
        } else if (index_value < 0) {
            color = [0.9, 0.3, 0.2, 1]; // Red-orange
        } else if (index_value < 0.2) {
            color = [0.95, 0.6, 0.2, 1]; // Orange
        } else if (index_value < 0.4) {
            color = [0.95, 0.9, 0.3, 1]; // Yellow
        } else if (index_value < 0.6) {
            color = [0.8, 0.95, 0.3, 1]; // Yellow-green
        } else if (index_value < 0.8) {
            color = [0.5, 0.8, 0.2, 1]; // Light green
        } else {
            color = [0.1, 0.6, 0.1, 1]; // Deep green
        }
        output_value = index_value;
    } else {
        color = [0, 0, 0, 0]; // Transparent for no data
        output_value = 0;
    }
    
    return {
        default: color,
        index_values: [output_value]
    };
}
"""

    # Customize evalscript based on index type
    if index_type == 'ndvi':
        # NDVI: (NIR - Red) / (NIR + Red)
        input_bands = '"B04", "B08", "dataMask"'
        index_calculation = 'const index_value = (samples.B08 - samples.B04) / (samples.B08 + samples.B04 + 0.0001);'
        valid_data_check = 'samples.dataMask === 1'
        
    elif index_type == 'ndmi':
        # NDMI: (NIR - SWIR) / (NIR + SWIR)
        input_bands = '"B08", "B11", "dataMask"'
        index_calculation = 'const index_value = (samples.B08 - samples.B11) / (samples.B08 + samples.B11 + 0.0001);'
        valid_data_check = 'samples.dataMask === 1'
        
    elif index_type == 'msavi':
        # MSAVI: (2*NIR + 1 - sqrt((2*NIR + 1)² - 8*(NIR - Red))) / 2
        input_bands = '"B04", "B08", "dataMask"'
        index_calculation = '''
        const NIR = samples.B08;
        const RED = samples.B04;
        const index_value = (2 * NIR + 1 - Math.sqrt((2 * NIR + 1) * (2 * NIR + 1) - 8 * (NIR - RED))) / 2;
        '''
        valid_data_check = 'samples.dataMask === 1'
        
    elif index_type == 'reci':
        # RECI: (NIR / Red Edge) - 1
        input_bands = '"B05", "B08", "dataMask"'
        index_calculation = 'const index_value = (samples.B08 / samples.B05) - 1;'
        valid_data_check = 'samples.dataMask === 1 && samples.B05 > 0'
        
    elif index_type == 'ndre':
        # NDRE: (NIR - Red Edge) / (NIR + Red Edge)
        input_bands = '"B05", "B08", "dataMask"'
        index_calculation = 'const index_value = (samples.B08 - samples.B05) / (samples.B08 + samples.B05 + 0.0001);'
        valid_data_check = 'samples.dataMask === 1'
        
    else:
        # Default to NDVI
        logger.warning(f"Unknown index type: {index_type}. Using NDVI instead.")
        input_bands = '"B04", "B08", "dataMask"'
        index_calculation = 'const index_value = (samples.B08 - samples.B04) / (samples.B08 + samples.B04 + 0.0001);'
        valid_data_check = 'samples.dataMask === 1'
    
    # Replace placeholders in template
    evalscript = evalscript_template
    evalscript = evalscript.replace('%INPUT_BANDS%', input_bands)
    evalscript = evalscript.replace('%INDEX_CALCULATION%', index_calculation)
    evalscript = evalscript.replace('%VALID_DATA_CHECK%', valid_data_check)
    
    return evalscript

@app.route('/flood-monitoring/<int:farmer_id>')
def flood_monitoring(farmer_id):
    try:
        url = f"{API_BASE_URL}/{farmer_id}/farms"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('farms') or len(data['farms']) == 0:
            raise ValueError("No farms found for this farmer")
        
        first_farm = data['farms'][0]
        if not first_farm.get('boundary'):
            raise ValueError("Farm boundary data not found")
        
        initial_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": first_farm.get('name'),
                        "size": first_farm.get('size'),
                        "location": first_farm.get('location')
                    },
                    "geometry": first_farm['boundary']
                }
            ]
        }
        
        centroid = get_geojson_centroid(initial_geojson)
        
        print("Flood Monitoring Center:", centroid)
        print("Initial GeoJSON:", json.dumps(initial_geojson))
        print("Farmer Name:", data.get('farmer', {}).get('name'))
        return render_template(
            'flood_monitoring.html',
            center=centroid,
            geojson=json.dumps(initial_geojson),
            farmer=data.get('farmer', {}),
            farms=data['farms'],
            selected_farm=first_farm
        )
    except Exception as e:
        logger.error(f"Failed to render flood monitoring: {e}")
        return "Error loading flood monitoring data", 500
    

@app.route('/get-water-detection', methods=['POST'])
def get_water_detection():
    try:
        geojson_data = request.json
        logger.info("Received water detection request")

        start_date = request.args.get('start_date') or geojson_data.get('start_date')
        end_date = request.args.get('end_date') or geojson_data.get('end_date')

        if not start_date or not end_date:
            today = datetime.utcnow()
            start_date = date(today.year, today.month, 1).isoformat()
            end_date = today.isoformat()

        time_interval = (start_date, end_date)
        logger.info(f"Using time interval: {time_interval}")

        validate_geojson(geojson_data)
        geometry = get_geometry_from_geojson(geojson_data)
        bbox = get_geojson_bbox(geojson_data)

        resolution = 10
        size = bbox_to_dimensions(bbox, resolution=resolution)

        config = SHConfig()
        config.sh_client_id = app.config['SH_CLIENT_ID']
        config.sh_client_secret = app.config['SH_CLIENT_SECRET']
        config.instance_id = app.config['SH_INSTANCE_ID']

        # Enhanced water detection using both NDWI and MNDWI
        evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B03", "B08", "B11", "dataMask"],
        output: [
            { id: "default", bands: 4 },
            { id: "ndwi", bands: 1, sampleType: "FLOAT32" },
            { id: "mndwi", bands: 1, sampleType: "FLOAT32" },
            { id: "water_mask", bands: 1, sampleType: "UINT8" },
            { id: "dataMask", bands: 1 }
        ]
    };
}

function evaluatePixel(samples) {
    // Calculate water indices
    const ndwi = (samples.B03 - samples.B08) / (samples.B03 + samples.B08 + 0.0001);
    const mndwi = (samples.B03 - samples.B11) / (samples.B03 + samples.B11 + 0.0001);
    
    // Combined water detection (NDWI > 0.2 OR MNDWI > 0)
    const isWater = (ndwi > 0.2 || mndwi > 0) && samples.dataMask === 1;
    
    // Visual representation
    let color;
    if (!samples.dataMask) {
        color = [0, 0, 0, 0]; // No data (transparent)
    } else if (isWater) {
        // Blue gradient based on confidence (darker blue = more confident)
        const confidence = Math.max(ndwi, mndwi);
        color = [0, 0, 0.5 + confidence * 0.5, 1];
    } else {
        // Non-water areas in light gray
        color = [0.9, 0.9, 0.9, 1];
    }
    
    return {
        default: color,
        ndwi: [ndwi],
        mndwi: [mndwi],
        water_mask: [isWater ? 1 : 0],
        dataMask: [samples.dataMask]
    };
}
"""

        request_sh = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order='leastCC'
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.PNG),
                SentinelHubRequest.output_response('water_mask', MimeType.TIFF)
            ],
            geometry=geometry,
            size=size,
            config=config
        )

        data = request_sh.get_data()
        if data:
            logger.info("Water detection data retrieved successfully")
            logger.debug(f"Data structure: {type(data)}")
            
            # Fix: Correctly access the returned data structure
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                # Extract image and water mask from the dictionary
                image_data = data[0].get('default.png')
                water_mask = data[0].get('water_mask.tif')
                
                if image_data is None or water_mask is None:
                    logger.error(f"Missing expected data keys. Available keys: {data[0].keys()}")
                    return jsonify({'status': 'error', 'message': 'Invalid data structure from SentinelHub'})
            else:
                # If structure is different, try to adapt
                try:
                    # Try original approach in case API returns different format
                    image_data = data[0]
                    water_mask = data[1]
                except (IndexError, TypeError):
                    logger.error(f"Unexpected data structure: {data}")
                    return jsonify({'status': 'error', 'message': 'Unexpected data structure from SentinelHub'})
            
            # Ensure data is numpy arrays
            image_data = np.array(image_data)
            water_mask = np.array(water_mask)
            
            # Calculate water coverage percentage
            total_pixels = np.sum(water_mask >= 0)  # All valid pixels
            water_pixels = np.sum(water_mask == 1)
            water_coverage = (water_pixels / total_pixels * 100) if total_pixels > 0 else 0

            return jsonify({
                'status': 'success',
                'image': image_data.tolist(),
                'water_coverage': round(water_coverage, 2),
                'risk_level': 12
            })
        
        else:
            logger.warning("No data available for the selected time range")
            return jsonify({'status': 'error', 'message': 'No data available'})

    except Exception as e:
        logger.exception("Error processing water detection request")
        return jsonify({'status': 'error', 'message': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
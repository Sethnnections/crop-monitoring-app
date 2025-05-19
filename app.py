import logging
from datetime import datetime, date
from flask import Flask, render_template, jsonify, request
from config import Config
import json
import requests  # Added for API requests
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform
from functools import partial
import pyproj
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
        response.raise_for_status()  # Raises exception for 4XX/5XX status codes
        
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

def calculate_ndvi(red, nir):
    red = np.asarray(red, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi.tolist()


def calculate_ndwi(green, nir):
    """Calculate Normalized Difference Water Index"""
    green = np.asarray(green, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)
    ndwi = (green - nir) / (green + nir + 1e-8)
    return ndwi.tolist()

def calculate_mndwi(green, swir):
    """Calculate Modified Normalized Difference Water Index"""
    green = np.asarray(green, dtype=np.float32)
    swir = np.asarray(swir, dtype=np.float32)
    mndwi = (green - swir) / (green + swir + 1e-8)
    return mndwi.tolist()

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

        # Handle MultiPolygon by converting to single Polygon (using convex hull)
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
    """Get bounding box of GeoJSON geometry, handling both Polygon and MultiPolygon"""
    try:
        geojson_data = validate_geojson(geojson_data)
        geom = shape(geojson_data['features'][0]['geometry'])
        
        # For MultiPolygon, get the bounds of the entire geometry
        minx, miny, maxx, maxy = geom.bounds
        logger.debug(f"GeoJSON BBox: {(minx, miny, maxx, maxy)}")
        return BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
    except Exception as e:
        logger.error(f"Error calculating bounding box: {e}")
        raise

def get_geometry_from_geojson(geojson_data):
    """Create SentinelHub Geometry from GeoJSON, handling both Polygon and MultiPolygon"""
    try:
        geojson_data = validate_geojson(geojson_data)
        return Geometry(geojson_data['features'][0]['geometry'], crs=CRS.WGS84)
    except Exception as e:
        logger.error(f"Error creating geometry: {e}")
        raise

@app.route('/monitoring/<int:farmer_id>')
def monitoring(farmer_id):
    try:
        url = f"{API_BASE_URL}/{farmer_id}/farms"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('farms') or len(data['farms']) == 0:
            raise ValueError("No farms found for this farmer")
        
        # Get the first farm's boundary for initial map display
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
        
        return render_template('index.html', 
                            center=centroid,
                            geojson=json.dumps(initial_geojson),
                            farmer=data['farms'][0]['farmer'],  # Pass farmer info
                            farms=data['farms'],                # Pass all farms
                            selected_farm=first_farm            # Pass initial farm
                            )
    except Exception as e:
        logger.error(f"Failed to render index: {e}")
        return "Error loading map data", 500


@app.route('/get-ndvi', methods=['POST'])
def get_ndvi():
    try:
        geojson_data = request.json
        logger.info("Received NDVI request")

        # Extract optional date parameters from the query string or JSON
        start_date = request.args.get('start_date') or geojson_data.get('start_date')
        end_date = request.args.get('end_date') or geojson_data.get('end_date')

        # Fallback to current month's first day and now
        if not start_date or not end_date:
            today = datetime.utcnow()
            start_date = date(today.year, today.month, 1).isoformat()
            end_date = today.isoformat()

        time_interval = (start_date, end_date)
        logger.info(f"Using time interval: {time_interval}")

        # Validate and process the GeoJSON
        validate_geojson(geojson_data)
        geometry = get_geometry_from_geojson(geojson_data)
        bbox = get_geojson_bbox(geojson_data)

        resolution = 10
        size = bbox_to_dimensions(bbox, resolution=resolution)

        config = SHConfig()
        config.sh_client_id = app.config['SH_CLIENT_ID']
        config.sh_client_secret = app.config['SH_CLIENT_SECRET']
        config.instance_id = app.config['SH_INSTANCE_ID']

        evalscript = """//VERSION=3
function setup() {
    return {
        input: ["B03","B04", "B08", "dataMask"],
        output: [
            { id: "default", bands: 4 },
            { id: "index", bands: 1, sampleType: "FLOAT32" },
            { id: "eobrowserStats", bands: 2, sampleType: 'FLOAT32' },
            { id: "dataMask", bands: 1 }
        ]
      };
}

function evaluatePixel(samples) {
    let val = index(samples.B08, samples.B04);
    let imgVals = null;
    const indexVal = samples.dataMask === 1 ? val : NaN;
  
    if (val<-0.5) imgVals = [0.05,0.05,0.05,samples.dataMask];  // Very dark (background)
    else if (val<-0.2) imgVals = [0.75,0.75,0.75,samples.dataMask];  // Light gray
    else if (val<-0.1) imgVals = [0.86,0.86,0.86,samples.dataMask];  // Lighter gray
    else if (val<0) imgVals = [0.92,0.92,0.92,samples.dataMask];  // Very light gray
    else if (val<0.05) imgVals = [1,0.2,0.2,samples.dataMask];  // Red
    else if (val<0.1) imgVals = [1,0.6,0.1,samples.dataMask];  // Orange
    else if (val<0.15) imgVals = [1,0.9,0,samples.dataMask];  // Golden yellow
    else if (val<0.2) imgVals = [1,1,0.2,samples.dataMask];  // Bright yellow
    else if (val<0.3) imgVals = [0.8,1,0.2,samples.dataMask];  // Yellow-green
    else if (val<0.4) imgVals = [0.4,0.9,0.1,samples.dataMask];  // Light green
    else if (val<0.5) imgVals = [0.2,0.7,0.2,samples.dataMask];  // Medium green
    else if (val<0.6) imgVals = [0.1,0.5,0.1,samples.dataMask];  // Dark green
    else imgVals = [0,0.3,0,samples.dataMask];  // Deep dark green
    
    return {
      default: imgVals,
      index: [indexVal],
      eobrowserStats:[val,isCloud(samples)?1:0],
      dataMask: [samples.dataMask]
    };
}

function isCloud(samples){
    const NGDR = index(samples.B03, samples.B04);
    const bRatio = (samples.B03 - 0.175) / (0.39 - 0.175);
    return bRatio > 1 || (bRatio > 0 && NGDR > 0);
}"""


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
                SentinelHubRequest.output_response('default', MimeType.PNG)
            ],
            geometry=geometry,
            size=size,
            config=config
        )

        data = request_sh.get_data()
        if data:
            logger.info("NDVI data retrieved successfully.")
            return jsonify({'status': 'success', 'image': data[0].tolist()})
        else:
            logger.warning("No data available for the selected time range.")
            return jsonify({'status': 'error', 'message': 'No data available for the selected time range'})

    except Exception as e:
        logger.exception("Error processing NDVI request")
        return jsonify({'status': 'error', 'message': str(e)})
    
  
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
        
        return render_template(
            'flood_monitoring.html',
            center=centroid,
            geojson=json.dumps(initial_geojson),
            farmer=data['farms'][0]['farmer'],
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
            print({
                'status': 'success',
                'image': image_data.tolist(),
                'water_coverage': round(water_coverage, 2),
                'risk_level': 12
            })

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
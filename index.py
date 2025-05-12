import logging
from datetime import datetime, date, timedelta
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

def handle_date_params(request_obj, geojson_data, default_days=30):
    """Helper function to extract and validate date parameters"""
    start_date = request_obj.args.get('start_date') or geojson_data.get('start_date')
    end_date = request_obj.args.get('end_date') or geojson_data.get('end_date')

    # Fallback to default if not specified
    if not end_date:
        end_date = datetime.utcnow().isoformat()
    if not start_date:
        start_date = (datetime.utcnow() - timedelta(days=default_days)).isoformat()

    time_interval = (start_date, end_date)
    logger.info(f"Using time interval: {time_interval}")
    return time_interval

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

        # Extract date parameters
        time_interval = handle_date_params(request, geojson_data, default_days=30)

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
                SentinelHubRequest.output_response('default', MimeType.PNG),
                SentinelHubRequest.output_response('index', MimeType.JSON)
            ],
            geometry=geometry,
            size=size,
            config=config
        )

        data = request_sh.get_data()
        if data:
            logger.info("NDVI data retrieved successfully.")
            return jsonify({
                'status': 'success', 
                'image': data[0].tolist(),
                'index_data': data[1]['index'] if len(data) > 1 else None
            })
        else:
            logger.warning("No data available for the selected time range.")
            return jsonify({'status': 'error', 'message': 'No data available for the selected time range'})

    except Exception as e:
        logger.exception("Error processing NDVI request")
        return jsonify({'status': 'error', 'message': str(e)})
    
@app.route('/get-fire-detection', methods=['POST'])
def get_fire_detection():
    try:
        geojson_data = request.json
        logger.info("Received fire detection request")

        # Extract date parameters
        time_interval = handle_date_params(request, geojson_data, default_days=7)

        # Validate and process the GeoJSON
        validate_geojson(geojson_data)
        geometry = get_geometry_from_geojson(geojson_data)
        bbox = get_geojson_bbox(geojson_data)

        resolution = 20  # Lower resolution for fire detection
        size = bbox_to_dimensions(bbox, resolution=resolution)

        config = SHConfig()
        config.sh_client_id = app.config['SH_CLIENT_ID']
        config.sh_client_secret = app.config['SH_CLIENT_SECRET']
        config.instance_id = app.config['SH_INSTANCE_ID']

        evalscript = """//VERSION=3
function setup() {
    return {
        input: ["B04", "B08", "B11", "B12", "dataMask"],
        output: [
            { id: "default", bands: 4 },
            { id: "fireMask", bands: 1, sampleType: "FLOAT32" },
            { id: "dataMask", bands: 1 }
        ]
    };
}

function evaluatePixel(samples) {
    // Normalized Burn Ratio (NBR) for fire detection
    const nbr = (samples.B08 - samples.B11) / (samples.B08 + samples.B11 + 1e-8);
    
    // SWIR/NIR ratio for active fire detection
    const swirnir = samples.B11 / samples.B08;
    
    // Thermal anomaly detection
    const thermalAnomaly = samples.B12 > 0.3 ? 1 : 0;
    
    // Fire probability (combining multiple indices)
    const fireProb = (swirnir > 1.5 ? 0.6 : 0) + 
                    (nbr < 0.1 ? 0.3 : 0) + 
                    (thermalAnomaly * 0.1);
    
    // Apply data mask
    const fireValue = samples.dataMask === 1 ? fireProb : 0;
    
    // Visualization
    let imgVals;
    if (fireProb > 0.7) {
        imgVals = [1, 0, 0, samples.dataMask];  // Red for high probability
    } else if (fireProb > 0.4) {
        imgVals = [1, 0.65, 0, samples.dataMask];  // Orange for medium probability
    } else if (fireProb > 0.1) {
        imgVals = [1, 1, 0, samples.dataMask];  // Yellow for low probability
    } else {
        imgVals = [0.2, 0.2, 0.2, samples.dataMask];  // Dark for no fire
    }
    
    return {
        default: imgVals,
        fireMask: [fireValue],
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
                SentinelHubRequest.output_response('fireMask', MimeType.JSON)
            ],
            geometry=geometry,
            size=size,
            config=config
        )

        data = request_sh.get_data()
        if data:
            logger.info("Fire detection data retrieved successfully.")
            return jsonify({
                'status': 'success',
                'image': data[0].tolist(),
                'fire_mask': data[1]['fireMask'] if len(data) > 1 else None
            })
        else:
            logger.warning("No data available for fire detection.")
            return jsonify({'status': 'error', 'message': 'No data available for fire detection'})

    except Exception as e:
        logger.exception("Error processing fire detection request")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get-flood-detection', methods=['POST'])
def get_flood_detection():
    try:
        geojson_data = request.json
        logger.info("Received flood detection request")

        # Extract date parameters
        time_interval = handle_date_params(request, geojson_data, default_days=30)

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
        input: ["B03", "B08", "B11", "B12", "dataMask"],
        output: [
            { id: "default", bands: 4 },
            { id: "waterMask", bands: 1, sampleType: "UINT8" },  // Changed from FLOAT32 to UINT8
            { id: "dataMask", bands: 1 }
        ]
    };
}

function evaluatePixel(samples) {
    // Modified Normalized Difference Water Index (MNDWI)
    const mndwi = (samples.B03 - samples.B11) / (samples.B03 + samples.B11 + 1e-8);
    
    // Water probability (scaled to 0-255 for UINT8)
    const waterProb = (mndwi > 0.1 ? 0.7 : 0) + 
                     (samples.B03 < 0.2 ? 0.3 : 0);
    const waterValue = Math.min(255, Math.max(0, waterProb * 255));
    
    // Apply data mask
    const maskedValue = samples.dataMask === 1 ? waterValue : 0;
    
    // Visualization
    let imgVals;
    if (waterProb > 0.7) {
        imgVals = [0, 0, 255, samples.dataMask * 255];  // Blue for high probability
    } else if (waterProb > 0.4) {
        imgVals = [0, 128, 255, samples.dataMask * 255];  // Light blue for medium probability
    } else if (waterProb > 0.1) {
        imgVals = [204, 229, 255, samples.dataMask * 255];  // Very light blue for low probability
    } else {
        imgVals = [50, 50, 50, samples.dataMask * 255];  // Dark for no water
    }
    
    return {
        default: imgVals,
        waterMask: [maskedValue],
        dataMask: [samples.dataMask * 255]
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
                SentinelHubRequest.output_response('waterMask', MimeType.JSON)
            ],
            geometry=geometry,
            size=size,
            config=config
        )

        data = request_sh.get_data()
        if data:
            logger.info("Flood detection data retrieved successfully.")
            return jsonify({
                'status': 'success',
                'image': data[0].tolist(),
                'water_mask': data[1]['waterMask'] if len(data) > 1 else None
            })
        else:
            logger.warning("No data available for flood detection.")
            return jsonify({'status': 'error', 'message': 'No data available for flood detection'})

    except Exception as e:
        logger.exception("Error processing flood detection request")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/fire-monitoring/<int:farmer_id>')
def fire_monitoring(farmer_id):
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
        
        return render_template('fire.html', 
                            center=centroid,
                            geojson=json.dumps(initial_geojson),
                            farmer=data['farms'][0]['farmer'],
                            farms=data['farms'],
                            selected_farm=first_farm
                            )
    except Exception as e:
        logger.error(f"Failed to render fire monitoring page: {e}")
        return "Error loading fire monitoring data", 500

@app.route('/flood-monitoring/<int:farmer_id>')
def flood_monitoring(farmer_id):
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
        
        return render_template('flood.html', 
                            center=centroid,
                            geojson=json.dumps(initial_geojson),
                            farmer=data['farms'][0]['farmer'],
                            farms=data['farms'],
                            selected_farm=first_farm
                            )
    except Exception as e:
        logger.error(f"Failed to render flood monitoring page: {e}")
        return "Error loading flood monitoring data", 500

if __name__ == '__main__':
    app.run(debug=True)
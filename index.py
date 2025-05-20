import logging
from datetime import datetime, date
from flask import Flask, render_template, jsonify, request
from config import Config
import json
import requests
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

app = Flask(__name__)
app.config.from_object(Config)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

API_BASE_URL = 'http://localhost:3000/api/farmers'

INDICES = {
    'NDVI': {
        'name': 'Normalized Difference Vegetation Index',
        'formula': '(NIR - Red) / (NIR + Red)',
        'description': 'Measures vegetation health and density',
        'range': [-1, 1],
        'healthy_range': [0.2, 0.8]
    },
    'NDMI': {
        'name': 'Normalized Difference Moisture Index',
        'formula': '(NIR - SWIR) / (NIR + SWIR)',
        'description': 'Measures vegetation water content',
        'range': [-1, 1],
        'healthy_range': [0.1, 0.6]
    },
    'MSAVI': {
        'name': 'Modified Soil Adjusted Vegetation Index',
        'formula': '(2 * NIR + 1 - sqrt((2 * NIR + 1)² - 8 * (NIR - Red))) / 2',
        'description': 'Soil-adjusted vegetation index for areas with bare soil',
        'range': [-1, 1],
        'healthy_range': [0.2, 0.8]
    },
    'RECI': {
        'name': 'Red Edge Chlorophyll Index',
        'formula': '(NIR / Red Edge) - 1',
        'description': 'Estimates chlorophyll content in vegetation',
        'range': [0, 10],
        'healthy_range': [1, 5]
    },
    'NDRE': {
        'name': 'Normalized Difference Red Edge',
        'formula': '(NIR - Red Edge) / (NIR + Red Edge)',
        'description': 'Measures chlorophyll content in crops',
        'range': [-1, 1],
        'healthy_range': [0.2, 0.7]
    }
}

def get_farm_geojson(farmer_id):
    """Fetch farm data from API and convert to GeoJSON format"""
    try:
        response = requests.get(f"{API_BASE_URL}/{farmer_id}/farms")
        response.raise_for_status()
        data = response.json()
        
        if not data.get('farms'):
            raise ValueError("No farms found for this farmer")
        
        farm = data['farms'][0]
        boundary = farm.get('boundary')
        
        if not boundary:
            raise ValueError("Farm boundary data not found")
        
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {
                    "name": farm.get('name'),
                    "size": farm.get('size'),
                    "location": farm.get('location')
                },
                "geometry": boundary
            }]
        }
    except Exception as e:
        logger.error(f"Error getting farm geojson: {e}")
        raise

def get_geojson_bbox(geojson_data):
    """Get bounding box of GeoJSON geometry"""
    try:
        geom = shape(geojson_data['features'][0]['geometry'])
        minx, miny, maxx, maxy = geom.bounds
        return BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
    except Exception as e:
        logger.error(f"Error calculating bounding box: {e}")
        raise

def get_geometry_from_geojson(geojson_data):
    """Create SentinelHub Geometry from GeoJSON"""
    try:
        return Geometry(geojson_data['features'][0]['geometry'], crs=CRS.WGS84)
    except Exception as e:
        logger.error(f"Error creating geometry: {e}")
        raise

def get_index_evalscript(index_name):
    """Generate evalscript for the selected vegetation index"""
    base_script = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04", "B05", "B08", "B11", "B12", "dataMask"],
        output: [
            { id: "default", bands: 4 },
            { id: "index", bands: 1, sampleType: "FLOAT32" },
            { id: "dataMask", bands: 1 }
        ]
    };
}

function evaluatePixel(samples) {
    // Band definitions
    const BLUE = samples.B02;
    const GREEN = samples.B03;
    const RED = samples.B04;
    const RED_EDGE1 = samples.B05;
    const NIR = samples.B08;
    const SWIR1 = samples.B11;
    const SWIR2 = samples.B12;
    
    // Calculate selected index
    let indexValue;
    """
    
    index_calculations = {
        'NDVI': '(NIR - RED) / (NIR + RED + 0.0001)',
        'NDMI': '(NIR - SWIR1) / (NIR + SWIR1 + 0.0001)',
        'MSAVI': '(2 * NIR + 1 - Math.sqrt(Math.pow((2 * NIR + 1), 2) - 8 * (NIR - RED))) / 2',
        'RECI': '(NIR / RED_EDGE1) - 1',
        'NDRE': '(NIR - RED_EDGE1) / (NIR + RED_EDGE1 + 0.0001)'
    }
    
    color_mapping = """
    // Color mapping based on index value
    let color;
    if (!samples.dataMask) {
        color = [0, 0, 0, 0]; // No data
    } else {
        // Normalize index value for coloring
        const normalized = (indexValue - {min}) / ({max} - {min});
        
        // Healthy vegetation gradient (green to red)
        if (normalized < 0.3) {
            // Brown to yellow (low vegetation)
            const intensity = normalized / 0.3;
            color = [0.5 + intensity * 0.5, 0.3 + intensity * 0.7, 0, 1];
        } else if (normalized < 0.7) {
            // Yellow to green (healthy vegetation)
            const intensity = (normalized - 0.3) / 0.4;
            color = [1 - intensity * 0.8, 0.8, 0, 1];
        } else {
            // Green to dark green (very healthy vegetation)
            const intensity = (normalized - 0.7) / 0.3;
            color = [0.2, 0.8 - intensity * 0.3, 0, 1];
        }
    }
    
    return {
        default: color,
        index: [indexValue],
        dataMask: [samples.dataMask]
    };
}
"""
    
    index_info = INDICES.get(index_name, INDICES['NDVI'])
    script = base_script + f"indexValue = {index_calculations[index_name]};" + color_mapping
    script = script.replace("{min}", str(index_info['range'][0]))
    script = script.replace("{max}", str(index_info['range'][1]))
    
    return script

@app.route('/farm-monitoring/<int:farmer_id>')
def farm_monitoring(farmer_id):
    try:
        geojson = get_farm_geojson(farmer_id)
        response = requests.get(f"{API_BASE_URL}/{farmer_id}/farms")
        farmer_data = response.json()
        
        return render_template(
            'farm_monitoring.html',
            geojson=json.dumps(geojson),
            farmer=farmer_data,
            indices=INDICES,
            default_index='NDVI'
        )
    except Exception as e:
        logger.error(f"Failed to render farm monitoring: {e}")
        return "Error loading farm monitoring data", 500

@app.route('/get-index-data', methods=['POST'])
def get_index_data():
    try:
        data = request.json
        geojson = data.get('geojson')
        index_name = data.get('index', 'NDVI')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        if not start_date or not end_date:
            today = datetime.utcnow()
            start_date = date(today.year, today.month, 1).isoformat()
            end_date = today.isoformat()

        time_interval = (start_date, end_date)
        geometry = get_geometry_from_geojson(geojson)
        bbox = get_geojson_bbox(geojson)
        resolution = 10
        size = bbox_to_dimensions(bbox, resolution=resolution)

        config = SHConfig()
        config.sh_client_id = app.config['SH_CLIENT_ID']
        config.sh_client_secret = app.config['SH_CLIENT_SECRET']
        config.instance_id = app.config['SH_INSTANCE_ID']

        request_sh = SentinelHubRequest(
            evalscript=get_index_evalscript(index_name),
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order='leastCC'
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.PNG),
                SentinelHubRequest.output_response('index', MimeType.TIFF)
            ],
            geometry=geometry,
            size=size,
            config=config
        )

        data = request_sh.get_data()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data available'})

        image_data = np.array(data[0])
        index_data = np.array(data[1])
        
        # Calculate statistics
        valid_pixels = index_data[index_data != 0]
        if valid_pixels.size > 0:
            stats = {
                'min': float(np.nanmin(valid_pixels)),
                'max': float(np.nanmax(valid_pixels)),
                'mean': float(np.nanmean(valid_pixels)),
                'median': float(np.nanmedian(valid_pixels))
            }
        else:
            stats = {'min': 0, 'max': 0, 'mean': 0, 'median': 0}

        return jsonify({
            'status': 'success',
            'image': image_data.tolist(),
            'index_data': index_data.tolist(),
            'stats': stats,
            'index_info': INDICES.get(index_name)
        })

    except Exception as e:
        logger.exception("Error processing index request")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
import logging
from flask import Flask, render_template, jsonify, request
from config import Config
import json
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

GEOJSON_PATH = 'assets/irregularshape.geojson'

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

@app.route('/')
def index():
    try:
        with open(GEOJSON_PATH) as f:
            geojson_data = json.load(f)
        centroid = get_geojson_centroid(geojson_data)
        logger.info("Rendering index page with GeoJSON and centroid.")
        return render_template('index.html', center=centroid, geojson=json.dumps(geojson_data))
    except Exception as e:
        logger.error(f"Failed to render index: {e}")
        return "Error loading map data", 500

@app.route('/get-ndvi', methods=['POST'])
def get_ndvi():
    try:
        geojson_data = request.json
        logger.info("Received NDVI request")
        
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

        time_interval = ('2023-01-01', '2023-01-30')

        # Your existing evalscript here...
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

if __name__ == '__main__':
    app.run(debug=True)
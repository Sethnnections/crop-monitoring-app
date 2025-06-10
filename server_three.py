from datetime import datetime, date, timedelta, timezone
from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from config import Config
import json
import requests
import logging
import numpy as np
import base64
import io
from PIL import Image
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    Geometry,
    BBox,
    CRS,
    bbox_to_dimensions,
    MosaickingOrder,
)
from shapely.geometry import shape
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

# Add database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crop_monitoring.db'  # Change to your preferred DB
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

API_BASE_URL = 'http://localhost:3000/api/farmers'

# Database Models
class CropMonitoringRecord(db.Model):
    """Store crop monitoring data for each farmer's farm"""
    __tablename__ = 'crop_monitoring_records'
    
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, nullable=False, index=True)
    farm_id = db.Column(db.String(100), nullable=False, index=True)  # Using string for flexibility
    farm_name = db.Column(db.String(200))
    farm_area = db.Column(db.Float)  # in hectares
    crop_name = db.Column(db.String(100))
    sowing_date = db.Column(db.Date)
    growth_stage = db.Column(db.String(200))
    
    # Image capture details
    image_date = db.Column(db.Date, nullable=False, index=True)
    cloud_coverage = db.Column(db.Float, default=0.0)  # percentage
    
    # Vegetation indices
    ndvi_value = db.Column(db.Float)
    reci_value = db.Column(db.Float)
    ndmi_value = db.Column(db.Float)
    ndre_value = db.Column(db.Float)
    msavi_value = db.Column(db.Float)
    
    # Weather data
    precipitation = db.Column(db.Float)  # mm
    min_temperature = db.Column(db.Float)  # Celsius
    max_temperature = db.Column(db.Float)  # Celsius
    
    # Image storage (base64 encoded)
    ndvi_image = db.Column(db.Text)
    reci_image = db.Column(db.Text)
    ndmi_image = db.Column(db.Text)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    # Composite index for efficient queries
    __table_args__ = (
        db.Index('idx_farmer_farm_date', 'farmer_id', 'farm_id', 'image_date'),
    )

class WeatherData(db.Model):
    """Store weather information for specific dates and locations"""
    __tablename__ = 'weather_data'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, index=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    precipitation = db.Column(db.Float)  # mm
    min_temperature = db.Column(db.Float)  # Celsius
    max_temperature = db.Column(db.Float)  # Celsius
    humidity = db.Column(db.Float)  # percentage
    wind_speed = db.Column(db.Float)  # km/h
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()

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

def numpy_to_base64_image(image_array):
    """Convert numpy array to base64 encoded PNG image"""
    try:
        # Ensure image is in the right format (0-255, uint8)
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Create PIL Image
        if len(image_array.shape) == 3:
            # RGB image
            image = Image.fromarray(image_array, 'RGB')
        else:
            # Grayscale image
            image = Image.fromarray(image_array, 'L')
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

def get_weather_data(centroid, date_str):
    """Mock weather data - replace with actual weather API"""
    # This is a placeholder - integrate with actual weather service
    return {
        'precipitation': 0.6,
        'min_temperature': 17,
        'max_temperature': 36,
        'humidity': 65,
        'wind_speed': 12
    }
def calculate_growth_stage(sowing_date, current_date):
    """Estimate growth stage based on days since sowing"""
    if not sowing_date:
        return "Unknown"
    
    days_since_sowing = (current_date - sowing_date).days
    
    if days_since_sowing < 7:
        return "BBCH 00-09 Germination"
    elif days_since_sowing < 21:
        return "BBCH 10-19 Leaf development"
    elif days_since_sowing < 35:
        return "BBCH 20-29 Tillering"
    elif days_since_sowing < 50:
        return "BBCH 30-39 Stem elongation"
    elif days_since_sowing < 70:
        return "BBCH 50-59 Inflorescence emergence"
    elif days_since_sowing < 90:
        return "BBCH 60-69 Flowering"
    elif days_since_sowing < 120:
        return "BBCH 70-79 Development of fruit"
    elif days_since_sowing < 150:
        return "BBCH 80-89 Ripening of fruit"
    else:
        return "BBCH 90-99 Senescence"

def generate_index_remarks(current_value, previous_value, index_type):
    """Generate remarks based on index changes"""
    if previous_value is None:
        return f"Initial {index_type.upper()} measurement recorded."
    
    change = current_value - previous_value
    
    if index_type.lower() == 'ndvi':
        if change < -0.05:
            return "Significant decrease in vegetation health detected. Field scouting recommended."
        elif change < -0.02:
            return "Moderate decrease in NDVI. Monitor closely for potential issues."
        elif change > 0.05:
            return "Excellent vegetation growth observed. Crop health is improving."
        elif change > 0.02:
            return "Positive vegetation development detected."
        else:
            return "NDVI values remain stable."
    
    elif index_type.lower() == 'reci':
        if change < -1.0:
            return "Significant decrease in chlorophyll levels detected."
        elif change < -0.5:
            return "Moderate decrease in chlorophyll content."
        elif change > 1.0:
            return "Excellent chlorophyll development observed."
        else:
            return "Chlorophyll levels remain stable."
    
    elif index_type.lower() == 'ndmi':
        if change < -0.05:
            return "Increased water stress detected. Consider irrigation."
        elif change < -0.02:
            return "Moderate water stress indication."
        elif change > 0.05:
            return "Good water content levels."
        else:
            return "Water stress levels remain stable."
    
    return f"{index_type.upper()} values recorded."

def calculate_fire_risk_level(fire_coverage):
    """Determine fire risk level based on fire coverage percentage"""
    if fire_coverage > 10:
        return 'Extreme'
    elif fire_coverage > 5:
        return 'High'
    elif fire_coverage > 2:
        return 'Medium'
    elif fire_coverage > 0.5:
        return 'Low'
    else:
        return 'Normal'

@app.route('/fire-monitoring/<int:farmer_id>')
def fire_monitoring(farmer_id):
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
            'fire_monitoring.html',
            center=centroid,
            geojson=json.dumps(initial_geojson),
            farmer=data.get('farmer', {}),
            farms=data['farms'],
            selected_farm=first_farm
        )
    except Exception as e:
        logger.error(f"Failed to render fire monitoring: {e}")
        return "Error loading fire monitoring data", 500

@app.route('/get-fire-detection', methods=['POST'])
def get_fire_detection():
    try:
        geojson_data = request.json
        logger.info("Received fire detection request")

        start_date = request.args.get('start_date') or geojson_data.get('start_date')
        end_date = request.args.get('end_date') or geojson_data.get('end_date')

        if not start_date or not end_date:
            today = datetime.utcnow()
            today = datetime.now(timezone.utc)
            start_date = (today - timedelta(days=7)).isoformat()
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

        # Fire detection using NBR (Normalized Burn Ratio) and thermal anomalies
        evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B8A", "B12", "dataMask"],
        output: [
            { id: "default", bands: 4 },
            { id: "fire_mask", bands: 1, sampleType: "UINT8" },
            { id: "nbr", bands: 1, sampleType: "FLOAT32" }
        ]
    };
}

function evaluatePixel(samples) {
    // Calculate NBR (Normalized Burn Ratio)
    const nbr = (samples.B8A - samples.B12) / (samples.B8A + samples.B12 + 0.0001);
    
    // Fire detection criteria (low NBR indicates burned areas)
    const isBurned = nbr < 0.1 && samples.dataMask === 1;
    
    // Thermal anomaly detection (using SWIR band)
    const isHotSpot = samples.B12 > 0.3 && samples.dataMask === 1;
    
    // Combined fire indicator
    const isFire = (isBurned || isHotSpot);
    
    // Visual representation
    let color;
    if (!samples.dataMask) {
        color = [0, 0, 0, 0]; // No data (transparent)
    } else if (isFire) {
        // Fire areas in red (intensity based on severity)
        const severity = isHotSpot ? 1.0 : 0.7;
        color = [severity, 0, 0, 1];
    } else {
        // Non-fire areas in natural color representation
        const vegetation = samples.B8A * 2.5;
        color = [0, vegetation, 0, 1];
    }
    
    return {
        default: color,
        fire_mask: [isFire ? 1 : 0],
        nbr: [nbr]
    };
}
"""

        request_sh = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order=MosaickingOrder.LEAST_CC
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.PNG),
                SentinelHubRequest.output_response('fire_mask', MimeType.TIFF)
            ],
            geometry=geometry,
            size=size,
            config=config
        )

        data = request_sh.get_data()
        if data:
            logger.info("Fire detection data retrieved successfully")
            
            # Extract image and fire mask
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                image_data = data[0].get('default.png')
                fire_mask = data[0].get('fire_mask.tif')
            else:
                try:
                    image_data = data[0]
                    fire_mask = data[1]
                except (IndexError, TypeError):
                    logger.error(f"Unexpected data structure: {data}")
                    return jsonify({'status': 'error', 'message': 'Unexpected data structure from SentinelHub'})
            
            # Calculate fire coverage percentage
            total_pixels = np.sum(fire_mask >= 0)  # All valid pixels
            fire_pixels = np.sum(fire_mask == 1)
            fire_coverage = (fire_pixels / total_pixels * 100) if total_pixels > 0 else 0

            return jsonify({
                'status': 'success',
                'image': image_data.tolist(),
                'fire_coverage': round(fire_coverage, 2),
                'risk_level': calculate_fire_risk_level(fire_coverage)
            })
        
        else:
            logger.warning("No data available for the selected time range")
            return jsonify({'status': 'error', 'message': 'No data available'})

    except Exception as e:
        logger.exception("Error processing fire detection request")
        return jsonify({'status': 'error', 'message': str(e)})



@app.route('/fetch-and-store-crop-data', methods=['POST'])
def fetch_and_store_crop_data():
    """Fetch satellite data and store it for specific farmer and farm"""
    try:
        data = request.json
        farmer_id = data.get('farmer_id')
        farm_id = data.get('farm_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not all([farmer_id, farm_id]):
            return jsonify({'status': 'error', 'message': 'farmer_id and farm_id are required'})
        
        # Get farm data from API
        geojson_data = get_farm_geojson(farmer_id)
        farm_info = geojson_data['features'][0]['properties']
        
        # Set default dates if not provided
        if not start_date or not end_date:
            end_date = datetime.utcnow().date()
            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=7)
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Get centroid for weather data
        centroid = get_geojson_centroid(geojson_data)
        
        # Prepare SentinelHub configuration
        geometry = get_geometry_from_geojson(geojson_data)
        bbox = get_geojson_bbox(geojson_data)
        resolution = 10
        size = bbox_to_dimensions(bbox, resolution=resolution)
        
        config = SHConfig()
        config.sh_client_id = app.config['SH_CLIENT_ID']
        config.sh_client_secret = app.config['SH_CLIENT_SECRET']
        config.instance_id = app.config['SH_INSTANCE_ID']
        
        time_interval = (start_date.isoformat(), end_date.isoformat())
        
        # Fetch different indices
        indices_to_fetch = ['ndvi', 'reci', 'ndmi']
        results = {}
        
        for index_type in indices_to_fetch:
            try:
                evalscript = get_evalscript_for_index(index_type)
                
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
                
                sh_data = request_sh.get_data()
                
                if sh_data:
                    if isinstance(sh_data, list) and len(sh_data) >= 2:
                        image_data = sh_data[0]
                        index_values = sh_data[1]
                    elif isinstance(sh_data, list) and len(sh_data) == 1 and isinstance(sh_data[0], dict):
                        image_data = sh_data[0].get('default.png')
                        index_values = sh_data[0].get('index_values.tif')
                    else:
                        continue
                    
                    # Calculate statistics
                    index_values = np.array(index_values)
                    valid_values = index_values[index_values != 0]
                    
                    if valid_values.size > 0:
                        average_value = float(np.mean(valid_values))
                        # Convert image to base64
                        image_base64 = numpy_to_base64_image(np.array(image_data))
                        
                        results[index_type] = {
                            'value': average_value,
                            'image': image_base64
                        }
                
            except Exception as e:
                logger.error(f"Error fetching {index_type}: {e}")
                continue
        
        # Get weather data
        weather = get_weather_data(centroid, end_date.isoformat())
        
        # Create or update monitoring record
        existing_record = CropMonitoringRecord.query.filter_by(
            farmer_id=farmer_id,
            farm_id=farm_id,
            image_date=end_date
        ).first()
        
        if existing_record:
            # Update existing record
            record = existing_record
        else:
            # Create new record  
            record = CropMonitoringRecord(
                farmer_id=farmer_id,
                farm_id=farm_id,
                image_date=end_date
            )
        
        # Update record with fetched data
        record.farm_name = farm_info.get('name')
        record.farm_area = farm_info.get('size')
        record.crop_name = data.get('crop_name', 'Corn (Maize)')
        
        if data.get('sowing_date'):
            record.sowing_date = datetime.strptime(data['sowing_date'], '%Y-%m-%d').date()
        
        record.growth_stage = calculate_growth_stage(record.sowing_date, end_date)
        
        # Store index values and images
        if 'ndvi' in results:
            record.ndvi_value = results['ndvi']['value']
            record.ndvi_image = results['ndvi']['image']
        
        if 'reci' in results:
            record.reci_value = results['reci']['value']
            record.reci_image = results['reci']['image']
        
        if 'ndmi' in results:
            record.ndmi_value = results['ndmi']['value']
            record.ndmi_image = results['ndmi']['image']
        
        # Store weather data
        record.precipitation = weather.get('precipitation')
        record.min_temperature = weather.get('min_temperature')
        record.max_temperature = weather.get('max_temperature')
        
        # Calculate cloud coverage (mock value - would come from actual data)
        record.cloud_coverage = 0.0
        
        db.session.add(record)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Crop monitoring data stored successfully',
            'record_id': record.id,
            'data_summary': {
                'ndvi': results.get('ndvi', {}).get('value'),
                'reci': results.get('reci', {}).get('value'),
                'ndmi': results.get('ndmi', {}).get('value'),
                'date': end_date.isoformat()
            }
        })
        
    except Exception as e:
        logger.exception("Error in fetch_and_store_crop_data")
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)})
        
@app.route('/report/<int:farmer_id>/<farm_id>', methods=['GET'])
def generate_crop_report(farmer_id, farm_id):
    """Generate comprehensive crop monitoring report"""
    try:
        # Get the two most recent records for comparison
        records = CropMonitoringRecord.query.filter_by(
            farmer_id=farmer_id,
            farm_id=farm_id
        ).order_by(desc(CropMonitoringRecord.image_date)).limit(2).all()
        
        if not records:
            return jsonify({
                'status': 'error',
                'message': 'No monitoring data found for this farm'
            })
        
        current_record = records[0]
        previous_record = records[1] if len(records) > 1 else None
        
        # Calculate changes
        changes = {}
        if previous_record:
            if current_record.ndvi_value and previous_record.ndvi_value:
                changes['ndvi'] = current_record.ndvi_value - previous_record.ndvi_value
            if current_record.reci_value and previous_record.reci_value:
                changes['reci'] = current_record.reci_value - previous_record.reci_value
            if current_record.ndmi_value and previous_record.ndmi_value:
                changes['ndmi'] = current_record.ndmi_value - previous_record.ndmi_value
        
        # Generate remarks
        remarks = {
            'ndvi': generate_index_remarks(
                current_record.ndvi_value,
                previous_record.ndvi_value if previous_record else None,
                'ndvi'
            ),
            'reci': generate_index_remarks(
                current_record.reci_value,
                previous_record.reci_value if previous_record else None,
                'reci'
            ),
            'ndmi': generate_index_remarks(
                current_record.ndmi_value,
                previous_record.ndmi_value if previous_record else None,
                'ndmi'
            )
        }
        
        # Build report structure similar to the uploaded sample
        report = {
            'status': 'success',
            'report_date': current_record.image_date.strftime('%d %b %Y'),
            'field_info': {
                'field_name': current_record.farm_name or f'Farm {farm_id}',
                'area': f"{current_record.farm_area} ha" if current_record.farm_area else "N/A",
                'crop_name': current_record.crop_name or 'Unknown',
                'sowing_date': current_record.sowing_date.strftime('%d %b %Y') if current_record.sowing_date else 'N/A',
                'growth_stage': current_record.growth_stage or 'Unknown'
            },
            'current_period': {
                'image_date': current_record.image_date.strftime('%d %b %Y'),
                'clouds': f"{current_record.cloud_coverage:.1f} ha / {current_record.cloud_coverage/current_record.farm_area*100:.2f}%" if current_record.farm_area else "0 ha / 0%",
                'ndvi': {
                    'value': round(current_record.ndvi_value, 2) if current_record.ndvi_value else 'N/A',
                    'change': round(changes.get('ndvi', 0), 2) if 'ndvi' in changes else 'N/A',
                    'remark': remarks['ndvi']
                },
                'reci': {
                    'value': round(current_record.reci_value, 2) if current_record.reci_value else 'N/A',
                    'change': round(changes.get('reci', 0), 2) if 'reci' in changes else 'N/A',
                    'remark': remarks['reci']
                },
                'ndmi': {
                    'value': round(current_record.ndmi_value, 2) if current_record.ndmi_value else 'N/A',
                    'change': round(changes.get('ndmi', 0), 2) if 'ndmi' in changes else 'N/A',
                    'remark': remarks['ndmi']
                }
            },
            'weather_info': {
                'weekly_precipitation': f"{current_record.precipitation} mm" if current_record.precipitation else 'N/A',
                'min_temperature': f"{current_record.min_temperature}°C" if current_record.min_temperature else 'N/A',
                'max_temperature': f"{current_record.max_temperature}°C" if current_record.max_temperature else 'N/A'
            }
        }
        
        # Add previous period data if available
        if previous_record:
            report['previous_period'] = {
                'image_date': previous_record.image_date.strftime('%d %b %Y'),
                'clouds': f"{previous_record.cloud_coverage:.1f} ha / {previous_record.cloud_coverage/previous_record.farm_area*100:.2f}%" if previous_record.farm_area else "0 ha / 0%",
                'ndvi': round(previous_record.ndvi_value, 2) if previous_record.ndvi_value else 'N/A',
                'reci': round(previous_record.reci_value, 2) if previous_record.reci_value else 'N/A',
                'ndmi': round(previous_record.ndmi_value, 2) if previous_record.ndmi_value else 'N/A'
            }
        
        # Add images if requested
        include_images = request.args.get('include_images', 'false').lower() == 'true'
        if include_images:
            report['images'] = {
                'ndvi': current_record.ndvi_image,
                'reci': current_record.reci_image,
                'ndmi': current_record.ndmi_image
            }
        
        return jsonify(report)
        
    except Exception as e:
        logger.exception("Error generating crop report")
        return jsonify({'status': 'error', 'message': str(e)})

# Keep existing routes
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
            selected_farm=first_farm,
            farmer_id=farmer_id,
            farm_id=first_farm.get('id')
        )
    except Exception as e:
        logger.error(f"Failed to render crop monitoring: {e}")
        return "Error loading crop monitoring data", 500

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
            today = datetime.now(timezone.utc)
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

def calculate_risk_level(water_coverage):
    """Determine flood risk level based on water coverage percentage"""
    if water_coverage > 30:
        return 'High'
    elif water_coverage > 15:
        return 'Medium'
    elif water_coverage > 5:
        return 'Low'
    else:
        return 'Normal'

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
            
            # Extract image and water mask
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                image_data = data[0].get('default.png')
                water_mask = data[0].get('water_mask.tif')
                
                if image_data is None or water_mask is None:
                    logger.error(f"Missing expected data keys. Available keys: {data[0].keys()}")
                    return jsonify({'status': 'error', 'message': 'Invalid data structure from SentinelHub'})
            else:
                try:
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
                'risk_level': calculate_risk_level(water_coverage)
            })
        
        else:
            logger.warning("No data available for the selected time range")
            return jsonify({'status': 'error', 'message': 'No data available'})

    except Exception as e:
        logger.exception("Error processing water detection request")
        return jsonify({'status': 'error', 'message': str(e)})

# New utility endpoints for database management
@app.route('/api/monitoring-records/<int:farmer_id>')
def get_monitoring_records(farmer_id):
    """Get all monitoring records for a farmer"""
    try:
        farm_id = request.args.get('farm_id')
        limit = int(request.args.get('limit', 10))
        
        query = CropMonitoringRecord.query.filter_by(farmer_id=farmer_id)
        
        if farm_id:
            query = query.filter_by(farm_id=farm_id)
        
        records = query.order_by(desc(CropMonitoringRecord.image_date)).limit(limit).all()
        
        result = []
        for record in records:
            result.append({
                'id': record.id,
                'farm_id': record.farm_id,
                'farm_name': record.farm_name,
                'image_date': record.image_date.isoformat(),
                'ndvi_value': record.ndvi_value,
                'reci_value': record.reci_value,
                'ndmi_value': record.ndmi_value,
                'growth_stage': record.growth_stage,
                'precipitation': record.precipitation,
                'min_temperature': record.min_temperature,
                'max_temperature': record.max_temperature
            })
        
        return jsonify({
            'status': 'success',
            'records': result,
            'total': len(result)
        })
        
    except Exception as e:
        logger.exception("Error retrieving monitoring records")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/text-report/<int:farmer_id>/<farm_id>', methods=['GET'])
def text_report(farmer_id, farm_id):
    """Render a text-based crop monitoring report as HTML"""
    try:
        # Get the two most recent records for comparison
        records = CropMonitoringRecord.query.filter_by(
            farmer_id=farmer_id,
            farm_id=farm_id
        ).order_by(desc(CropMonitoringRecord.image_date)).limit(2).all()
        
        if not records:
            return "No monitoring data found for this farm", 404
        
        current_record = records[0]
        previous_record = records[1] if len(records) > 1 else None
        
        changes = {}
        if previous_record:
            if current_record.ndvi_value and previous_record.ndvi_value:
                changes['ndvi'] = current_record.ndvi_value - previous_record.ndvi_value
            if current_record.reci_value and previous_record.reci_value:
                changes['reci'] = current_record.reci_value - previous_record.reci_value
            if current_record.ndmi_value and previous_record.ndmi_value:
                changes['ndmi'] = current_record.ndmi_value - previous_record.ndmi_value
        
        remarks = {
            'ndvi': generate_index_remarks(
                current_record.ndvi_value,
                previous_record.ndvi_value if previous_record else None,
                'ndvi'
            ),
            'reci': generate_index_remarks(
                current_record.reci_value,
                previous_record.reci_value if previous_record else None,
                'reci'
            ),
            'ndmi': generate_index_remarks(
                current_record.ndmi_value,
                previous_record.ndmi_value if previous_record else None,
                'ndmi'
            )
        }
        
        return render_template(
            'text_report.html',
            current=current_record,
            previous=previous_record,
            changes=changes,
            remarks=remarks
        )
    except Exception as e:
        logger.exception("Error rendering text report")
        return "Error generating report", 500
    
  
# Initialize Flask app and database
@app.route('/fire-monitoring/<int:farmer_id>')
def fire_monitoring(farmer_id):
    try:
        response = requests.get(f"{API_BASE_URL}/{farmer_id}/farms")
        response.raise_for_status()
        data = response.json()
        
        if not data.get('farms') or len(data['farms']) == 0:
            raise ValueError("No farms found for this farmer")
        
        first_farm = data['farms'][0]
        if not first_farm.get('boundary'):
            raise ValueError("Farm boundary data not found")
        
        initial_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {
                    "name": first_farm.get('name'),
                    "size": first_farm.get('size'),
                    "location": first_farm.get('location')
                },
                "geometry": first_farm['boundary']
            }]
        }
        
        return render_template(
            'fire_monitoring.html',
            center=get_geojson_centroid(initial_geojson),
            geojson=json.dumps(initial_geojson),
            farmer=data.get('farmer', {}),
            farms=data['farms'],
            selected_farm=first_farm,
            farmer_id=farmer_id,
            farm_id=first_farm.get('id')
        )
    except Exception as e:
        logger.error(f"Failed to render fire monitoring: {e}")
        return "Error loading fire monitoring data", 500

@app.route('/get-fire-detection', methods=['POST'])
def get_fire_detection():
    try:
        geojson_data = request.json
        start_date = request.args.get('start_date') or geojson_data.get('start_date')
        end_date = request.args.get('end_date') or geojson_data.get('end_date') or datetime.now(timezone.utc).date()
        start_date = start_date or (end_date - timedelta(days=7) if isinstance(end_date, date) else 
                                  datetime.strptime(end_date, '%Y-%m-%d').date() - timedelta(days=7))
        
        geometry = get_geometry_from_geojson(geojson_data)
        bbox = get_geojson_bbox(geojson_data)
        size = bbox_to_dimensions(bbox, resolution=10)
        
        config = SHConfig()
        config.sh_client_id = app.config['SH_CLIENT_ID']
        config.sh_client_secret = app.config['SH_CLIENT_SECRET']
        config.instance_id = app.config['SH_INSTANCE_ID']
        
        # Fire detection evalscript using thermal bands and spectral indices
        evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B04", "B08", "B11", "B12", "dataMask"],
        output: [
            { id: "default", bands: 4 },
            { id: "fire_mask", bands: 1, sampleType: "UINT8" },
            { id: "temperature", bands: 1, sampleType: "FLOAT32" }
        ]
    };
}

function evaluatePixel(samples) {
    // Calculate fire indices
    const nbr = (samples.B08 - samples.B12) / (samples.B08 + samples.B12 + 0.0001);
    const ndvi = (samples.B08 - samples.B04) / (samples.B08 + samples.B04 + 0.0001);
    
    // Brightness temperature approximation for B11 (SWIR1)
    const brightness_temp = (samples.B11 > 0.1) ? (samples.B11 * 300) : 0;
    
    // Fire detection logic combining multiple criteria
    const fire_condition1 = samples.B12 > 0.1 && samples.B11 > 0.05; // High SWIR values
    const fire_condition2 = nbr < -0.2; // Low NBR indicates burned areas
    const fire_condition3 = brightness_temp > 320; // High temperature threshold
    const fire_condition4 = samples.B12 > (samples.B11 * 1.5); // SWIR2 > SWIR1 ratio
    
    // Combine conditions for fire detection
    const is_fire = samples.dataMask === 1 && (
        (fire_condition1 && fire_condition3) ||
        (fire_condition1 && fire_condition2) ||
        (fire_condition3 && fire_condition4)
    );
    
    // Create risk levels based on temperature and spectral response
    let risk_level = 0;
    if (is_fire) {
        if (brightness_temp > 350 || samples.B12 > 0.3) risk_level = 3; // High risk
        else if (brightness_temp > 330 || samples.B12 > 0.2) risk_level = 2; // Medium risk
        else risk_level = 1; // Low risk
    }
    
    // Color coding for visualization
    let color;
    if (!samples.dataMask) {
        color = [0, 0, 0, 0];
    } else if (risk_level === 3) {
        color = [1, 0, 0, 1]; // Red for high risk
    } else if (risk_level === 2) {
        color = [1, 0.5, 0, 1]; // Orange for medium risk
    } else if (risk_level === 1) {
        color = [1, 1, 0, 1]; // Yellow for low risk
    } else {
        // Normal vegetation color
        if (ndvi > 0.3) color = [0, 0.7, 0, 0.7]; // Green
        else color = [0.6, 0.4, 0.2, 0.7]; // Brown
    }
    
    return {
        default: color,
        fire_mask: [risk_level],
        temperature: [brightness_temp]
    };
}"""

        request_sh = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date.isoformat(), end_date.isoformat()),
                mosaicking_order='leastCC'
            )],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.PNG),
                SentinelHubRequest.output_response('fire_mask', MimeType.TIFF),
                SentinelHubRequest.output_response('temperature', MimeType.TIFF)
            ],
            geometry=geometry,
            size=size,
            config=config
        )

        data = request_sh.get_data()
        if not data:
            return jsonify({'status': 'error', 'message': 'No satellite data available for the selected time range'})
        
        # Process the returned data
        if isinstance(data[0], dict):
            image_data = data[0].get('default.png')
            fire_mask = data[0].get('fire_mask.tif')
            temperature_data = data[0].get('temperature.tif')
        else:
            image_data, fire_mask, temperature_data = data[0], data[1], data[2] if len(data) > 2 else None
        
        # Analyze fire mask for statistics
        fire_mask = np.array(fire_mask)
        temperature_data = np.array(temperature_data) if temperature_data is not None else np.zeros_like(fire_mask)
        
        # Calculate fire statistics
        total_pixels = np.sum(fire_mask >= 0)
        fire_pixels = np.sum(fire_mask > 0)
        high_risk_pixels = np.sum(fire_mask == 3)
        medium_risk_pixels = np.sum(fire_mask == 2)
        low_risk_pixels = np.sum(fire_mask == 1)
        
        # Calculate areas (assuming 10m resolution)
        pixel_area_ha = (10 * 10) / 10000  # 0.01 hectares per pixel
        high_risk_area = high_risk_pixels * pixel_area_ha
        medium_risk_area = medium_risk_pixels * pixel_area_ha
        low_risk_area = low_risk_pixels * pixel_area_ha
        total_fire_area = fire_pixels * pixel_area_ha
        
        # Calculate average temperature in fire areas
        fire_areas = fire_mask > 0
        avg_temp = float(np.mean(temperature_data[fire_areas])) if np.any(fire_areas) else 0
        max_temp = float(np.max(temperature_data)) if temperature_data.size > 0 else 0
        
        # Determine overall risk level
        if high_risk_pixels > 0:
            overall_risk = 'High'
        elif medium_risk_pixels > 0:
            overall_risk = 'Medium'
        elif low_risk_pixels > 0:
            overall_risk = 'Low'
        else:
            overall_risk = 'Normal'

        return jsonify({
            'status': 'success',
            'image': np.array(image_data).tolist() if isinstance(image_data, np.ndarray) else image_data,
            'fire_statistics': {
                'total_fire_area': round(total_fire_area, 2),
                'high_risk_area': round(high_risk_area, 2),
                'medium_risk_area': round(medium_risk_area, 2),
                'low_risk_area': round(low_risk_area, 2),
                'fire_count': int(fire_pixels),
                'overall_risk': overall_risk,
                'average_temperature': round(avg_temp, 1),
                'max_temperature': round(max_temp, 1)
            },
            'detection_date': end_date.isoformat() if isinstance(end_date, date) else end_date
        })
        
    except Exception as e:
        logger.exception("Error processing fire detection request")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/fetch-and-store-fire-data', methods=['POST'])
def fetch_and_store_fire_data():
    try:
        data = request.json
        farmer_id = data.get('farmer_id')
        farm_id = data.get('farm_id')
        
        if not all([farmer_id, farm_id]):
            return jsonify({'status': 'error', 'message': 'farmer_id and farm_id are required'})
        
        geojson_data = get_farm_geojson(farmer_id)
        farm_info = geojson_data['features'][0]['properties']
        
        end_date = data.get('end_date') or datetime.now(timezone.utc).date()
        start_date = data.get('start_date') or (end_date - timedelta(days=1) if isinstance(end_date, date) else 
                                             datetime.strptime(end_date, '%Y-%m-%d').date() - timedelta(days=1))
        
        geometry = get_geometry_from_geojson(geojson_data)
        bbox = get_geojson_bbox(geojson_data)
        size = bbox_to_dimensions(bbox, resolution=10)
        
        config = SHConfig()
        config.sh_client_id = app.config['SH_CLIENT_ID']
        config.sh_client_secret = app.config['SH_CLIENT_SECRET']
        config.instance_id = app.config['SH_INSTANCE_ID']
        
        # Use the same fire detection evalscript
        evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B04", "B08", "B11", "B12", "dataMask"],
        output: [
            { id: "default", bands: 4 },
            { id: "fire_mask", bands: 1, sampleType: "UINT8" }
        ]
    };
}

function evaluatePixel(samples) {
    const nbr = (samples.B08 - samples.B12) / (samples.B08 + samples.B12 + 0.0001);
    const ndvi = (samples.B08 - samples.B04) / (samples.B08 + samples.B04 + 0.0001);
    const brightness_temp = (samples.B11 > 0.1) ? (samples.B11 * 300) : 0;
    
    const fire_condition1 = samples.B12 > 0.1 && samples.B11 > 0.05;
    const fire_condition2 = nbr < -0.2;
    const fire_condition3 = brightness_temp > 320;
    const fire_condition4 = samples.B12 > (samples.B11 * 1.5);
    
    const is_fire = samples.dataMask === 1 && (
        (fire_condition1 && fire_condition3) ||
        (fire_condition1 && fire_condition2) ||
        (fire_condition3 && fire_condition4)
    );
    
    let risk_level = 0;
    if (is_fire) {
        if (brightness_temp > 350 || samples.B12 > 0.3) risk_level = 3;
        else if (brightness_temp > 330 || samples.B12 > 0.2) risk_level = 2;
        else risk_level = 1;
    }
    
    let color;
    if (!samples.dataMask) {
        color = [0, 0, 0, 0];
    } else if (risk_level === 3) {
        color = [1, 0, 0, 1];
    } else if (risk_level === 2) {
        color = [1, 0.5, 0, 1];
    } else if (risk_level === 1) {
        color = [1, 1, 0, 1];
    } else {
        if (ndvi > 0.3) color = [0, 0.7, 0, 0.7];
        else color = [0.6, 0.4, 0.2, 0.7];
    }
    
    return {
        default: color,
        fire_mask: [risk_level]
    };
}"""
        
        request_sh = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date.isoformat(), end_date.isoformat()),
                mosaicking_order='leastCC'
            )],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.PNG),
                SentinelHubRequest.output_response('fire_mask', MimeType.TIFF)
            ],
            geometry=geometry,
            size=size,
            config=config
        )
        
        sh_data = request_sh.get_data()
        if not sh_data:
            return jsonify({'status': 'error', 'message': 'No satellite data available'})
        
        # Process data
        if isinstance(sh_data[0], dict):
            colored_image = sh_data[0].get('default.png')
            fire_mask = sh_data[0].get('fire_mask.tif')
        else:
            colored_image, fire_mask = sh_data[0], sh_data[1]
        
        fire_mask = np.array(fire_mask)
        
        # Calculate statistics
        pixel_area_ha = 0.01  # 10m resolution
        high_risk_pixels = np.sum(fire_mask == 3)
        medium_risk_pixels = np.sum(fire_mask == 2)
        low_risk_pixels = np.sum(fire_mask == 1)
        fire_pixels = np.sum(fire_mask > 0)
        
        high_risk_area = high_risk_pixels * pixel_area_ha
        medium_risk_area = medium_risk_pixels * pixel_area_ha
        low_risk_area = low_risk_pixels * pixel_area_ha
        
        # Create or update fire monitoring record
        detection_datetime = datetime.combine(end_date, datetime.min.time()) if isinstance(end_date, date) else datetime.strptime(end_date, '%Y-%m-%d')
        
        record = FireMonitoringRecord.query.filter_by(
            farmer_id=farmer_id,
            farm_id=farm_id,
            detection_date=detection_datetime
        ).first() or FireMonitoringRecord(
            farmer_id=farmer_id,
            farm_id=farm_id,
            detection_date=detection_datetime
        )
        
        record.farm_name = farm_info.get('name')
        record.fire_count = int(fire_pixels)
        record.high_risk_area = high_risk_area
        record.medium_risk_area = medium_risk_area
        record.low_risk_area = low_risk_area
        record.heatmap_image = numpy_to_base64_image(np.array(colored_image))
        
        db.session.add(record)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Fire monitoring data stored successfully',
            'record_id': record.id,
            'data_summary': {
                'fire_count': record.fire_count,
                'total_risk_area': round(high_risk_area + medium_risk_area + low_risk_area, 2),
                'high_risk_area': round(high_risk_area, 2),
                'detection_date': detection_datetime.isoformat()
            }
        })
        
    except Exception as e:
        logger.exception("Error in fetch_and_store_fire_data")
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/fire-report/<int:farmer_id>/<farm_id>', methods=['GET'])
def generate_fire_report(farmer_id, farm_id):
    try:
        records = FireMonitoringRecord.query.filter_by(
            farmer_id=farmer_id,
            farm_id=farm_id
        ).order_by(desc(FireMonitoringRecord.detection_date)).limit(5).all()
        
        if not records:
            return jsonify({'status': 'error', 'message': 'No fire monitoring data found for this farm'})
        
        current_record = records[0]
        
        # Calculate trend
        trend = "stable"
        if len(records) > 1:
            recent_avg = sum(r.fire_count for r in records[:3]) / min(3, len(records))
            older_avg = sum(r.fire_count for r in records[2:]) / max(1, len(records) - 2)
            if recent_avg > older_avg * 1.2:
                trend = "increasing"
            elif recent_avg < older_avg * 0.8:
                trend = "decreasing"
        
        # Determine risk level
        total_risk_area = (current_record.high_risk_area or 0) + (current_record.medium_risk_area or 0) + (current_record.low_risk_area or 0)
        if current_record.high_risk_area and current_record.high_risk_area > 0:
            risk_level = "High"
        elif current_record.medium_risk_area and current_record.medium_risk_area > 0:
            risk_level = "Medium"
        elif current_record.low_risk_area and current_record.low_risk_area > 0:
            risk_level = "Low"
        else:
            risk_level = "Normal"
        
        report = {
            'status': 'success',
            'report_date': current_record.detection_date.strftime('%d %b %Y'),
            'field_info': {
                'field_name': current_record.farm_name or f'Farm {farm_id}',
                'detection_date': current_record.detection_date.strftime('%d %b %Y %H:%M')
            },
            'fire_analysis': {
                'risk_level': risk_level,
                'fire_count': current_record.fire_count or 0,
                'total_affected_area': round(total_risk_area, 2),
                'high_risk_area': round(current_record.high_risk_area or 0, 2),
                'medium_risk_area': round(current_record.medium_risk_area or 0, 2),
                'low_risk_area': round(current_record.low_risk_area or 0, 2),
                'trend': trend
            },
            'recommendations': generate_fire_recommendations(risk_level, current_record.fire_count or 0)
        }
        
        # Include historical data
        if len(records) > 1:
            report['historical_data'] = [{
                'date': r.detection_date.strftime('%d %b %Y'),
                'fire_count': r.fire_count or 0,
                'total_area': round((r.high_risk_area or 0) + (r.medium_risk_area or 0) + (r.low_risk_area or 0), 2)
            } for r in records[1:]]
        
        if request.args.get('include_image', 'false').lower() == 'true':
            report['heatmap_image'] = current_record.heatmap_image
        
        return jsonify(report)
        
    except Exception as e:
        logger.exception("Error generating fire report")
        return jsonify({'status': 'error', 'message': str(e)})

def generate_fire_recommendations(risk_level, fire_count):
    """Generate fire management recommendations based on risk level and fire count"""
    recommendations = []
    
    if risk_level == "High":
        recommendations.extend([
            "IMMEDIATE ACTION REQUIRED: Contact emergency services if active fires are detected.",
            "Implement firebreaks around high-risk areas immediately.",
            "Ensure water sources and firefighting equipment are readily available.",
            "Consider evacuating livestock and equipment from high-risk zones.",
            "Monitor weather conditions closely - avoid any activities that could spark fires."
        ])
    elif risk_level == "Medium":
        recommendations.extend([
            "Increase fire monitoring frequency in affected areas.",
            "Clear dry vegetation and debris from around buildings and equipment.",
            "Ensure firebreaks are well-maintained and effective.",
            "Restrict burning activities and use of machinery during high-risk periods.",
            "Prepare emergency response plan and ensure all staff are informed."
        ])
    elif risk_level == "Low":
        recommendations.extend([
            "Continue routine fire prevention measures.",
            "Maintain existing firebreaks and access roads.",
            "Monitor weather forecasts for changing fire conditions.",
            "Ensure fire extinguishing equipment is in good working order."
        ])
    else:
        recommendations.extend([
            "Maintain standard fire prevention protocols.",
            "Regular inspection of electrical equipment and machinery.",
            "Keep vegetation management up to date.",
            "Conduct periodic fire risk assessments."
        ])
    
    if fire_count > 10:
        recommendations.append("High fire detection count - consider professional fire risk assessment.")
    
    return recommendations

@app.route('/api/fire-records/<int:farmer_id>')
def get_fire_records(farmer_id):
    try:
        query = FireMonitoringRecord.query.filter_by(farmer_id=farmer_id)
        if request.args.get('farm_id'):
            query = query.filter_by(farm_id=request.args.get('farm_id'))
        
        records = query.order_by(desc(FireMonitoringRecord.detection_date))\
                     .limit(int(request.args.get('limit', 10))).all()
        
        return jsonify({
            'status': 'success',
            'records': [{
                'id': r.id,
                'farm_id': r.farm_id,
                'farm_name': r.farm_name,
                'detection_date': r.detection_date.isoformat(),
                'fire_count': r.fire_count,
                'high_risk_area': r.high_risk_area,
                'medium_risk_area': r.medium_risk_area,
                'low_risk_area': r.low_risk_area,
                'total_risk_area': (r.high_risk_area or 0) + (r.medium_risk_area or 0) + (r.low_risk_area or 0)
            } for r in records],
            'total': len(records)
        })
        
    except Exception as e:
        logger.exception("Error retrieving fire monitoring records")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

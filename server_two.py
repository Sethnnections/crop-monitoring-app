from datetime import datetime, date, timedelta
from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc, and_
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
)
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform
from functools import partial
import pyproj
from scipy.ndimage import gaussian_filter


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crop_monitoring.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
API_BASE_URL = 'http://localhost:3000/api/farmers'

# Database Models
from datetime import datetime, timezone

def utc_now():
    """Helper function to get current UTC time"""
    return datetime.now(timezone.utc)

class CropMonitoringRecord(db.Model):
    __tablename__ = 'crop_monitoring_records'
    
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, nullable=False, index=True)
    farm_id = db.Column(db.String(100), nullable=False, index=True)
    farm_name = db.Column(db.String(200))
    farm_area = db.Column(db.Float)
    crop_name = db.Column(db.String(100))
    sowing_date = db.Column(db.Date)
    growth_stage = db.Column(db.String(200))
    image_date = db.Column(db.Date, nullable=False, index=True)
    cloud_coverage = db.Column(db.Float, default=0.0)
    ndvi_value = db.Column(db.Float)
    reci_value = db.Column(db.Float)
    ndmi_value = db.Column(db.Float)
    ndre_value = db.Column(db.Float)
    msavi_value = db.Column(db.Float)
    precipitation = db.Column(db.Float)
    min_temperature = db.Column(db.Float)
    max_temperature = db.Column(db.Float)
    ndvi_image = db.Column(db.Text)
    reci_image = db.Column(db.Text)
    ndmi_image = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=utc_now)
    updated_at = db.Column(db.DateTime, default=utc_now, onupdate=utc_now)
    
    __table_args__ = (
        db.Index('idx_farmer_farm_date', 'farmer_id', 'farm_id', 'image_date'),
    )
    
class FireMonitoringRecord(db.Model):
    __tablename__ = 'fire_monitoring_records'
    
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, nullable=False, index=True)
    farm_id = db.Column(db.String(100), nullable=False, index=True)
    farm_name = db.Column(db.String(200))
    detection_date = db.Column(db.DateTime, nullable=False, index=True)
    fire_count = db.Column(db.Integer)
    high_risk_area = db.Column(db.Float)  # in hectares
    medium_risk_area = db.Column(db.Float)
    low_risk_area = db.Column(db.Float)
    heatmap_image = db.Column(db.Text)  # base64 encoded
    created_at = db.Column(db.DateTime, default=utc_now)
    
    __table_args__ = (
        db.Index('idx_fire_farmer_farm_date', 'farmer_id', 'farm_id', 'detection_date'),
    )

class WeatherData(db.Model):
    __tablename__ = 'weather_data'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, index=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    precipitation = db.Column(db.Float)
    min_temperature = db.Column(db.Float)
    max_temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    wind_speed = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=utc_now)


with app.app_context():
    db.create_all()

# Helper Functions
def validate_geojson(geojson_data):
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

def get_farm_geojson(farmer_id):
    try:
        url = f"{API_BASE_URL}/{farmer_id}/farms"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('farms') or len(data['farms']) == 0:
            raise ValueError("No farms found for this farmer")
        
        farm = data['farms'][0]
        if not farm.get('boundary'):
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
                "geometry": farm['boundary']
            }]
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing farm data: {e}")
        raise

def transform_geometry(geom, from_crs, to_crs):
    project = partial(
        pyproj.transform,
        pyproj.Proj(init=from_crs),
        pyproj.Proj(init=to_crs)
    )
    return transform(project, geom)

def get_geojson_centroid(geojson_data):
    geojson_data = validate_geojson(geojson_data)
    geom = shape(geojson_data['features'][0]['geometry'])
    if geom.geom_type == 'MultiPolygon':
        geom = geom.convex_hull
    
    projected_geom = transform_geometry(geom, 'epsg:4326', 'epsg:3857')
    centroid = transform_geometry(projected_geom.centroid, 'epsg:3857', 'epsg:4326')
    return [centroid.y, centroid.x]

def get_geojson_bbox(geojson_data):
    geojson_data = validate_geojson(geojson_data)
    geom = shape(geojson_data['features'][0]['geometry'])
    return BBox(bbox=geom.bounds, crs=CRS.WGS84)

def get_geometry_from_geojson(geojson_data):
    return Geometry(validate_geojson(geojson_data)['features'][0]['geometry'], crs=CRS.WGS84)

def numpy_to_base64_image(image_array):
    try:
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        mode = 'RGB' if len(image_array.shape) == 3 else 'L'
        buffer = io.BytesIO()
        Image.fromarray(image_array, mode).save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

def get_weather_data(centroid, date_str):
    return {
        'precipitation': 0.6,
        'min_temperature': 17,
        'max_temperature': 36,
        'humidity': 65,
        'wind_speed': 12
    }

def calculate_growth_stage(sowing_date, current_date):
    if not sowing_date:
        return "Unknown"
    
    days = (current_date - sowing_date).days
    if days < 7: return "BBCH 00-09 Germination"
    if days < 21: return "BBCH 10-19 Leaf development"
    if days < 35: return "BBCH 20-29 Tillering"
    if days < 50: return "BBCH 30-39 Stem elongation"
    if days < 70: return "BBCH 50-59 Inflorescence emergence"
    if days < 90: return "BBCH 60-69 Flowering"
    if days < 120: return "BBCH 70-79 Development of fruit"
    if days < 150: return "BBCH 80-89 Ripening of fruit"
    return "BBCH 90-99 Senescence"

def generate_index_remarks(current, previous, index_type):
    if previous is None:
        return f"Initial {index_type.upper()} measurement recorded."
    
    change = current - previous
    index_type = index_type.lower()
    
    if index_type == 'ndvi':
        if change < -0.05: return "Significant decrease in vegetation health detected. Field scouting recommended."
        if change < -0.02: return "Moderate decrease in NDVI. Monitor closely for potential issues."
        if change > 0.05: return "Excellent vegetation growth observed. Crop health is improving."
        if change > 0.02: return "Positive vegetation development detected."
    elif index_type == 'reci':
        if change < -1.0: return "Significant decrease in chlorophyll levels detected."
        if change < -0.5: return "Moderate decrease in chlorophyll content."
        if change > 1.0: return "Excellent chlorophyll development observed."
    elif index_type == 'ndmi':
        if change < -0.05: return "Increased water stress detected. Consider irrigation."
        if change < -0.02: return "Moderate water stress indication."
        if change > 0.05: return "Good water content levels."
    
    return f"{index_type.upper()} values remain stable."

def get_evalscript_for_index(index_type):
    index_params = {
        'ndvi': {
            'input_bands': '"B04", "B08", "dataMask"',
            'calculation': 'const index_value = (samples.B08 - samples.B04) / (samples.B08 + samples.B04 + 0.0001);',
            'valid_check': 'samples.dataMask === 1',
            'color_scale': """
                if (index_value < 0) color = [0.5, 0.5, 0.5, 1];
                else if (index_value < 0.2) color = [0.8, 0.2, 0.2, 1];
                else if (index_value < 0.4) color = [0.9, 0.6, 0.2, 1];
                else if (index_value < 0.6) color = [0.9, 0.9, 0.2, 1];
                else if (index_value < 0.8) color = [0.2, 0.7, 0.2, 1];
                else color = [0.1, 0.3, 0.1, 1];"""
        },
        'reci': {
            'input_bands': '"B05", "B08", "dataMask"',
            'calculation': 'const index_value = (samples.B08 / samples.B05) - 1;',
            'valid_check': 'samples.dataMask === 1 && samples.B05 > 0',
            'color_scale': """
                if (index_value < 1) color = [0.8, 0.2, 0.2, 1];
                else if (index_value < 3) color = [0.9, 0.6, 0.2, 1];
                else if (index_value < 5) color = [0.9, 0.9, 0.2, 1];
                else if (index_value < 7) color = [0.2, 0.7, 0.2, 1];
                else color = [0.1, 0.3, 0.1, 1];"""
        },
        'ndmi': {
            'input_bands': '"B08", "B11", "dataMask"',
            'calculation': 'const index_value = (samples.B08 - samples.B11) / (samples.B08 + samples.B11 + 0.0001);',
            'valid_check': 'samples.dataMask === 1',
            'color_scale': """
                if (index_value < -0.5) color = [0.8, 0.1, 0.1, 1];
                else if (index_value < -0.2) color = [0.9, 0.5, 0.2, 1];
                else if (index_value < 0) color = [0.9, 0.9, 0.2, 1];
                else if (index_value < 0.2) color = [0.2, 0.7, 0.2, 1];
                else if (index_value < 0.4) color = [0.1, 0.5, 0.8, 1];
                else color = [0.1, 0.2, 0.9, 1];"""
        },
        'ndre': {
            'input_bands': '"B05", "B08", "dataMask"',
            'calculation': 'const index_value = (samples.B08 - samples.B05) / (samples.B08 + samples.B05 + 0.0001);',
            'valid_check': 'samples.dataMask === 1',
            'color_scale': """
                if (index_value < 0) color = [0.8, 0.2, 0.2, 1];
                else if (index_value < 0.2) color = [0.9, 0.6, 0.2, 1];
                else if (index_value < 0.4) color = [0.9, 0.9, 0.2, 1];
                else if (index_value < 0.6) color = [0.2, 0.7, 0.2, 1];
                else color = [0.1, 0.3, 0.1, 1];"""
        },
        'msavi': {
            'input_bands': '"B04", "B08", "dataMask"',
            'calculation': '''
                const NIR = samples.B08;
                const RED = samples.B04;
                const index_value = (2 * NIR + 1 - Math.sqrt((2 * NIR + 1) * (2 * NIR + 1) - 8 * (NIR - RED))) / 2;''',
            'valid_check': 'samples.dataMask === 1',
            'color_scale': """
                if (index_value < 0) color = [0.5, 0.5, 0.5, 1];
                else if (index_value < 0.2) color = [0.8, 0.2, 0.2, 1];
                else if (index_value < 0.4) color = [0.9, 0.6, 0.2, 1];
                else if (index_value < 0.6) color = [0.9, 0.9, 0.2, 1];
                else if (index_value < 0.8) color = [0.2, 0.7, 0.2, 1];
                else color = [0.1, 0.3, 0.1, 1];"""
        }
    }

    params = index_params.get(index_type.lower(), index_params['ndvi'])
    evalscript = f"""
//VERSION=3
function setup() {{
    return {{
        input: [{params['input_bands']}],
        output: [
            {{ id: "default", bands: 4 }},
            {{ id: "index_values", bands: 1, sampleType: "FLOAT32" }}
        ]
    }};
}}

function evaluatePixel(samples) {{
    {params['calculation']}
    
    let color;
    let output_value;
    if ({params['valid_check']}) {{
        {params['color_scale']}
        output_value = index_value;
    }} else {{
        color = [0, 0, 0, 0];
        output_value = 0;
    }}
    
    return {{
        default: color,
        index_values: [output_value]
    }};
}}"""
    return evalscript

def create_sentinelhub_request(evalscript, geometry, time_interval, size, config):
    return SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
            mosaicking_order='leastCC'
        )],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.PNG),
            SentinelHubRequest.output_response('index_values', MimeType.TIFF)
        ],
        geometry=geometry,
        size=size,
        config=config
    )

def process_sh_data(data):
    if isinstance(data, list) and len(data) >= 2:
        return data[0], data[1]
    elif isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        return data[0].get('default.png'), data[0].get('index_values.tif')
    return None, None

# Routes
@app.route('/fetch-and-store-crop-data', methods=['POST'])
def fetch_and_store_crop_data():
    try:
        data = request.json
        farmer_id = data.get('farmer_id')
        farm_id = data.get('farm_id')
        
        if not all([farmer_id, farm_id]):
            return jsonify({'status': 'error', 'message': 'farmer_id and farm_id are required'})
        
        geojson_data = get_farm_geojson(farmer_id)
        farm_info = geojson_data['features'][0]['properties']
        
        start_date = data.get('start_date')
        end_date = data.get('end_date') or datetime.now(timezone.utc).date()
        start_date = start_date or (end_date - timedelta(days=7) if isinstance(end_date, date) else 
                                  datetime.strptime(end_date, '%Y-%m-%d').date() - timedelta(days=7))
        
        centroid = get_geojson_centroid(geojson_data)
        geometry = get_geometry_from_geojson(geojson_data)
        bbox = get_geojson_bbox(geojson_data)
        size = bbox_to_dimensions(bbox, resolution=10)
        
        config = SHConfig()
        config.sh_client_id = app.config['SH_CLIENT_ID']
        config.sh_client_secret = app.config['SH_CLIENT_SECRET']
        config.instance_id = app.config['SH_INSTANCE_ID']
        
        time_interval = (start_date.isoformat() if isinstance(start_date, date) else start_date,
                        end_date.isoformat() if isinstance(end_date, date) else end_date)
        
        results = {}
        for index_type in ['ndvi', 'reci', 'ndmi']:
            try:
                request_sh = create_sentinelhub_request(
                    get_evalscript_for_index(index_type),
                    geometry,
                    time_interval,
                    size,
                    config
                )
                
                sh_data = request_sh.get_data()
                colored_image, index_values = process_sh_data(sh_data)
                
                if index_values is not None:
                    index_values = np.array(index_values)
                    valid_values = index_values[index_values != 0]
                    
                    if valid_values.size > 0:
                        results[index_type] = {
                            'value': float(np.mean(valid_values)),
                            'image': numpy_to_base64_image(np.array(colored_image))
                        }
            except Exception as e:
                logger.error(f"Error fetching {index_type}: {e}")
                continue
        
        weather = get_weather_data(centroid, end_date.isoformat())
        
        record = CropMonitoringRecord.query.filter_by(
            farmer_id=farmer_id,
            farm_id=farm_id,
            image_date=end_date
        ).first() or CropMonitoringRecord(
            farmer_id=farmer_id,
            farm_id=farm_id,
            image_date=end_date
        )
        
        record.farm_name = farm_info.get('name')
        record.farm_area = farm_info.get('size')
        record.crop_name = data.get('crop_name', 'Corn (Maize)')
        
        if data.get('sowing_date'):
            record.sowing_date = datetime.strptime(data['sowing_date'], '%Y-%m-%d').date()
        
        record.growth_stage = calculate_growth_stage(record.sowing_date, end_date)
        
        for index in ['ndvi', 'reci', 'ndmi']:
            if index in results:
                setattr(record, f"{index}_value", results[index]['value'])
                setattr(record, f"{index}_image", results[index]['image'])
        
        record.precipitation = weather.get('precipitation')
        record.min_temperature = weather.get('min_temperature')
        record.max_temperature = weather.get('max_temperature')
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
    try:
        records = CropMonitoringRecord.query.filter_by(
            farmer_id=farmer_id,
            farm_id=farm_id
        ).order_by(desc(CropMonitoringRecord.image_date)).limit(2).all()
        
        if not records:
            return jsonify({'status': 'error', 'message': 'No monitoring data found for this farm'})
        
        current_record, previous_record = records[0], records[1] if len(records) > 1 else None
        
        changes = {}
        if previous_record:
            for index in ['ndvi', 'reci', 'ndmi']:
                current_val = getattr(current_record, f"{index}_value")
                prev_val = getattr(previous_record, f"{index}_value")
                if current_val and prev_val:
                    changes[index] = current_val - prev_val
        
        remarks = {
            index: generate_index_remarks(
                getattr(current_record, f"{index}_value"),
                getattr(previous_record, f"{index}_value") if previous_record else None,
                index
            ) for index in ['ndvi', 'reci', 'ndmi']
        }
        
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
                'clouds': f"{current_record.cloud_coverage:.1f}%" if current_record.cloud_coverage is not None else "0%",
                **{
                    index: {
                        'value': round(getattr(current_record, f"{index}_value"), 2) if getattr(current_record, f"{index}_value") else 'N/A',
                        'change': round(changes.get(index, 0), 2) if index in changes else 'N/A',
                        'remark': remarks[index]
                    } for index in ['ndvi', 'reci', 'ndmi']
                }
            },
            'weather_info': {
                'weekly_precipitation': f"{current_record.precipitation} mm" if current_record.precipitation else 'N/A',
                'min_temperature': f"{current_record.min_temperature}°C" if current_record.min_temperature else 'N/A',
                'max_temperature': f"{current_record.max_temperature}°C" if current_record.max_temperature else 'N/A'
            }
        }
        
        if previous_record:
            report['previous_period'] = {
                'image_date': previous_record.image_date.strftime('%d %b %Y'),
                'clouds': f"{previous_record.cloud_coverage:.1f}%" if previous_record.cloud_coverage is not None else "0%",
                **{
                    index: round(getattr(previous_record, f"{index}_value"), 2) if getattr(previous_record, f"{index}_value") else 'N/A'
                    for index in ['ndvi', 'reci', 'ndmi']
                }
            }
        
        if request.args.get('include_images', 'false').lower() == 'true':
            report['images'] = {
                index: getattr(current_record, f"{index}_image")
                for index in ['ndvi', 'reci', 'ndmi']
            }
        
        return jsonify(report)
        
    except Exception as e:
        logger.exception("Error generating crop report")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/monitoring/<int:farmer_id>')
def crop_monitoring(farmer_id):
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
            'crop_monitoring.html',
            center=get_geojson_centroid(initial_geojson),
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

@app.route('/get-crop-index', methods=['POST'])
def get_crop_index():
    try:
        geojson_data = request.json
        index_type = geojson_data.get('index_type', 'ndvi').lower()
        
        start_date = geojson_data.get('start_date')
        end_date = geojson_data.get('end_date') or datetime.now(timezone.utc).date()
        start_date = start_date or (end_date - timedelta(days=30) if isinstance(end_date, date) else 
                                  datetime.strptime(end_date, '%Y-%m-%d').date() - timedelta(days=30))
        
        geometry = get_geometry_from_geojson(geojson_data)
        bbox = get_geojson_bbox(geojson_data)
        size = bbox_to_dimensions(bbox, resolution=10)
        
        config = SHConfig()
        config.sh_client_id = app.config['SH_CLIENT_ID']
        config.sh_client_secret = app.config['SH_CLIENT_SECRET']
        config.instance_id = app.config['SH_INSTANCE_ID']
        
        request_sh = create_sentinelhub_request(
            get_evalscript_for_index(index_type),
            geometry,
            (start_date.isoformat() if isinstance(start_date, date) else start_date,
             end_date.isoformat() if isinstance(end_date, date) else end_date),
            size,
            config
        )
        
        data = request_sh.get_data()
        image_data, index_values = process_sh_data(data)
        
        if index_values is None:
            return jsonify({'status': 'error', 'message': 'No data available for the selected time range'})
        
        index_values = np.array(index_values)
        valid_values = index_values[index_values != 0]
        
        if valid_values.size > 0:
            stats = {
                'average_value': float(np.mean(valid_values)),
                'min_value': float(np.min(valid_values)),
                'max_value': float(np.max(valid_values))
            }
        else:
            stats = {'average_value': 0, 'min_value': 0, 'max_value': 0}
        
        return jsonify({
            'status': 'success',
            'image': image_data.tolist() if isinstance(image_data, np.ndarray) else image_data,
            **stats,
            'index_type': index_type
        })
        
    except Exception as e:
        logger.exception(f"Error processing crop index request: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

def calculate_risk_level(water_coverage):
    if water_coverage > 30: return 'High'
    if water_coverage > 15: return 'Medium'
    if water_coverage > 5: return 'Low'
    return 'Normal'

@app.route('/flood-monitoring/<int:farmer_id>')
def flood_monitoring(farmer_id):
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
            'flood_monitoring.html',
            center=get_geojson_centroid(initial_geojson),
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
        start_date = request.args.get('start_date') or geojson_data.get('start_date')
        end_date = request.args.get('end_date') or geojson_data.get('end_date') or datetime.now(timezone.utc).date()
        start_date = start_date or date(end_date.year, end_date.month, 1) if isinstance(end_date, date) else \
                    datetime.strptime(end_date, '%Y-%m-%d').replace(day=1).date()
        
        geometry = get_geometry_from_geojson(geojson_data)
        bbox = get_geojson_bbox(geojson_data)
        size = bbox_to_dimensions(bbox, resolution=10)
        
        config = SHConfig()
        config.sh_client_id = app.config['SH_CLIENT_ID']
        config.sh_client_secret = app.config['SH_CLIENT_SECRET']
        config.instance_id = app.config['SH_INSTANCE_ID']
        
        evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B03", "B08", "B11", "dataMask"],
        output: [
            { id: "default", bands: 4 },
            { id: "water_mask", bands: 1, sampleType: "UINT8" }
        ]
    };
}

function evaluatePixel(samples) {
    const ndwi = (samples.B03 - samples.B08) / (samples.B03 + samples.B08 + 0.0001);
    const mndwi = (samples.B03 - samples.B11) / (samples.B03 + samples.B11 + 0.0001);
    const isWater = (ndwi > 0.2 || mndwi > 0) && samples.dataMask === 1;
    
    return {
        default: !samples.dataMask ? [0, 0, 0, 0] : 
                isWater ? [0, 0, 0.5 + Math.max(ndwi, mndwi) * 0.5, 1] : [0.9, 0.9, 0.9, 1],
        water_mask: [isWater ? 1 : 0]
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
                SentinelHubRequest.output_response('water_mask', MimeType.TIFF)
            ],
            geometry=geometry,
            size=size,
            config=config
        )

        data = request_sh.get_data()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data available'})
        
        if isinstance(data[0], dict):
            image_data = data[0].get('default.png')
            water_mask = data[0].get('water_mask.tif')
        else:
            image_data, water_mask = data[0], data[1]
        
        water_mask = np.array(water_mask)
        total_pixels = np.sum(water_mask >= 0)
        water_pixels = np.sum(water_mask == 1)
        water_coverage = (water_pixels / total_pixels * 100) if total_pixels > 0 else 0

        return jsonify({
            'status': 'success',
            'image': np.array(image_data).tolist(),
            'water_coverage': round(water_coverage, 2),
            'risk_level': calculate_risk_level(water_coverage)
        })
        
    except Exception as e:
        logger.exception("Error processing water detection request")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/monitoring-records/<int:farmer_id>')
def get_monitoring_records(farmer_id):
    try:
        query = CropMonitoringRecord.query.filter_by(farmer_id=farmer_id)
        if request.args.get('farm_id'):
            query = query.filter_by(farm_id=request.args.get('farm_id'))
        
        records = query.order_by(desc(CropMonitoringRecord.image_date))\
                     .limit(int(request.args.get('limit', 10))).all()
        
        return jsonify({
            'status': 'success',
            'records': [{
                'id': r.id,
                'farm_id': r.farm_id,
                'farm_name': r.farm_name,
                'image_date': r.image_date.isoformat(),
                'ndvi_value': r.ndvi_value,
                'reci_value': r.reci_value,
                'ndmi_value': r.ndmi_value,
                'growth_stage': r.growth_stage,
                'precipitation': r.precipitation,
                'min_temperature': r.min_temperature,
                'max_temperature': r.max_temperature
            } for r in records],
            'total': len(records)
        })
        
    except Exception as e:
        logger.exception("Error retrieving monitoring records")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/text-report/<int:farmer_id>/<farm_id>', methods=['GET'])
def text_report(farmer_id, farm_id):
    try:
        records = CropMonitoringRecord.query.filter_by(
            farmer_id=farmer_id,
            farm_id=farm_id
        ).order_by(desc(CropMonitoringRecord.image_date)).limit(2).all()
        
        if not records:
            return "No monitoring data found for this farm", 404
        
        current_record, previous_record = records[0], records[1] if len(records) > 1 else None
        
        changes = {}
        if previous_record:
            for index in ['ndvi', 'reci', 'ndmi']:
                current_val = getattr(current_record, f"{index}_value")
                prev_val = getattr(previous_record, f"{index}_value")
                if current_val and prev_val:
                    changes[index] = current_val - prev_val
        
        remarks = {
            index: generate_index_remarks(
                getattr(current_record, f"{index}_value"),
                getattr(previous_record, f"{index}_value") if previous_record else None,
                index
            ) for index in ['ndvi', 'reci', 'ndmi']
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
  

if __name__ == '__main__':
    app.run(debug=True)
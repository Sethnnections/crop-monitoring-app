import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-123')
    
    # Sentinel Hub configuration
    SH_CLIENT_ID = os.getenv('SH_CLIENT_ID')
    SH_CLIENT_SECRET = os.getenv('SH_CLIENT_SECRET')
    SH_INSTANCE_ID = os.getenv('SH_INSTANCE_ID')
    
    # Default map settings
    DEFAULT_CENTER = [0, 0] 
    DEFAULT_ZOOM = 12
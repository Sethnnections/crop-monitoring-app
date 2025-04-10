document.addEventListener('DOMContentLoaded', function() {
    // Initialize map
    const map = new ol.Map({
        target: 'map',
        layers: [
            new ol.layer.Tile({
                source: new ol.source.OSM()
            })
        ],
        view: new ol.View({
            center: ol.proj.fromLonLat([mapCenter[1], mapCenter[0]]),
            zoom: 12
        })
    });

    // Validate and process GeoJSON data
    function validateAndProcessGeoJSON(geojson) {
        try {
            if (!geojson || !geojson.type || geojson.type !== 'FeatureCollection') {
                throw new Error('Invalid GeoJSON: Must be a FeatureCollection');
            }

            if (!geojson.features || !Array.isArray(geojson.features) || geojson.features.length === 0) {
                throw new Error('Invalid GeoJSON: No features found');
            }

            // Check each feature's geometry type
            const validGeometries = ['Polygon', 'MultiPolygon'];
            for (const feature of geojson.features) {
                if (!feature.geometry || !validGeometries.includes(feature.geometry.type)) {
                    throw new Error(`Invalid geometry type: Only Polygon and MultiPolygon are supported (found ${feature.geometry?.type})`);
                }
            }

            return geojson;
        } catch (error) {
            console.error('GeoJSON validation error:', error);
            throw error;
        }
    }

    // Process the initial GeoJSON data
    let processedGeoJSON;
    try {
        processedGeoJSON = validateAndProcessGeoJSON(geojsonData);
    } catch (error) {
        alert('Error loading map: ' + error.message);
        return;
    }

    // Add GeoJSON layer with error handling
    let vectorSource;
    try {
        vectorSource = new ol.source.Vector({
            features: new ol.format.GeoJSON().readFeatures(processedGeoJSON, {
                featureProjection: 'EPSG:3857',
                dataProjection: 'EPSG:4326'
            })
        });
    } catch (error) {
        console.error('Error creating vector source:', error);
        alert('Error displaying boundary: ' + error.message);
        return;
    }

    const vectorLayer = new ol.layer.Vector({
        source: vectorSource,
        style: new ol.style.Style({
            stroke: new ol.style.Stroke({
                color: 'rgb(0, 255, 123)',
                width: 2
            }),
            fill: new ol.style.Fill({
                color: 'rgba(0, 0, 255, 0.1)'
            })
        })
    });

    map.addLayer(vectorLayer);

    // Fit view to GeoJSON extent with padding
    try {
        map.getView().fit(vectorSource.getExtent(), { 
            padding: [50, 50, 50, 50],
            duration: 1000 // Smooth animation
        });
    } catch (error) {
        console.error('Error fitting view:', error);
    }

    // NDVI layer reference
    let ndviLayer = null;

    // Enhanced NDVI button click handler
    document.getElementById('load-ndvi').addEventListener('click', function() {
        const button = this;
        button.disabled = true;
        button.textContent = 'Loading...';

        // Show loading indicator
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'loading-indicator';
        loadingIndicator.innerHTML = '<div class="spinner"></div><p>Processing NDVI data...</p>';
        document.body.appendChild(loadingIndicator);

        // Validate GeoJSON before sending
        let validGeoJSON;
        try {
            validGeoJSON = validateAndProcessGeoJSON(geojsonData);
        } catch (error) {
            console.error('GeoJSON validation failed:', error);
            button.disabled = false;
            button.textContent = 'Load NDVI';
            document.body.removeChild(loadingIndicator);
            alert('Invalid boundary: ' + error.message);
            return;
        }

        fetch('/get-ndvi', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(validGeoJSON)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            // Clean up UI
            button.disabled = false;
            button.textContent = 'Load NDVI';
            document.body.removeChild(loadingIndicator);

            if (data.status === 'success') {
                // Remove previous NDVI layer if exists
                if (ndviLayer) {
                    map.removeLayer(ndviLayer);
                    ndviLayer = null;
                }

                // Process the NDVI image data
                processNdviImage(data.image, vectorSource)
                    .then(imageLayer => {
                        ndviLayer = imageLayer;
                        map.addLayer(ndviLayer);
                        
                        // Zoom to the NDVI layer with animation
                        map.getView().fit(vectorSource.getExtent(), { 
                            padding: [50, 50, 50, 50],
                            duration: 1000
                        });
                    })
                    .catch(error => {
                        console.error('NDVI processing error:', error);
                        alert('Error displaying NDVI: ' + error.message);
                    });
            } else {
                throw new Error(data.message || 'Unknown error occurred');
            }
        })
        .catch(error => {
            console.error('NDVI request failed:', error);
            button.disabled = false;
            button.textContent = 'Load NDVI';
            document.body.removeChild(loadingIndicator);
            alert('NDVI request failed: ' + error.message);
        });
    });

    /**
     * Processes NDVI image data and creates an OpenLayers image layer
     * @param {Array} imageData - 3D array of image data from server
     * @param {ol.source.Vector} source - Vector source for extent reference
     * @returns {Promise<ol.layer.Image>} Promise resolving to the image layer
     */
    function processNdviImage(imageData, source) {
        return new Promise((resolve, reject) => {
            try {
                if (!imageData || !Array.isArray(imageData) || imageData.length === 0) {
                    throw new Error('No image data received');
                }

                // Get dimensions
                const height = imageData.length;
                const width = imageData[0].length;
                
                // Flatten the 3D array into a 1D array of RGBA values
                const flatPixels = [];
                for (const row of imageData) {
                    if (!Array.isArray(row) || row.length !== width) {
                        throw new Error('Inconsistent image data structure');
                    }
                    for (const pixel of row) {
                        if (!Array.isArray(pixel) || pixel.length !== 4) {
                            throw new Error('Invalid pixel format');
                        }
                        flatPixels.push(...pixel);
                    }
                }

                // Create canvas and context
                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                
                // Create ImageData
                const imageArray = new Uint8ClampedArray(flatPixels);
                const imageDataObj = new ImageData(imageArray, width, height);
                
                // Draw on canvas
                ctx.putImageData(imageDataObj, 0, 0);
                
                // Create image layer
                const extent = source.getExtent();
                const imageUrl = canvas.toDataURL('image/png');
                
                const imageLayer = new ol.layer.Image({
                    source: new ol.source.ImageStatic({
                        url: imageUrl,
                        imageExtent: extent,
                        projection: 'EPSG:3857'
                    }),
                    opacity: 0.8
                });

                resolve(imageLayer);
            } catch (error) {
                reject(error);
            }
        });
    }

        // Checkbox toggle for NDVI layer
        document.getElementById('toggle-ndvi').addEventListener('change', function () {
            if (ndviLayer) {
                if (this.checked) {
                    map.addLayer(ndviLayer);
                    document.getElementById('ndvi-legend').style.display = 'block';
                } else {
                    map.removeLayer(ndviLayer);
                    document.getElementById('ndvi-legend').style.display = 'none';
                }
            }
        });


    // Add NDVI legend
    createNdviLegend();
    
    /**
     * Creates and styles the NDVI legend with enhanced visualization
     */
    function createNdviLegend() {
        const legend = document.getElementById('ndvi-legend');
        
        const gradientStops = [
            { color: '#050505', value: -1.0, label: 'Water/Built-up (NDVI < -0.5)' },
            { color: '#bfbfbf', value: -0.5, label: 'Bare Soil/Urban (-0.5 ≤ NDVI < -0.2)' },
            { color: '#dbdbdb', value: -0.2, label: 'Dry Soil/Rock (-0.2 ≤ NDVI < -0.1)' },
            { color: '#ebebeb', value: -0.1, label: 'Sand/Snow (-0.1 ≤ NDVI < 0)' },
            { color: '#ffffff', value: 0.0, label: 'No Vegetation (0 ≤ NDVI < 0.05)' },
            { color: '#ff3333', value: 0.05, label: 'Stress/Burned (0.05 ≤ NDVI < 0.1)' },
            { color: '#ff9900', value: 0.1, label: 'Sparse Grass/Shrubs (0.1 ≤ NDVI < 0.15)' },
            { color: '#ffe600', value: 0.15, label: 'Moderate Grassland (0.15 ≤ NDVI < 0.2)' },
            { color: '#ffff33', value: 0.2, label: 'Healthy Pasture (0.2 ≤ NDVI < 0.3)' },
            { color: '#ccff33', value: 0.3, label: 'Shrubs/Brush (0.3 ≤ NDVI < 0.4)' },
            { color: '#66e619', value: 0.4, label: 'Moderate Vegetation (0.4 ≤ NDVI < 0.5)' },
            { color: '#33b333', value: 0.5, label: 'Dense Grass/Crops (0.5 ≤ NDVI < 0.6)' },
            { color: '#1a801a', value: 0.6, label: 'Deciduous Forest (0.6 ≤ NDVI < 0.7)' },
            { color: '#004d00', value: 1.0, label: 'Evergreen Forest (NDVI ≥ 0.7)' }
        ];
        
        // Create gradient string
        let gradientString = 'linear-gradient(to bottom, ';
        gradientStops.forEach((stop, index) => {
            const position = ((stop.value + 1) / 2) * 100; // Scale from [-1,1] to [0,100]
            gradientString += `${stop.color} ${position}%`;
            if (index < gradientStops.length - 1) {
                gradientString += ', ';
            }
        });
        
        // Apply to legend
        const gradientElement = legend.querySelector('.legend-gradient');
        gradientElement.style.background = gradientString;

        // Add value markers and labels
        const labelsContainer = legend.querySelector('.legend-labels');
        labelsContainer.innerHTML = '';
        
        gradientStops.forEach(stop => {
            if (stop.label) {
                const position = ((stop.value + 1) / 2) * 100;
                const labelElement = document.createElement('div');
                labelElement.className = 'legend-label';
                labelElement.style.bottom = `${position}%`;
                labelElement.innerHTML = `<span class="legend-value">${stop.value.toFixed(2)}</span>
                                         <span class="legend-text">${stop.label}</span>`;
                labelsContainer.appendChild(labelElement);
            }
        });
    }

    // Add CSS for loading indicator
    const style = document.createElement('style');
    style.textContent = `
        .loading-indicator {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            color: white;
            font-size: 1.2em;
        }
        .spinner {
            border: 5px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 5px solid #fff;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin-bottom: 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .legend-label {
            position: absolute;
            left: 110%;
            transform: translateY(50%);
            white-space: nowrap;
        }
        .legend-value {
            font-weight: bold;
            margin-right: 5px;
        }
    `;
    document.head.appendChild(style);
});
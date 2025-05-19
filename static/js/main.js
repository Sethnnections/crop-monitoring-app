        // Pass Flask variables to JavaScript
        const mapCenter = {{ center| tojson }};
        const geojsonData = {{ geojson| safe }};
        const farmerName = "{{ farmer.name }} {{ farmer.surname }}";
        let ndviLayer = null;
        let ndviImageData = null;
        
        // Initialize map with ArcGIS World Imagery
        const map = new ol.Map({
            target: 'map',
            layers: [
                new ol.layer.Tile({
                    source: new ol.source.XYZ({
                        url: "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        attributions: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
                    })
                })
            ],
            view: new ol.View({
                center: ol.proj.fromLonLat([mapCenter[1], mapCenter[0]]),
                zoom: 12
            })
        });

        // Value display element
        const valueDisplay = document.getElementById('value-display');

        // Initialize date pickers with default values (current month)
        function initializeDatePickers() {
            const today = new Date();
            const firstDayOfMonth = new Date(today.getFullYear(), today.getMonth(), 1);
            
            document.getElementById('start-date').valueAsDate = firstDayOfMonth;
            document.getElementById('end-date').valueAsDate = today;
            
            updateSelectedPeriodDisplay();
        }
        
        // Update the selected period display text
        function updateSelectedPeriodDisplay() {
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            
            if (startDate && endDate) {
                const formattedStart = new Date(startDate).toLocaleDateString();
                const formattedEnd = new Date(endDate).toLocaleDateString();
                document.getElementById('selected-period').textContent = `Selected period: ${formattedStart} to ${formattedEnd}`;
            }
        }

        // Initialize date pickers when the page loads
        initializeDatePickers();
        
        // Add event listeners for date changes
        document.getElementById('start-date').addEventListener('change', updateSelectedPeriodDisplay);
        document.getElementById('end-date').addEventListener('change', updateSelectedPeriodDisplay);

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
            }),
            name: 'farm-boundary'
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

        // Farm selector functionality
        document.getElementById('farm-select').addEventListener('change', function() {
            const selectedOption = this.options[this.selectedIndex];
            const farmGeoJSON = JSON.parse(selectedOption.dataset.geojson);
            
            // Update farm details
            document.getElementById('farm-name').textContent = farmGeoJSON.features[0].properties.name;
            document.getElementById('farm-location').textContent = farmGeoJSON.features[0].properties.location;
            document.getElementById('farm-size').textContent = `${farmGeoJSON.features[0].properties.size} hectares`;
            
            // Update the map with the new farm boundary
            updateMapWithNewFarm(farmGeoJSON);
        });
        
        function updateMapWithNewFarm(geojson) {
            // Remove existing vector layer if it exists
            map.getLayers().forEach(layer => {
                if (layer.get('name') === 'farm-boundary') {
                    map.removeLayer(layer);
                }
            });
            
            // Create new vector source and layer
            const vectorSource = new ol.source.Vector({
                features: new ol.format.GeoJSON().readFeatures(geojson, {
                    featureProjection: 'EPSG:3857',
                    dataProjection: 'EPSG:4326'
                })
            });
            
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
                }),
                name: 'farm-boundary'
            });
            
            map.addLayer(vectorLayer);
            
            // Fit view to the new farm's extent
            map.getView().fit(vectorSource.getExtent(), {
                padding: [50, 50, 50, 50],
                duration: 1000
            });
            
            // Clear any existing NDVI data
            if (ndviLayer) {
                map.removeLayer(ndviLayer);
                ndviLayer = null;
            }
            ndviImageData = null;
            document.getElementById('toggle-ndvi').checked = false;
            document.getElementById('ndvi-legend').style.display = 'none';
        }

        // Enhanced NDVI button click handler with date range
        document.getElementById('load-ndvi').addEventListener('click', function () {
            const button = this;
            button.disabled = true;
            button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';

            // Show loading indicator
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'loading-indicator';
            loadingIndicator.innerHTML = '<div class="spinner"></div><p>Processing NDVI data...</p>';
            document.body.appendChild(loadingIndicator);

            // Get current farm's GeoJSON
            const selectedOption = document.getElementById('farm-select').options[document.getElementById('farm-select').selectedIndex];
            const currentGeoJSON = JSON.parse(selectedOption.dataset.geojson);

            // Get date range values
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;

            // Validate dates
            if (!startDate || !endDate) {
                alert('Please select both start and end dates');
                button.disabled = false;
                button.innerHTML = '<i class="bi bi-cloud-arrow-down"></i> Load NDVI Data';
                document.body.removeChild(loadingIndicator);
                return;
            }

            if (new Date(startDate) > new Date(endDate)) {
                alert('End date must be after start date');
                button.disabled = false;
                button.innerHTML = '<i class="bi bi-cloud-arrow-down"></i> Load NDVI Data';
                document.body.removeChild(loadingIndicator);
                return;
            }

            // Add dates to the GeoJSON payload
            currentGeoJSON.start_date = startDate;
            currentGeoJSON.end_date = endDate;

            // Validate GeoJSON before sending
            let validGeoJSON;
            try {
                validGeoJSON = validateAndProcessGeoJSON(currentGeoJSON);
            } catch (error) {
                console.error('GeoJSON validation failed:', error);
                button.disabled = false;
                button.innerHTML = '<i class="bi bi-cloud-arrow-down"></i> Load NDVI Data';
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
                    button.innerHTML = '<i class="bi bi-cloud-arrow-down"></i> Load NDVI Data';
                    document.body.removeChild(loadingIndicator);

                    if (data.status === 'success') {
                        // Remove previous NDVI layer if exists
                        if (ndviLayer) {
                            map.removeLayer(ndviLayer);
                            ndviLayer = null;
                        }

                        // Store the image data for hover functionality
                        ndviImageData = data.image;

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

                                // Show legend by default
                                document.getElementById('ndvi-legend').style.display = 'block';
                                document.getElementById('toggle-ndvi').checked = true;
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
                    button.innerHTML = '<i class="bi bi-cloud-arrow-down"></i> Load NDVI Data';
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
                        opacity: 0.6
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

        // Opacity slider control
        const opacitySlider = document.getElementById('opacity-slider');
        const opacityValue = document.querySelector('.opacity-value');

        opacitySlider.addEventListener('input', function () {
            const opacity = parseFloat(this.value);
            opacityValue.textContent = `${Math.round(opacity * 100)}%`;

            if (ndviLayer) {
                ndviLayer.setOpacity(opacity);
            }
        });

        // Mouse move handler for value display
        map.on('pointermove', function (evt) {
            if (!ndviImageData || !ndviLayer || !document.getElementById('toggle-ndvi').checked) {
                valueDisplay.style.display = 'none';
                return;
            }

            const coordinate = evt.coordinate;
            const extent = ndviLayer.getSource().getImageExtent();
            const resolution = map.getView().getResolution();

            // Check if the coordinate is within the NDVI image extent
            if (!ol.extent.containsCoordinate(extent, coordinate)) {
                valueDisplay.style.display = 'none';
                return;
            }

            // Calculate pixel position in the image
            const width = ndviImageData[0].length;
            const height = ndviImageData.length;

            const x = Math.floor((coordinate[0] - extent[0]) / (extent[2] - extent[0]) * width);
            const y = Math.floor((extent[3] - coordinate[1]) / (extent[3] - extent[1]) * height);

            // Ensure we're within bounds
            if (x >= 0 && x < width && y >= 0 && y < height) {
                const pixel = ndviImageData[y][x];

                // Calculate NDVI value from RGBA (assuming server sends NDVI encoded in RGBA)
                // This calculation depends on how your server encodes NDVI values in the image
                // Here's a common approach where NDVI is encoded in the red channel
                const ndviValue = (pixel[0] / 255 * 2) - 1; // Scale from [0,255] to [-1,1]

                valueDisplay.textContent = `NDVI: ${ndviValue.toFixed(4)}`;
                valueDisplay.style.display = 'block';
            } else {
                valueDisplay.style.display = 'none';
            }
        });

        // Hide value display when mouse leaves map
        map.getViewport().addEventListener('mouseout', function () {
            valueDisplay.style.display = 'none';
        });

        // Add NDVI legend
        createNdviLegend();

        /**
         * Creates and styles the NDVI legend with enhanced visualization
         */
        function createNdviLegend() {
            const legend = document.getElementById('ndvi-legend');

            // Enhanced color gradient with more stops for better visualization
            const gradientStops = [
                { color: '#050505', value: -1.0, label: 'Excellent Health'},
                { color: '#dbdbdb', value: -0.7, label: 'Good Health' },
                { color: '#ff0000', value: -0.5, label: 'Moderate Health' },
                { color: '#ff9900', value: 0.5, label: 'Highly Stressed' },
                { color: '#66e619', value: 0.7, label: 'Bare Soil'},
                { color: '#1a801a', value: 1.0, label: 'No Data/Water' },
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

        // Logout button functionality
        document.querySelector('.logout-btn').addEventListener('click', function() {
            if (confirm('Are you sure you want to logout?')) {
                // Add logout functionality here
                window.location.href = '/logout';
            }
        });

        // Apply dates button functionality
        document.getElementById('apply-dates').addEventListener('click', function() {
            // This just updates the display - the actual dates are used when loading NDVI
            updateSelectedPeriodDisplay();
        });

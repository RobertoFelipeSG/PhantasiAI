// EMG and Accel graph management (Chart.js objects)

/* VARIABLES */
let emgGraph = null;
let accelGraph = null;
let annotations = [];
let pendingEmgData = []; // EMG Data Buffer
let pendingAccelData = []; // Accel Data Buffer
let isGraphPaused = false;

/* FUNCTIONS */
// Initialize graphs
function getCommonChartOptions(yLabel) {
    return {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: { 
            legend: { display: true },
            annotation: { annotations: [] } // event marker lines
        },
        parsing: false, // disable automatic parsing 
        normalized: true, // data is already sorted 
        scales: { 
            x: { 
                type: 'linear',
                min: undefined,
                max: undefined,
                title: { display: true, text: 'Time (s)', color: '#333' },
                id: 'x', // reference for annotations 
                ticks: { display: true, color: '#333', maxRotation: 0, minRotation: 0, autoSkip: true, maxTicksLimit: 10,
                        callback: function(value) {
                            const numericValue = parseFloat(this.getLabelForValue(value));
                            return Number.isFinite(numericValue) ? numericValue.toFixed(1) : this.getLabelForValue(value); }
                        }, 
                grid: { display: true, lineWidth: 0.5 }
            }, 
            y: { 
                title: { display: true, text: yLabel },
                grid: { display: true, lineWidth: 0.5 }
            }}
    }
};

function initGraphs() {
    // 1: EMG Graph
    const ctxEmg = document.getElementById('emgGraph').getContext('2d');
    emgGraph = new Chart(ctxEmg, {
        type: 'line',
        data: { 
            datasets: [{
                label: 'Channel 1 (EMG Signal)',
                data: [], 
                borderColor: 'blue',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                tension: 0.1 // Change for slight curve
            }]
        },
        options: getCommonChartOptions('Amplitude (µV)') 
    });
    
    // 2: Accel Graph
    const ctxAccel = document.getElementById('accelGraph').getContext('2d');
    accelGraph = new Chart(ctxAccel, {
        type: 'line',
        data: { 
            datasets: [
                { label: 'X', data: [], borderColor: 'red', borderWidth: 1.5, pointRadius: 0 },
                { label: 'Y', data: [], borderColor: 'green', borderWidth: 1.5, pointRadius: 0 },
                { label: 'Z', data: [], borderColor: 'blue', borderWidth: 1.5, pointRadius: 0 }
            ]
        },
        options: getCommonChartOptions('G-Force') 
    });

}

// Real-time data updates
function batchUpdateGraph(chart, buffer, type) {
    if (!chart || buffer.length === 0) return;

    // 1: Push all pending data to Chart dataset
    if (type === 'emg') {
        chart.data.datasets[0].data.push(...buffer);
    } 
    else if (type == 'accel') {
        buffer.forEach(pt => {
            chart.data.datasets[0].data.push({ x: pt.x, y: pt.y_x });
            chart.data.datasets[1].data.push({ x: pt.x, y: pt.y_y });
            chart.data.datasets[2].data.push({ x: pt.x, y: pt.y_z });
        });
    }

    buffer.length = 0 // 2: Clear buffer

    // 3: Prune graph (along x-axis) 
    const currentLength = chart.data.datasets[0].data.length;
    if (currentLength > MAX_DATA_POINTS) {
        const removeCount = currentLength - MAX_DATA_POINTS;
        chart.data.datasets.forEach(dataset => {
            dataset.data.splice(0, removeCount);
        });
    }
        // Graph configuration    
    // 4: Update x-axis (time) window 
    const lastPoint = chart.data.datasets[0].data[chart.data.datasets[0].data.length - 1];
    if (lastPoint) {
        const latestTime = lastPoint.x;
        
        // Manually set the view window for smooth scrolling effect
        chart.options.scales.x.min = latestTime - WINDOW_SIZE;
        chart.options.scales.x.max = latestTime;
    }

    // 5: Marker interval filtering
    annotations = annotations.filter(a => a.value >= chart.options.scales.x.min);

    chart.options.plugins.annotation.annotations = annotations.filter(a => 
        a.value <= chart.options.scales.x.max
    );

    chart.update('none'); // 6: Render
}

// Event marker display
function updateEventDisplay(timestamp) {
    if (isGraphPaused) return;
    
    timestamp = Number(timestamp);
    
    const annotationConfig = {
    type: 'line',
    mode: 'vertical',
    scaleID: 'x',
    value: timestamp,
    borderColor: 'red',
    borderWidth: 1,
    label: {
        content: 'EVENT',
        enabled: true,
        position: 'top',
        backgroundColor: 'rgba(255, 0, 0, 0.7)',
        color: 'white'
        }
    };

    annotations.push(annotationConfig); // update annotations
}

// Trial marker display
function updateTrialDisplay(timestamp) {
    if (isGraphPaused) return;
    
    timestamp = Number(timestamp);
    
    const annotationConfig = {
    type: 'line',
    mode: 'vertical',
    scaleID: 'x',
    value: timestamp,
    borderColor: 'green',
    borderWidth: 1,
    label: {
        content: 'TRIAL',
        enabled: true,
        position: 'top',
        backgroundColor: 'rgba(0, 255, 8, 0.7)',
        color: 'white'
        }
    };

    annotations.push(annotationConfig); // update annotations
}

// Pause feature (synchronized)
function togglePauseGraph() {
    isGraphPaused = !isGraphPaused;
    document.getElementById('pauseGraphBtn').innerText = isGraphPaused ? 'Resume Graphs' : 'Pause Graphs';
}

// Zoom feature (individualized)
function getGraphRangeY(graph) { // Calculate min/max of graphs current data
    let min = Infinity;
    let max = -Infinity;
    let hasData = false;

    // Loop through all datasets (handles Accel X, Y, Z simultaneously)
    graph.data.datasets.forEach(dataset => {
        dataset.data.forEach(point => {
            // Handle data whether it is a raw number (EMG) or an object {x, y} (Accel)
            const val = (typeof point === 'object' && point !== null) ? point.y : point;
            
            if (val < min) min = val;
            if (val > max) max = val;
            hasData = true;
        });
    });

    if (!hasData || min === Infinity || max === -Infinity) return null;
    
    // Add a tiny buffer if flatline
    if (min === max) {
        min -= 1;
        max += 1;
    }
    return { min, max };
}

function applyZoom(graph, factor) { // Zoom in/out Function
    if (!graph) return;

    // Get current scale limits
    let currentMin = graph.options.scales.y.min;
    let currentMax = graph.options.scales.y.max;

    // If manual zoom isn't set yet, calculate it from the data
    if (currentMin === undefined || currentMax === undefined) {
        const limits = getGraphRangeY(graph);
        if (!limits) return; // No data to zoom on
        currentMin = limits.min;
        currentMax = limits.max;
    }

    // Calculate center and span
    const center = (currentMin + currentMax) / 2;
    const span = (currentMax - currentMin);
    
    // Apply Zoom Factor (newSpan = oldSpan * factor)
    // Factor < 1 zooms IN (smaller span), Factor > 1 zooms OUT (larger span)
    const newHalfSpan = (span * factor) / 2;

    graph.options.scales.y.min = center - newHalfSpan;
    graph.options.scales.y.max = center + newHalfSpan;
    
    graph.update('none');
}

function resetZoom(graph) { // Reset function
    if (!graph) return;
    // deleting min/max returns Chart.js to "Auto-Scale" mode
    delete graph.options.scales.y.min;
    delete graph.options.scales.y.max;
    graph.update('none');
}

// EMG zoom controls
function zoomEMGIn() { applyZoom(emgGraph, 1 / Y_ZOOM_FACTOR); }
function zoomEMGOut() { applyZoom(emgGraph, Y_ZOOM_FACTOR); }
function resetEMGZoom() { resetZoom(emgGraph); }

// Accelerometer zoom controls
function zoomAccelIn() { applyZoom(accelGraph, 1 / Y_ZOOM_FACTOR); }
function zoomAccelOut() { applyZoom(accelGraph, Y_ZOOM_FACTOR); }
function resetAccelZoom() { resetZoom(accelGraph); }
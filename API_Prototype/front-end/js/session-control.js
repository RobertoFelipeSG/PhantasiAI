// Session management (start/stop stream + render loops)

/* VARIABLES */
let stimSuccess = true; // stores whether stimulation from current trial was successful
let isStopping = false;
let renderLoopRunning = false; // for all visuals (graphs + instructions)


/* FUNCTIONS */ 
// Session-dependant Graph display + Real-time visuals
function startRenderLoop() {
    if (renderLoopRunning) return;

    renderLoopRunning = true;
    requestAnimationFrame(renderLoop);
}

function renderLoop() {
    if (!renderLoopRunning) return;
    
    if (!isGraphPaused && currentMode === 'developer') {
        batchUpdateGraph(emgGraph, pendingEmgData, 'emg');
        batchUpdateGraph(accelGraph, pendingAccelData, 'accel');
    }

    if (latestTimestamp > 0) {
        updateTimerVisuals(latestTimestamp)
    }

    requestAnimationFrame(renderLoop); // Continue looping
}

// Session data download button (used once stream has stopped)
function handleDataDownload(dataType) {
    let url;
    
    if (dataType === 'all') {
        if (!sessionDataFolder) return;
        url = `/download?data_path=${encodeURIComponent(sessionDataFolder)}`; // URL pointing to FastAPI /download route
    }

    else if (dataType === 'gt') {
        if (!sessionGroundTruth) return;
        url = `/download?data_path=${encodeURIComponent(sessionGroundTruth)}`; // URL pointing to FastAPI /download route
    }
    
    if (!url) {
        console.error("Download URL could not be generated.");
        return;
    }
    
    // Create an invisible anchor tag, click it, and remove it
    const a = document.createElement('a');
    a.href = url;
    a.download = ''; // Lets the backend dictate the filename
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Ground truth generator function (used after calibration)
function handleGenerateGT() {
    // Disable button and show "Generating..." message
    const gtBtn = document.getElementById('generateGTBtn');
    const gtStatusMsg = document.getElementById('gtStatusMessage');
    
    gtBtn.disabled = true;
    gtBtn.style.cursor = 'not-allowed';
    gtBtn.style.opacity = '0.5';
    gtStatusMsg.innerText = "Generating...";
    gtStatusMsg.style.color = '#e65100'; // Orange

    // Send generate ground truth command to backend
    const payload = {
        action: "generate_gt",
        mode: "individual"
    };

    // Send gt generation command to server
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
    }
}

// Helper function to track streaming state
function setStreamingState(active, statusMessage, color="333") {
    isStreaming = active;
    const button = document.getElementById('streamBtn');
    const input = document.getElementById('trialInput');
    const iters = document.getElementById('iters');
    const reps = document.getElementById('reps');
    const boardId = document.getElementById('boardID');
    const serialPort = document.getElementById('serialPort');
    const folderName = document.getElementById('folderName');
    const status = document.getElementById('streamingStatus');

    if (button) {
        button.innerText = active ? 'Stop' : 'Start';
        button.disabled = false;
    }
    
    // Lock input while active
    if (input) input.disabled = active;
    if (iters) iters.disabled = active;
    if (reps) reps.disabled = active;
    if (boardId) boardId.disabled = active;
    if (serialPort) serialPort.disabled = active;
    if (folderName) folderName.disabled = active;

    if (status) {
        status.innerText = statusMessage;
        status.style.color = color;
    }

    // Update heartbeat interval + timeout
    if (isStreaming) { 
        console.info("Setting pings to active mode")
        CLIENT_PING_INTERVAL = ACTIVE_PING_INTERVAL; 
        CLIENT_PING_TIMEOUT = ACTIVE_PING_TIMEOUT; 
    }
    else { 
        console.info("Setting pings to idle mode")
        CLIENT_PING_INTERVAL = IDLE_PING_INTERVAL; 
        CLIENT_PING_TIMEOUT = IDLE_PING_TIMEOUT; 
    }

    scheduleNextPing();
}

// Helper function to reset stream-dependent components back to default
function handleReset() {
    // Reset ALL Buffers and States
    CLIENT_PING_INTERVAL = IDLE_PING_INTERVAL;
    CLIENT_PING_TIMEOUT = IDLE_PING_TIMEOUT;
    pendingEmgData.length = 0;
    pendingAccelData.length = 0;
    lastInstructionPhase = "";
    nextEventTargetTime = null;
    totalTrials = null;
    trialsRemaining = null;
    lastPhase = "";
    isGraphPaused = false;
    resetEMGZoom();
    resetAccelZoom();
    document.getElementById('pauseGraphBtn').innerText = 'Pause Graphs';
    document.getElementById('streamBtn').disabled = false; // Enable stream + recording
    document.getElementById('trialInput').disabled = false;
    document.getElementById('iters').disabled = false;
    document.getElementById('reps').disabled = false;
    document.getElementById('boardID').disabled = false;
    document.getElementById('serialPort').disabled = false;
    
    // Show sidebar if hidden
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.remove('hidden');
    window.dispatchEvent(new Event('resize'));
    
    // Clear the synchronized marker countdown display
    const display = document.getElementById('markerCountdownDisplay');
    if (display) {
        display.innerText = "";
        display.style.color = 'blue';
    }

    // Clear trial countdown
    const trials = document.getElementById('trialCounterDisplay');
    if (trials) {
        trials.innerText = "Trials Remaining: --";
    }

    // Set instructions guide to default
    const instruction = document.getElementById('movementGuide');
    if (instruction) {
        instruction.innerText = "REST";
        instruction.style.color = "var(--color-rest)";
    }

    // Reset dots to default and ensure they're visible
    const dotContainer = document.getElementById('movementVisuals');
    if (dotContainer) {
        dotContainer.classList.add('resting');
        document.querySelectorAll('.dot-wrapper').forEach(d => d.classList.remove('active'));
    }

    // Clear graphs and event marker displays
    annotations.length = 0; 
    [emgGraph, accelGraph].forEach(graph => {
        if(graph && graph.options.plugins && graph.options.plugins.annotation) {
            graph.data.datasets.forEach(d => d.data = []);
            graph.options.plugins.annotation.annotations = [];
            graph.update('none');
        }
    });
}

// Helper function to reset frontend once backend session stops
function stopAction() {
    // Reset UI
    handleReset();
    setStreamingState(false, 'Idle', '#333');
    renderLoopRunning = false;
}

// Main streaming feature (start/stop via stream button)
async function handleStreamingClick() {
    // Case 1: Stop streaming + recording
    if (isStreaming) {
        await stopStream();
        return
    }

    // Case 2: Start streaming + recording
    // Get session data from controls inputs
    sessionDataFolder = null;
    sessionGroundTruth = null;

    const boardID = document.getElementById('boardID').value;
    let useSyntheticData = false;
    if (boardID === "Synthetic") { 
        useSyntheticData = true;
        SAMPLE_RATE = 250;
    } else if (boardID === "Ganglion") {
        useSyntheticData = false;
        SAMPLE_RATE = 200;
    }
    MAX_DATA_POINTS = SAMPLE_RATE * WINDOW_SIZE 

    const serialPort = document.getElementById('serialPort').value;
    let portToUse = 'serial_port_A';
    if (serialPort === "PortA") {
        portToUse = 'serial_port_A';
    } else if (serialPort === "PortB") {
        portToUse = 'serial_port_B';
    }
    
    const trialInput = document.getElementById('trialInput').value;
    let numTrials = parseFloat(trialInput);
    if (isNaN(numTrials) || numTrials < 1 || numTrials > 20) {
        numTrials = defaultNumTrials;
    }

    const itersInput = document.getElementById('iters').value;
    let numIters = parseFloat(itersInput);
    if (isNaN(numIters) || numIters < 1 || numIters > 50) {
        numIters = currentMode === 'calibrator' ? defaultCalbNumIters : defaultNumIters;
    }

    const repsInput = document.getElementById('reps').value;
    let numReps = parseFloat(repsInput);
    if (isNaN(numReps) || numReps < 1 || numReps > 20) {
        numReps = currentMode === 'calibrator' ? defaultCalbNumReps : defaultNumReps;
    }

    const folderName = document.getElementById('folderName').value;
    let folderToUse = (folderName === 'Auto' || folderName === '') ? null : folderName;

    // Lock controls before async loop starts and start main countdown
    document.getElementById('streamBtn').disabled = true;
    document.getElementById('trialInput').disabled = true;
    document.getElementById('iters').disabled = true;
    document.getElementById('reps').disabled = true;
    document.getElementById('boardID').disabled = true;
    document.getElementById('serialPort').disabled = true;
    document.getElementById('folderName').disabled = true;
    document.getElementById('downloadDataRow').style.display = 'none';
    document.getElementById('gtButtonRow').style.display = 'none';

    // Update UI
    document.getElementById("instructionsTab").click();
    
    const status = document.getElementById('streamingStatus');
    status.style.color = '#e65100'; // Orange for warning

    for (let i = 3; i > 0; i--) {
        status.innerText = `Starting in ${i}...`;
        
        // Wait 1 second
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    // Send number of trials per session and start stream + recording
    await startStream(numTrials, numIters, numReps, useSyntheticData, portToUse, folderToUse);
}

async function startStream(numTrials, numIters, numReps, useSyntheticData, portToUse, folderToUse)  {
    // If websocket closed/not auto created, attempt connection
    if (!ws || ws.readyState === WebSocket.CLOSED) {
        connectWebSocket();
    }
    
    // Wait briefly if connecting
    if (ws.readyState === WebSocket.CONNECTING) {
        let attempts = 0;
        while (ws.readyState === WebSocket.CONNECTING && attempts < 10) {
            await new Promise(r => setTimeout(r, 100));
            attempts++;
        }
    }
    
    // Final check for WS connection before streaming 
    if (ws.readyState !== WebSocket.OPEN) {
        alert("Connection is not ready. Please check the WS Status.");
        return;
    }

    // Prepare backend command payload
    const payload = {
        action: currentMode === 'calibrator' ? "start_calibration" : "start_stream",
        serial_port: portToUse,
        num_trials: numTrials,
        num_iters: numIters,
        num_reps: numReps,
        synthetic: useSyntheticData,
        folder_name: folderToUse
    };

    try { 
        // Reset global states and visual elements
        document.getElementById('streamBtn').disabled = true; // lock button briefly
        handleReset();
        startRenderLoop();

        ws.send(JSON.stringify(payload)); // Send start command to server
        setStreamingState(false, 'Connecting to board...', 'orange'); // update UI
    
    } catch (error) { 
        console.error(error);
        document.getElementById('streamBtn').disabled = false;
        setStreamingState(false, 'Error sending start command', 'red');
        document.getElementById("sessionStatusMessage").innerText = "Error sending start command."
        document.getElementById("sessionStatusTab").click();
        //alert("Error sending start command: " + error.message);
    }
}

async function stopStream() { 
    // Show sidebar if hidden
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.remove('hidden');
    window.dispatchEvent(new Event('resize'));
    
    try {
        // Lock button briefly
        document.getElementById('streamBtn').disabled = true; 
        document.getElementById('streamingStatus').innerText = "Stopping...";

        // Set ping interval/timeout to idle mode
        console.info("Setting pings to idle mode")
        CLIENT_PING_INTERVAL = IDLE_PING_INTERVAL; 
        CLIENT_PING_TIMEOUT = IDLE_PING_TIMEOUT; 
        scheduleNextPing();
        
        // Send stop command to server
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: "stop_stream" }));
        }
    
    } catch (e) {
        console.error(e);
        setStreamingState(true, 'Error sending stop command.', 'red'); // Revert if failed
        document.getElementById("sessionStatusMessage").innerText = "Error sending stop command."
        document.getElementById("sessionStatusTab").click();
        document.getElementById('streamBtn').disabled = false;
    }
}
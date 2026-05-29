// Manage connections to FastAPI backend 

/* VARIABLES */
let ws = null;
let serverRestarting = false;
let heartbeat = null;
let pingTimeout = null;
let pongReceived = true;

/* FUNCTIONS */

// Heartbeat logic (connection monitoring)
function scheduleNextPing() {
    if (heartbeat) clearTimeout(heartbeat);

    heartbeat = setTimeout(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            pongReceived = false;
            ws.send(JSON.stringify({ action: "client_ping" }));
    
            if (pingTimeout) clearTimeout(pingTimeout); // clear timeout before sending new one
            
            pingTimeout = setTimeout(() => {
                console.warn("Server pong timeout! Closing connection.");
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ action: "hb_timeout" }));
                }
                pongReceived = false;
                if (ws) ws.close();

                // Show sidebar and display warning
                const sidebar = document.querySelector('.sidebar');
                sidebar.classList.remove('hidden');
                window.dispatchEvent(new Event('resize'));
                const status = document.getElementById('websocketStatus');
                status.style.color = "var(--color-lower)";
                status.innerText = "WS Status: Connection Error";
                
            
            }, CLIENT_PING_TIMEOUT);
        }
        
    }, CLIENT_PING_INTERVAL);
}

// Server connection (via WebSocket)
function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    // Connect to server via WebSocket + update server connection status
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${protocol}://${window.location.host}/ws`;
    
    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            const status = document.getElementById('websocketStatus');
            status.innerText = "WS Status: Connected";
            status.style.color = "var(--color-raise)";
        
            CLIENT_PING_INTERVAL = IDLE_PING_INTERVAL;
            CLIENT_PING_TIMEOUT = IDLE_PING_TIMEOUT;
            scheduleNextPing();
        }
        
        // DATA COMMUNICATION 
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Heartbeat ping/pong logic: acknowledge server's pong response
            if (data.type === "server_pong") {
                console.info("Server pong response received");
                pongReceived = true;
                if (pingTimeout) clearTimeout(pingTimeout);
                scheduleNextPing();
                return;
            }

            // Heartbeat ping/pong logic: respond to server's ping
            if (data.type === "server_ping") {
                console.info("Responding to server's ping");
                ws.send(JSON.stringify({action: "client_pong"}));
                return;
            }
            
            // Update graph buffers with most recent EMG + Accel data
            if (data.type === 'emg_data') {
                if (data.timestamp && data.timestamp.length > 0) {
                    latestTimestamp = data.timestamp[data.timestamp.length - 1];
                }

                if (currentMode === 'developer') {
                    data.value.forEach((val, i) => {
                        pendingEmgData.push({ x: data.timestamp[i], y: val });
                    });
                    if (pendingEmgData.length > MAX_DATA_POINTS * 2) {
                        pendingEmgData = pendingEmgData.slice(-MAX_DATA_POINTS);
                    }
                }
            }

            else if (ACCEL_ENABLED && data.type === 'accel_data') {
                if (currentMode === 'developer') {
                    data.value[0].forEach((_, i) => {
                        pendingAccelData.push({ x: data.timestamp[i],
                                                y_x: data.value[0][i],
                                                y_y: data.value[1][i],
                                                y_z: data.value[2][i] });
                    });
                }
            }
            
            // Add most recent event markers (vertical lines)
            else if (data.type === 'event_times') {
                if (currentMode === 'developer') {
                    data.event_timestamps.forEach(event_time => {
                    updateEventDisplay(event_time);
                    });
                }
            }

            // Add most recent trial markers (vertical lines)
            else if (data.type === 'trial_times') {
                if (currentMode === 'developer') {
                    data.trial_timestamps.forEach(trial_time => {
                    updateTrialDisplay(trial_time);
                    });
                }
            }

            // Update marker timer countdown display 
            else if (data.type === 'event_target_time') {
                nextEventTargetTime = data.event_target_time;
            }

            // Update trial timer countdown display and exercise instructions
            else if (data.type === 'trial_target_time') {
                nextTrialTargetTime = data.trial_target_time;
            }
            
            // update trial countdown display
            else if (data.type === 'stim_failed') {
                console.warn("Stimulation failure for trial");
                stimSuccess = false;
            }
            
            else if (data.type === 'trial_completion') {
                if (stimSuccess) { updateTrialCountdown(data.total_trials); }
                stimSuccess = true; // reset for next trial
            }

            // Update ready timer countdown display for exercise instructions
            else if (data.type === 'ready_target_time') {
                nextReadyTargetTime = data.ready_target_time;
            }
            

            // Update chat-terminal
            else if (data.type === 'server_log') {
                addLog(data.message);
            }

            // Update break UI
            else if (data.type === 'break_status') {
                const animationElement = document.getElementById('movementAnimation');

                if (data.status === 'started') {
                    console.log("Break started");
                    inBreak = true;
                    animationElement.style.display = "none";
                }
                else if (data.status === 'ended') {
                    console.log("Break ended");
                    inBreak = false;
                    animationElement.style.display = "block";
                }
            }
            else if (data.type === 'break_end_time') {
                breakEndTime = data.break_end_time;
            }

            // Handle command responses (success/error feedback)
            else if (data.status) {
                if (data.status === 'error') {
                    console.error("Server Error:", data.message);
                    
                    // show side panel if hidden
                    const sidebar = document.querySelector('.sidebar');
                    sidebar.classList.remove('hidden');
                    window.dispatchEvent(new Event('resize'));

                    // reset UI if failure
                    handleReset();
                    renderLoopRunning = false;
                    
                    if (data.message === "Failed to start EMG stream") {
                        setStreamingState(false, 'Failed to start stream', 'red');
                    }

                    if (data.message === "Unexpected error in ws loop") {
                        setStreamingState(false, 'Unexpected error', 'red');
                    }

                    if (data.type === "board_already_in_use") {
                        setStreamingState(false, 'Board already in use. Restarting server, please wait...', 'red');
                        serverRestarting = true;
                    }

                    if (data.type === "port_already_in_use") {
                        setStreamingState(false, 'Port already in use. Restarting server, please wait...', 'red');
                        serverRestarting = true;
                    }
                    
                    if (data.type === "general_brainflow_error") {
                        setStreamingState(false, data.message, 'red');
                    }
                    
                    if (data.type === "data_timeout") {
                        setStreamingState(false, 'Stream data timeout error, check board connection', 'red');
                    }

                    if (data.type === "general_EMG_error") {
                        setStreamingState(false, data.message, 'red');
                    }

                    // error handling if ground truth generation failed
                    if (data.type === "GT generation failed") {
                        const gtStatusMsg = document.getElementById('gtStatusMessage');
                        const gtBtn = document.getElementById('generateGTBtn');
                        
                        gtStatusMsg.innerText = "Error! " + data.message;
                        gtStatusMsg.style.color = 'var(--color-hold)'; // Red
                        
                        // Re-enable button so user can try again
                        gtBtn.disabled = false;
                        gtBtn.style.cursor = 'pointer';
                        gtBtn.style.opacity = '1';
                    }
                } 
                
                else if (data.status === 'success') {
                    console.log("Command executed successfully:", data.message);
                    
                    if (data.message === "EMG streaming started") {
                        setStreamingState(true, 'Active', 'green'); // update streaming state
                        initializeTrialCountdown(); // start trial countdown

                        // hide side panel if in experimenter mode
                        if (currentMode === 'experimenter' || currentMode === 'calibrator') {
                            const sidebar = document.querySelector('.sidebar');
                            sidebar.classList.add('hidden');
                            window.dispatchEvent(new Event('resize'));
                        }

                        // hide ground truth generation button (for calibrator mode)
                        document.getElementById('gtButtonRow').style.display = 'none';
                        document.getElementById('downloadGTBtn').style.display = 'none';
                        document.getElementById('gtStatusMessage').innerText = "";
                    }
                    
                    else if (data.message === "EMG streaming stopped") {
                        // Reset UI once backend has successfully stopped the stream
                        stopAction();

                        if (data.folder) {
                            console.log("Received data folder");
                            sessionDataFolder = data.folder; 
                            document.getElementById('downloadDataRow').style.display = "block";
                        }

                        // Show final completion message if all trials were complete
                        if (allTrialsCompleted) {
                            document.getElementById("sessionStatusMessage").innerText = "Session complete! Please remain seated and await further instructions."
                            document.getElementById("sessionStatusTab").click();
                        }
                    }

                    else if (data.message === "Calibration complete") {
                        // Enable the Generate Ground Truth button
                        if (currentMode === 'calibrator') {
                            document.getElementById('gtButtonRow').style.display = 'block';
                        }
                    }

                    else if (data.message === "Ground truth generated") {
                        // Show success message if ground truth generated
                        const gtStatusMsg = document.getElementById('gtStatusMessage');
                        gtStatusMsg.innerText = "Success! Please proceed to experimenter mode";
                        gtStatusMsg.style.color = 'var(--color-raise)'; // Green

                        const gtBtn = document.getElementById('generateGTBtn');
                        gtBtn.disabled = false;
                        gtBtn.style.cursor = 'pointer';
                        gtBtn.style.opacity = '1';

                        if (data.file) {
                            console.log("Received ground truth file");
                            sessionGroundTruth = data.file; 
                            document.getElementById('downloadGTBtn').style.display = 'block';
                        }
                    }
                }
            }
        }

        ws.onclose = async (event) => {
            console.warn("WebSocket Disconnected:", event);
            if (heartbeat) clearTimeout(heartbeat);
            
            // Show sidebar if hidden
            const sidebar = document.querySelector('.sidebar');
            sidebar.classList.remove('hidden');
            window.dispatchEvent(new Event('resize'));
            
            // Reset UI and update streaming state
            stopAction();
            const status = document.getElementById('websocketStatus');
            status.style.color = "var(--color-lower)";
            if (serverRestarting) { status.innerText = "WS Status: Server Restarting, please refresh page"; }
            else if (!pongReceived) { status.innerText = "WS Status: Connection Error"; }
            else { status.innerText = "WS Status: Server Shutdown"; }
            
            ws = null;
        };

        ws.onerror = async (error) => {
            console.error("WebSocket Error:", error);
            if (heartbeat) clearTimeout(heartbeat);

            // Show sidebar if hidden
            const sidebar = document.querySelector('.sidebar');
            sidebar.classList.remove('hidden');
            window.dispatchEvent(new Event('resize'));
            
            // Reset UI and update streaming state
            stopAction();
            const status = document.getElementById('websocketStatus');
            status.innerText = "WS Status: Connection Error";
            status.style.color = "var(--color-hold)";
            
            ws = null;
        }
    } catch (e) {
        console.error("Failed to create WebSocket:", e);
    }
}
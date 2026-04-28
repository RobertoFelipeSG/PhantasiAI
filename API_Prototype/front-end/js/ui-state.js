// State management for UI features

/* VARIABLES */
// App setup
let currentMode = 'developer';

// Side panel state
let terminalEnabled = true;

// Session management 
let totalTrials = null;
let trialsRemaining = null;
let allTrialsCompleted = false;
let isStreaming = false;
let sessionDataFolder = null;


/* FUNCTIONS */
// Mode selection
function selectMode(mode) {
    currentMode = mode;
    const modeSelect = document.getElementById('modeSelect');
    const appWrapper = document.getElementById('appWrapper');
    
    // Hide the mode screen, show the app using fade transition
    modeSelect.classList.add('fade-out');
    setTimeout(() => {
        modeSelect.style.display = 'none';
        appWrapper.style.display = 'flex';
        appWrapper.style.opacity = '0';

        if (mode === 'experimenter') {
            // Hide the tab bar and Graphs entirely
            document.querySelector('.tab').style.display = 'none';
            document.getElementById('Graphs').style.display = 'none';

            // Set sidebar width and hide the terminal
            document.querySelector('.sidebar').style.width = '15%';
            document.querySelector('.sidebar').style.minWidth = 'unset';
            document.querySelector('.terminal-wrapper').style.display = 'none';
            terminalEnabled = false;

            // Hide developer-only controls (keep: WS Status, Folder Name, Streaming Status)
            const hideIds = ['boardIDRow', 'serialPortRow', 'trialInputRow', 'itersRow', 'repsRow', 'markerRow'];
            hideIds.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = 'none';
            });
        
        } else if (mode === 'calibrator') {
            // Hide the tab bar and Graphs entirely 
            document.querySelector('.tab').style.display = 'none';
            document.getElementById('Graphs').style.display = 'none';

            // Set sidebar width and hide the terminal (same as experimenter)
            document.querySelector('.sidebar').style.width = '15%';
            document.querySelector('.sidebar').style.minWidth = 'unset';
            document.querySelector('.terminal-wrapper').style.display = 'none';
            terminalEnabled = false;

            // Hide specific controls (keep: WS Status, Board ID, Repetitions, Folder Name, Streaming Status)
            const hideIds = ['serialPortRow', 'trialInputRow', 'itersRow', 'markerRow'];
            hideIds.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = 'none';
            });

            // Set default iterations and repetitions for calibrator mode
            document.getElementById('iters').value = defaultCalbNumIters;
            const repsInput = document.getElementById('reps');
            repsInput.value = defaultCalbNumReps
            repsInput.onfocus = function() { if (this.value == defaultCalbNumReps) { this.value = '';}};
            repsInput.onblur = function() {if (this.value === '') {this.value = defaultCalbNumReps;}};
        
        } else { // developer mode
            initGraphs();
        }

        // Start transition once layout changes are applied 
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                appWrapper.style.opacity = '1';
            });
        });

        // Common init for both modes
        document.getElementById('downloadDataRow').style.display = 'none'; // hide data download button until stream stops
        document.getElementById('gtButtonRow').style.display = 'none'; // hide GT generation button (only appears once calibration done)
        document.getElementById('SessionStatus').style.display = 'flex'; // Show SessionStatus tab directly (no tab click needed)
        setStreamingState(false, 'Idle');
        connectWebSocket();

    }, 300); // transition duration    
}

// Tab control
function openTab(evt, tabName) {
    // Hide all elements of previously active tab
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab
    const activeTab = document.getElementById(tabName);
    if (tabName === "Instructions" || tabName === "SessionStatus") {
        activeTab.style.display = "flex"; // This enables vertical centering
    } else {
        activeTab.style.display = "block";
    }
    evt.currentTarget.className += " active";
}

// Sidebar control
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');

    if (sidebar.classList.contains('hidden')) {
        sidebar.classList.remove('hidden');
    } else {
        sidebar.classList.add('hidden');
    }

    window.dispatchEvent(new Event('resize'));
}

// Chat window
function toggleTerminal() {
    terminalEnabled = !terminalEnabled;
    const btn      = document.getElementById('terminalToggleBtn');
    const terminal = document.getElementById('chatTerminal');

    if (terminalEnabled) {
        terminal.style.display = 'block';
        btn.textContent = 'ON';
        btn.classList.remove('off');
        addLog('[System] Terminal resumed.');
    } else {
        terminal.style.display = 'none';
        btn.textContent = 'OFF';
        btn.classList.add('off');
    }
}

function addLog(message) {
    if (!terminalEnabled) return; // zero DOM work when off

    const terminal = document.getElementById('chatTerminal');

    // Cap entries to 200
    const entries = terminal.querySelectorAll('.log-entry');
    if (entries.length > 200) {
        for (let i = 0; i < 50; i++) entries[i].remove();
    }

    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.textContent = message;
    terminal.appendChild(entry);
    terminal.scrollTop = terminal.scrollHeight;
}

// Trial countdown logic
function initializeTrialCountdown() {
    const itersInput = document.getElementById('iters');
    const repsInput = document.getElementById('reps');
    const itersValue = parseInt(itersInput.value) || (currentMode === 'calibrator' ? defaultCalbNumIters : defaultNumIters);
    const repsValue = parseInt(repsInput.value) || (currentMode === 'calibrator' ? defaultCalbNumReps : defaultNumReps);
    const extraTrial = currentMode === 'calibrator' ? 0 : 1;
    const totalTrialsInput = (itersValue > 0 && repsValue > 0) ? (itersValue * repsValue) + extraTrial : (currentMode === 'calibrator' ? defaultCalbTotalTrials : defaultTotalTrials);
    const display = document.getElementById('trialCounterDisplay');

    if (!display) return;

    totalTrials = totalTrialsInput // update the totalTrials global variable ONCE (at initialization)
    trialsRemaining = totalTrials;

    // Update trial display
    const newText = `Trials Remaining: ${Math.max(0, trialsRemaining)}`;
    if (display.innerText !== newText) {
        display.innerText = newText;
    }
}

function updateTrialCountdown(trialsComplete) {
    const display = document.getElementById('trialCounterDisplay');
    if (!display) return;
    
    // subtract total trials if trial complete AND stimulation was successful 
    trialsRemaining = totalTrials - trialsComplete;
    
    // Update trial display
    const newText = `Trials Remaining: ${Math.max(0, trialsRemaining)}`;
    if (display.innerText !== newText) {
        display.innerText = newText;
    }

    // Check if all trials are complete
    if (trialsRemaining <= 0) {
        allTrialsCompleted = true;
        document.getElementById("sessionStatusMessage").innerText = "All trials complete! Stopping stream and saving data..."
        document.getElementById("sessionStatusTab").click();
    }
}
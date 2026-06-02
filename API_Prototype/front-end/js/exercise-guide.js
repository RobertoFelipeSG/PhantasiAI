// Real-time instructions management

/* VARIABLES */
let latestTimestamp = null;
let nextEventTargetTime = null;
let nextTrialTargetTime = null;
let nextReadyTargetTime = null;
let lastPhase = ""; // stores only the instruction
let lastInstructionPhase = ""; // stores instruction + active dot
let inBreak = false;
let breakEndTime = null;
let targetBiofeedbackPercentage = 0;
let maxEmgExpected = null; // defined in session-control.js

const countdownElements = {
    EVENT: document.getElementById('markerCountdownDisplay'),
};

const movementElements = {
    instruction: document.getElementById('movementGuide'),
    video: document.getElementById('instructionVideo')
    
    // CURRENTLY NOT IN USE
    /*dotContainer: document.getElementById('movementVisuals'),
    dots: {
        left:   document.getElementById('dot-left'),
        middle: document.getElementById('dot-middle'),
        right:  document.getElementById('dot-right')
    }*/
};


/* FUNCTIONS */
// Main function (updates all animation components except for the biofeedback bar)
function updateTimerVisuals(currentTime) {

    // when NOT in break: calculate time until next trial and update visuals
    if ((nextReadyTargetTime !== null) && (!inBreak)) {
        let readyTimeRemaining = Math.max(0, nextReadyTargetTime - currentTime);
        
        updateMovementGuide(readyTimeRemaining);
        if (currentMode === 'developer') updateMarkerCountdown(readyTimeRemaining);
    }

    // if in break: display break countdown instead of instructions
    if ((breakEndTime !== null) && (inBreak)) {
        let breakTimeRemaining = Math.max(0, breakEndTime - currentTime);
        updateBreakUI(breakTimeRemaining);
    }
}

// Trial timer countdown (in controls bar)
function updateMarkerCountdown(timeUntilReady) {
    const display = countdownElements['EVENT'];
    if (!display) return;
    
    const formattedTime = timeUntilReady.toFixed(1) + "s";
    const targetText = `NEXT STIMULATION: ${formattedTime}`;
        
    if (display.innerText !== targetText) { // DOM optimization: only update if text has changed
        display.innerText = targetText;

        // REST: 0-2.5; READY: 2.5-2.5/4.0; GO: 2.5/4.0-4.5/6.0
        if (timeUntilReady <= REST_PHASE) { // REST PHASE = no dorsiflexion/electrical stim
            display.style.color = "var(--color-rest)"; 
        }
        else if (timeUntilReady <= GO_PHASE + REST_PHASE) { // GO PHASE = dorsiflexion + electrical stim
            display.style.color = "var(--color-raise)";
        } 
        else {
            display.style.color = "var(--color-lower)"; // READY PHASE = no dorsiflexion, electrical stim
        }
    }
}

// Basic go-rest display
function updateMovementGuide(timeUntilReady) {
    const { instruction, video } = movementElements; // NOT IN USE: dotsContainer, dots

    if (!instruction) return;
    
    let phase = "";
    let color = "";
    // let activeDot = "";

    if (timeUntilReady <= REST_PHASE) { // REST PHASE
            phase = "REST";
            color = "var(--color-rest)"; // Blue
        }
    else if (timeUntilReady <= GO_PHASE + REST_PHASE) { // GO PHASE
        phase = "GO!";
        color = "var(--color-raise)"; // Green
    } 
    else { // READY PHASE
        phase = "READY";
        color = "var(--color-lower)"; // Orange
    }

    // DOM updates (optimized): text + colors + dots
    if (phase !== lastInstructionPhase) { // NOT IN USE: + activeDot
        instruction.innerText = phase;
        instruction.style.color = color;

        // NOT IN USE
        // Play dorsiflexion animation exactly when entering the RAISE (GO) phase
        /*if (phase === "GO!" && video) {
            // Play video (with safety in case browser blocks autoplay)
            video.play().catch(err => console.warn("Video playback prevented by browser:", err));
        
        } else if (video) {
            // Freeze video and rewind to start
            video.pause();
            video.currentTime = 0;
        }*/
        
        // NOT IN USE
        /*if (phase === "REST") {
            dotContainer.classList.add('resting');
        } else {
            dotContainer.classList.remove('resting');
        }
        
        Object.values(dots).forEach(d => d.classList.remove('active'));
        if (dots[activeDot]) {
            dots[activeDot].classList.add('active');
        }*/
        
        lastInstructionPhase = phase; // NOT IN USE: + activeDot
    }
    
}

function updateBreakUI(breakTimeRemaining) {
    const { instruction, _ } = movementElements; 

    if (!instruction) return;
    
    const formattedTime = breakTimeRemaining.toFixed(0) + "s";
    const targetText = `BREAK REMAINING: ${formattedTime}`;
        
    if (instruction.innerText !== targetText) { // DOM optimization: only update if text has changed
        instruction.innerText = targetText;
        instruction.style.color = "var(--color-rest)";
    }
}

function processBiofeedbackData(ganglionData) {
    if (!ganglionData || ganglionData.length === 0) return;

    let sum = 0;
    ganglionData.forEach(val => {
        sum += Math.abs(val); 
    });
    const averageAmplitude = sum / ganglionData.length;

    let percentage = (averageAmplitude / maxEmgExpected) * 100;
    
    // Store the result in the global state variable
    targetBiofeedbackPercentage = Math.max(0, Math.min(100, percentage));
}

function updateBiofeedbackBar() {
    const feedbackBar = document.getElementById('emgBar');
    if (!feedbackBar) return;

    // Update height based on the latest state
    feedbackBar.style.height = `${targetBiofeedbackPercentage}%`;

    // Update colors
    if (targetBiofeedbackPercentage > 75) {
        feedbackBar.style.backgroundColor = '#4CAF50'; 
    } else if (targetBiofeedbackPercentage > 50) {
        feedbackBar.style.backgroundColor = '#ff9800'; 
    } else {
        feedbackBar.style.backgroundColor = '#f44336';
    }
}

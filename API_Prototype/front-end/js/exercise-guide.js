// Real-time instructions management

/* VARIABLES */
let latestTimestamp = null;
let nextEventTargetTime = null;
let nextTrialTargetTime = null;
let lastPhase = ""; // stores only the instruction
let lastInstructionPhase = ""; // stores instruction + active dot

const countdownElements = {
    EVENT: document.getElementById('markerCountdownDisplay'),
    TRIAL: document.getElementById('trialTimeCountdownDisplay')
};

const movementElements = {
    instruction: document.getElementById('movementGuide'),
    dotContainer: document.getElementById('movementVisuals'),
    dots: {
        left:   document.getElementById('dot-left'),
        middle: document.getElementById('dot-middle'),
        right:  document.getElementById('dot-right')
    }
};


/* FUNCTIONS */
// Main function (updates all animation components)
function updateTimerVisuals(currentTime) {
    // NO LONGER NEEDED: calculate time until next event marker and update visuals
    /*if ((nextEventTargetTime !== null) && (currentMode === 'developer')) {
        let eventTimeRemaining = Math.max(0, nextEventTargetTime - currentTime);
        
        updateMarkerCountdown('EVENT', eventTimeRemaining);
    }*/

    // calculate time until next trial and update visuals
    if (nextTrialTargetTime !== null) {
        let trialTimeRemaining = Math.max(0, nextTrialTargetTime - currentTime);
        
        updateMovementGuide(trialTimeRemaining);
        if (currentMode === 'developer') updateMarkerCountdown('TRIAL', trialTimeRemaining);
    }
}

// Event marker / Trial timer countdown (in controls bar)
function updateMarkerCountdown(type, timeRemaining) {
    const display = countdownElements[type];
    if (!display) return;
    
    const formattedTime = timeRemaining.toFixed(1) + "s";
    const targetText = `NEXT ${type}: ${formattedTime}`;
        
    if (display.innerText !== targetText) { // DOM optimization: only update if text has changed
        display.innerText = targetText;

        if (type === 'TRIAL') { // 1-3 = go; 3-6 = rest
            if (timeRemaining > PHASE_CHANGE) { 
                display.style.color = "var(--color-rest)"; 
            } 
            else {// Detect event marker (movement begins)
                display.style.color = "var(--color-raise)";
            }
        }
        if (type === 'EVENT') { // 1-3 = rest; 3-6 = go
            if (timeRemaining <= PHASE_CHANGE) { 
                display.style.color = "var(--color-rest)"; 
            } 
            else {// Detect event marker (movement begins)
                display.style.color = "var(--color-raise)";
            }
        }
    }
}

// Basic go-rest display
function updateMovementGuide(timeRemaining) {
    const { instruction, dotContainer, dots } = movementElements;

    if (!instruction) return;
    
    let phase = "";
    let color = "";
    let activeDot = "";

    if (timeRemaining <= PHASE_CHANGE) { // GO PHASE
        const timeInGO = PHASE_CHANGE - timeRemaining;
        if (timeInGO <= 1.0) {
            phase = "RAISE";
            color = "var(--color-raise)"; // Green
            activeDot = "left";
        } else if (timeInGO <= 2.0) {
            phase = "HOLD";
            color = "var(--color-hold)"; // Red
            activeDot = "middle";
        } else {
            phase = "LOWER";
            color = "var(--color-lower)"; // Orange
            activeDot = "right";
        }

    } 
    else {  // REST PHASE
        phase = "REST";
        color = "var(--color-rest)"; // Blue
    }

    // DOM updates (optimized): text + colors + dots
    if (phase + activeDot !== lastInstructionPhase) {
        instruction.innerText = phase;
        instruction.style.color = color;
        
        if (phase === "REST") {
            dotContainer.classList.add('resting');
        } else {
            dotContainer.classList.remove('resting');
        }
        
        Object.values(dots).forEach(d => d.classList.remove('active'));
        if (dots[activeDot]) {
            dots[activeDot].classList.add('active');
        }
        
        lastInstructionPhase = phase + activeDot;
    }
    
}
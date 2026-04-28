// Real-time instructions management

/* VARIABLES */
let latestTimestamp = null;
let nextEventTargetTime = null;
let lastPhase = ""; // stores only the instruction
let lastInstructionPhase = ""; // stores instruction + active dot


/* FUNCTIONS */
// Main function (updates all animation components)
function updateTimerVisuals(currentTime) {
    if (nextEventTargetTime === null) return;

    let timeRemaining = Math.max(0, nextEventTargetTime - currentTime);
    
    updateMarkerCountdown(timeRemaining);
    updateMovementGuide(timeRemaining);
}

// Event marker countdown (in controls bar)
function updateMarkerCountdown(timeRemaining) {
    const display = document.getElementById('markerCountdownDisplay');
    
    const formattedTime = timeRemaining.toFixed(1) + "s";
    const targetText = `NEXT MARKER: ${formattedTime}`;
        
    if (display.innerText !== targetText) {
        if (timeRemaining < PHASE_CHANGE) { // Detect rest period
            display.style.color = "var(--color-rest)"; 
        } 
        else {// Detect event marker (movement begins)
            display.style.color = "var(--color-raise)";
        }

        display.innerText = targetText;
    }
}

// Basic go-rest display
function updateMovementGuide(timeRemaining) {
    const instruction = document.getElementById('movementGuide');
    const dotContainer = document.getElementById('movementVisuals');
    const dots = {
        left:   document.getElementById('dot-left'),
        middle: document.getElementById('dot-middle'),
        right:  document.getElementById('dot-right')
    };

    if (!instruction) return;
    
    let phase = "";
    let color = "";
    let activeDot = "";

    if (timeRemaining <= PHASE_CHANGE) { // REST PHASE
        phase = "REST";
        color = "var(--color-rest)"; // Blue
        const timeElapsed = PHASE_CHANGE - timeRemaining;
        if (timeElapsed < 1.0) activeDot = "left";
        else if (timeElapsed < 2.0) activeDot = "middle";
        else activeDot = "right";
    } 
    else {  // GO PHASES
        const timeInGO = timeRemaining - PHASE_CHANGE;

        if (timeInGO > ((PHASE_CHANGE / 3.0) * 2.0)) {
            phase = "RAISE";
            color = "var(--color-raise)"; // Green
            activeDot = "left";
        } else if (timeInGO > (PHASE_CHANGE / 3.0)) {
            phase = "HOLD";
            color = "var(--color-hold)"; // Red
            activeDot = "middle";
        } else {
            phase = "LOWER";
            color = "var(--color-lower)"; // Orange
            activeDot = "right";
        }
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
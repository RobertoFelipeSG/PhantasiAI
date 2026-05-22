// Contains global variables for connection tracking, session configuration, and graphs

// Heartbeat connection variables
const IDLE_PING_INTERVAL = 10000;
const ACTIVE_PING_INTERVAL = 3000;
let CLIENT_PING_INTERVAL = IDLE_PING_INTERVAL; // intial heartbeat until board connects
const IDLE_PING_TIMEOUT = 30000;
const ACTIVE_PING_TIMEOUT = 6000;
let CLIENT_PING_TIMEOUT = IDLE_PING_TIMEOUT; // initial timeout until board connects

// Session constants
const defaultTotalTrials = 401;
const defaultCalbTotalTrials = 75;
const defaultNumTrials = 1;
const defaultNumIters = 40;
const defaultCalbNumIters = 25;
const defaultNumReps = 10;
const defaultCalbNumReps = 3;
const PHASE_CHANGE = 3.0; // halfway point of a trial (seconds)

// Graph configuration
const ACCEL_ENABLED = false; 
const Y_ZOOM_FACTOR = 1.5; // controls how aggressive the Y zoom is
const WINDOW_SIZE = PHASE_CHANGE * 2; // Controls x-axis window size (seconds)
let SAMPLE_RATE = 200; // Default sample rate (for Ganglion)
let MAX_DATA_POINTS = WINDOW_SIZE * SAMPLE_RATE; // Limit data for scrolling graph
// Application entry point

document.addEventListener('DOMContentLoaded', () => {
    // main app initializes only after mode is selected

    const devBtn = document.getElementById('btnDev');
    const expBtn = document.getElementById('btnExp');
    const calibBtn = document.getElementById('btnCalib');

    if (devBtn) { devBtn.addEventListener('click', () => selectMode('developer')); }
    if (expBtn) { expBtn.addEventListener('click', () => selectMode('experimenter')); }
    if (calibBtn) { calibBtn.addEventListener('click', () => selectMode('calibrator')); }
})
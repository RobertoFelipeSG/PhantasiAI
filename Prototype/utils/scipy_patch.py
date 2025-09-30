# utils.py
import scipy.signal
#replace deprecated 'hanning' window with 'hann' in scipy.signal.welch

def patch_scipy_welch(*args, **kwargs):
    """
    A wrapper for scipy.signal.welch that replaces 'hanning' with 'hann' if needed.
    Use this to overwrite scipy.signal.welch globally.
    """
    if kwargs.get('window') == 'hanning':
        kwargs['window'] = 'hann'

    return scipy.signal._spectral_py.welch(*args, **kwargs)

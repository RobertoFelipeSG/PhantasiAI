# utils.py
import scipy.signal
#replace deprecated 'hanning' window with 'hann' in scipy.signal.welch
def patch_scipy_welch():
    _original_welch = scipy.signal.welch
    def patched(*args, **kwargs):
        if kwargs.get('window') == 'hanning':
            kwargs['window'] = 'hann'
        return _original_welch(*args, **kwargs)
    scipy.signal.welch = patched

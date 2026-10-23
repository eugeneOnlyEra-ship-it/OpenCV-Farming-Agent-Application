"""One-stop image collection for the application side.

Every place that collects a crop/pod image (the model-input loader in
perception.py, the follow-up frames in frame_source.py, the app's image
viewer) calls these two functions. They use the C++ pod_capture module when
it is built, and plain cv2 when it is not, so nothing breaks either way.
"""
import cv2

try:
    import pod_capture as _cpp
except ImportError:
    _cpp = None


def load_image(path):
    """Read an image from disk as a BGR ndarray, or None if unreadable (like cv2.imread)."""
    if _cpp is not None:
        return _cpp.load_image(str(path))
    return cv2.imread(str(path))


def save_image(bgr, path, jpeg_quality=92):
    """Write a BGR ndarray to disk. Returns True on success."""
    if _cpp is not None:
        return _cpp.save_image(bgr, str(path), int(jpeg_quality))
    return cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])


def using_cpp():
    return _cpp is not None

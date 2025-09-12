# Ensure vendored ultralytics is always used
import os
import sys

# Absolute path to the vendored ultralytics package
_vendored_ultralytics = os.path.abspath(os.path.join(os.path.dirname(__file__), "ultralytics"))

# Insert at front of sys.path if not already there
if sys.path[0] != _vendored_ultralytics:
    if _vendored_ultralytics in sys.path:
        sys.path.remove(_vendored_ultralytics)
    sys.path.insert(0, _vendored_ultralytics)

# Warn if another ultralytics is found elsewhere on sys.path
import importlib.util
spec = importlib.util.find_spec("ultralytics")
origin = getattr(spec, "origin", None)
if spec is not None and origin is not None and not os.path.abspath(origin).startswith(_vendored_ultralytics):
    import warnings
    warnings.warn(
        f"A non-vendored 'ultralytics' was found at {origin}. The vendored version at {_vendored_ultralytics} will be used. "
        "To avoid conflicts, ensure no other 'ultralytics' is installed in your environment."
    )

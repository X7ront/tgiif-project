"""Platform helpers for the optional OpenKE sampling/evaluation library."""

import ctypes
import os


def load_base_library():
    """Load the native backend, preferring the Windows DLL when present."""
    release = os.path.join(os.path.dirname(__file__), "release")
    names = ("Base.dll", "Base.so", "Base.dylib")
    errors = []
    for name in names:
        path = os.path.join(release, name)
        if not os.path.isfile(path):
            continue
        try:
            return ctypes.cdll.LoadLibrary(path)
        except OSError as exc:
            errors.append("{}: {}".format(path, exc))
    detail = "; ".join(errors) if errors else "no native library found"
    raise RuntimeError(
        "OpenKE native backend is unavailable ({}). On Windows run "
        "openke\\build_windows.ps1, or use a prebuilt Base.dll.".format(detail)
    )

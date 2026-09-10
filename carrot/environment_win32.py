"""Read-only Windows environment probe (Slice 6, Phase 5).

The ONLY module in the repo permitted to import ``ctypes`` (enforced
by ``test_no_os_write_primitives``). Never imported by production code;
constructed explicitly by the caller if ever used. Every OS call is
read-only, guarded, and total: any failure yields an empty observation
rather than raising. Uses ``ctypes`` scalar types directly (never
``wintypes``) so this module imports cleanly on every platform.
"""
from __future__ import annotations

import ctypes
import os
import sys

from .environment import EMPTY_RAW, RawObservation

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_MAX_PATH_CHARS = 260


class _LastInputInfo(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint),
                ("dwTime", ctypes.c_uint32)]


class WindowsEnvironmentProbe:
    """Read-only Windows probe. Never constructs anything that writes
    to the OS. All failures return empty observations rather than
    raising."""

    def __init__(self, idle_threshold_s: float = 300.0) -> None:
        try:
            threshold = float(idle_threshold_s)
        except (TypeError, ValueError):
            threshold = 300.0
        self.idle_threshold_s = threshold

    def observe(self) -> RawObservation:
        if sys.platform != "win32":
            return EMPTY_RAW
        try:
            return RawObservation(process_name=self._query_process_name(),
                                  idle=self._query_idle())
        except Exception:
            return EMPTY_RAW

    def _query_process_name(self):
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        user32.GetForegroundWindow.restype = ctypes.c_void_p
        user32.GetWindowThreadProcessId.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
        user32.GetWindowThreadProcessId.restype = ctypes.c_uint32
        kernel32.OpenProcess.argtypes = (
            ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.QueryFullProcessImageNameW.argtypes = (
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_wchar_p,
            ctypes.POINTER(ctypes.c_uint32))
        kernel32.QueryFullProcessImageNameW.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        pid = ctypes.c_uint32(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return None
        handle = kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION, 0, pid.value)
        if not handle:
            return None
        try:
            buf = ctypes.create_unicode_buffer(_MAX_PATH_CHARS)
            size = ctypes.c_uint32(_MAX_PATH_CHARS)
            if not kernel32.QueryFullProcessImageNameW(
                    handle, 0, buf, ctypes.byref(size)):
                return None
            name = os.path.basename(buf.value).lower()
            return name or None
        finally:
            kernel32.CloseHandle(handle)

    def _query_idle(self) -> bool:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        kernel32.GetTickCount64.restype = ctypes.c_ulonglong
        user32.GetLastInputInfo.argtypes = (ctypes.c_void_p,)
        user32.GetLastInputInfo.restype = ctypes.c_int

        info = _LastInputInfo()
        info.cbSize = ctypes.sizeof(_LastInputInfo)
        if not user32.GetLastInputInfo(ctypes.byref(info)):
            return False
        idle_ms = kernel32.GetTickCount64() - info.dwTime
        if idle_ms < 0:
            idle_ms = 0
        return (idle_ms / 1000.0) >= self.idle_threshold_s

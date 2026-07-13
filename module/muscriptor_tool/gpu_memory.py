from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable

DXGI_ERROR_NOT_FOUND = 0x887A0002
DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL = 1
HRESULT = ctypes.c_long


class _GUID(ctypes.Structure):
    _fields_ = (
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    )

    @classmethod
    def from_string(cls, value: str) -> "_GUID":
        import uuid

        parsed = uuid.UUID(value)
        raw = parsed.bytes_le
        return cls(
            int.from_bytes(raw[0:4], "little"),
            int.from_bytes(raw[4:6], "little"),
            int.from_bytes(raw[6:8], "little"),
            (ctypes.c_ubyte * 8)(*raw[8:]),
        )


class _VideoMemoryInfo(ctypes.Structure):
    _fields_ = (
        ("budget", ctypes.c_uint64),
        ("current_usage", ctypes.c_uint64),
        ("available_for_reservation", ctypes.c_uint64),
        ("current_reservation", ctypes.c_uint64),
    )


_IID_IDXGI_FACTORY1 = _GUID.from_string("770aae78-f26f-4dba-a829-253c83d1b387")
_IID_IDXGI_ADAPTER3 = _GUID.from_string("645967a4-1392-4310-a798-8053ce3e93fd")


def _com_method(
    pointer: ctypes.c_void_p,
    index: int,
    restype: type,
    *argtypes: type,
) -> Callable[..., object]:
    vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(vtable[index])


def _release(pointer: ctypes.c_void_p) -> None:
    if pointer:
        _com_method(pointer, 2, wintypes.ULONG)(pointer)


def _query_adapter_non_local_usage(adapter: ctypes.c_void_p) -> int | None:
    adapter3 = ctypes.c_void_p()
    query_interface = _com_method(
        adapter,
        0,
        HRESULT,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    )
    result = int(query_interface(adapter, ctypes.byref(_IID_IDXGI_ADAPTER3), ctypes.byref(adapter3)))
    if result < 0 or not adapter3:
        return None
    try:
        info = _VideoMemoryInfo()
        query_video_memory_info = _com_method(
            adapter3,
            14,
            HRESULT,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(_VideoMemoryInfo),
        )
        result = int(
            query_video_memory_info(
                adapter3,
                0,
                DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
                ctypes.byref(info),
            )
        )
        return int(info.current_usage) if result >= 0 else None
    finally:
        _release(adapter3)


def query_process_shared_gpu_memory_bytes() -> int | None:
    """Return this process' largest non-local DXGI allocation on Windows."""

    if sys.platform != "win32":
        return None
    try:
        dxgi = ctypes.WinDLL("dxgi", use_last_error=True)
        create_factory = dxgi.CreateDXGIFactory1
        create_factory.argtypes = (
            ctypes.POINTER(_GUID),
            ctypes.POINTER(ctypes.c_void_p),
        )
        create_factory.restype = HRESULT
        factory = ctypes.c_void_p()
        result = int(create_factory(ctypes.byref(_IID_IDXGI_FACTORY1), ctypes.byref(factory)))
        if result < 0 or not factory:
            return None
    except (AttributeError, OSError):
        return None

    usages: list[int] = []
    try:
        enum_adapters = _com_method(
            factory,
            12,
            HRESULT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
        )
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            result = int(enum_adapters(factory, index, ctypes.byref(adapter)))
            if result & 0xFFFFFFFF == DXGI_ERROR_NOT_FOUND:
                break
            if result < 0 or not adapter:
                return max(usages) if usages else None
            try:
                usage = _query_adapter_non_local_usage(adapter)
                if usage is not None:
                    usages.append(usage)
            finally:
                _release(adapter)
            index += 1
    except (AttributeError, OSError, ValueError):
        return None
    finally:
        _release(factory)
    return max(usages) if usages else None


@dataclass
class SharedMemoryMonitor:
    reader: Callable[[], int | None] = query_process_shared_gpu_memory_bytes
    detection_threshold_bytes: int = 64 * 1024**2

    def sample(self) -> int | None:
        try:
            value = self.reader()
        except Exception:
            return None
        if value is None:
            return None
        return max(0, int(value))

    def grew_into_shared_memory(self, before: int | None, after: int | None) -> bool:
        if after is None:
            return False
        baseline = max(0, int(before or 0))
        return after >= self.detection_threshold_bytes and (
            after - baseline >= self.detection_threshold_bytes
        )

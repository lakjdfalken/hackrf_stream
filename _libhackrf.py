"""Minimal ctypes binding to libhackrf.

Only the calls a receive-only spectrum source needs. libhackrf is a system
library here, not a bundled one: it is loaded from wherever the platform put
it, the same copy hackrf_sweep uses. Nothing in this package links it at build
time and nothing ships a copy of it.
"""

import ctypes
import ctypes.util

#: The subset of libhackrf we call, as (name, argtypes, restype).
_SIGNATURES = (
    ("hackrf_init", (), ctypes.c_int),
    ("hackrf_exit", (), ctypes.c_int),
    ("hackrf_open", (ctypes.c_void_p,), ctypes.c_int),
    ("hackrf_open_by_serial", (ctypes.c_char_p, ctypes.c_void_p), ctypes.c_int),
    ("hackrf_close", (ctypes.c_void_p,), ctypes.c_int),
    ("hackrf_set_sample_rate", (ctypes.c_void_p, ctypes.c_double), ctypes.c_int),
    ("hackrf_set_freq", (ctypes.c_void_p, ctypes.c_uint64), ctypes.c_int),
    ("hackrf_set_lna_gain", (ctypes.c_void_p, ctypes.c_uint32), ctypes.c_int),
    ("hackrf_set_vga_gain", (ctypes.c_void_p, ctypes.c_uint32), ctypes.c_int),
    ("hackrf_set_amp_enable", (ctypes.c_void_p, ctypes.c_uint8), ctypes.c_int),
    ("hackrf_set_antenna_enable", (ctypes.c_void_p, ctypes.c_uint8), ctypes.c_int),
    ("hackrf_set_baseband_filter_bandwidth", (ctypes.c_void_p, ctypes.c_uint32), ctypes.c_int),
    ("hackrf_compute_baseband_filter_bw", (ctypes.c_uint32,), ctypes.c_uint32),
    ("hackrf_stop_rx", (ctypes.c_void_p,), ctypes.c_int),
    ("hackrf_is_streaming", (ctypes.c_void_p,), ctypes.c_int),
    ("hackrf_error_name", (ctypes.c_int,), ctypes.c_char_p),
    ("hackrf_library_release", (), ctypes.c_char_p),
    ("hackrf_usb_board_id_name", (ctypes.c_int,), ctypes.c_char_p),
)


class HackRFError(Exception):
    """A libhackrf call returned an error"""


class Transfer(ctypes.Structure):
    """hackrf_transfer, as declared in libhackrf/hackrf.h

    This layout is the one contract with libhackrf that could drift without
    telling us, so check_layout() asserts what we assume about it."""
    _fields_ = [
        ("device", ctypes.c_void_p),
        ("buffer", ctypes.POINTER(ctypes.c_uint8)),
        ("buffer_length", ctypes.c_int),
        ("valid_length", ctypes.c_int),
        ("rx_ctx", ctypes.c_void_p),
        ("tx_ctx", ctypes.c_void_p),
    ]


class DeviceList(ctypes.Structure):
    """hackrf_device_list, as declared in libhackrf/hackrf.h

    Only the first four fields are read; the libusb handles after them are
    libhackrf's own business."""
    _fields_ = [
        ("serial_numbers", ctypes.POINTER(ctypes.c_char_p)),
        ("usb_board_ids", ctypes.POINTER(ctypes.c_int)),
        ("usb_device_index", ctypes.POINTER(ctypes.c_int)),
        ("devicecount", ctypes.c_int),
        ("usb_devices", ctypes.POINTER(ctypes.c_void_p)),
        ("usb_devicecount", ctypes.c_int),
    ]


#: Signature of the receive callback libhackrf invokes on its own thread
RX_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(Transfer))


def load(path=None):
    """Load libhackrf and declare the calls we use"""
    if path is None:
        path = ctypes.util.find_library("hackrf")
    if path is None:
        # find_library misses Homebrew's prefix on some setups
        for candidate in ("libhackrf.so.0", "libhackrf.so", "libhackrf.0.dylib", "libhackrf.dylib"):
            try:
                return _declare(ctypes.CDLL(candidate))
            except OSError:
                continue
        raise HackRFError(
            "libhackrf not found. Install it with your package manager "
            "(brew install hackrf, apt install libhackrf0) — it is the same "
            "library hackrf_sweep uses."
        )
    return _declare(ctypes.CDLL(path))


def _declare(lib):
    """Attach argument and return types, so ctypes does not guess"""
    for name, argtypes, restype in _SIGNATURES:
        function = getattr(lib, name)
        function.argtypes = list(argtypes)
        function.restype = restype

    # start_rx is declared separately because its second argument is our callback
    lib.hackrf_start_rx.argtypes = [ctypes.c_void_p, RX_CALLBACK, ctypes.c_void_p]
    lib.hackrf_start_rx.restype = ctypes.c_int

    # ...and the device list because it returns a struct we declared above
    lib.hackrf_device_list.argtypes = []
    lib.hackrf_device_list.restype = ctypes.POINTER(DeviceList)
    lib.hackrf_device_list_free.argtypes = [ctypes.POINTER(DeviceList)]
    lib.hackrf_device_list_free.restype = None
    return lib


def check(lib, result, what):
    """Raise with libhackrf's own message when a call fails"""
    if result != 0:
        name = lib.hackrf_error_name(result)
        raise HackRFError("{}: {}".format(what, name.decode() if name else result))
    return result


def check_layout():
    """Fail loudly if hackrf_transfer is not the shape we compiled against

    A libhackrf that reordered or grew this struct would otherwise hand the
    callback garbage lengths and a wrong buffer pointer, which reads as noise
    rather than as an error."""
    pointer = ctypes.sizeof(ctypes.c_void_p)
    expected = pointer * 4 + ctypes.sizeof(ctypes.c_int) * 2
    if ctypes.sizeof(Transfer) < expected:
        raise HackRFError(
            "hackrf_transfer is {} bytes, expected at least {}. This libhackrf "
            "is not the one this binding was written against."
            .format(ctypes.sizeof(Transfer), expected)
        )

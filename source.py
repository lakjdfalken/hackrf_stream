"""A continuously tuned HackRF spectrum source.

The HackRF can sweep, retuning as it goes, and that is what hackrf_sweep does.
It is also capped by the retune: about 405 tuning steps a second whatever the
span or bin size, which is 8 GHz/s of coverage but only 6.6 MB/s of the 40 MB/s
the USB link can carry.

Staying on one frequency instead removes the retune entirely and uses the whole
stream. The cost is that one tune only ever sees one sample rate of spectrum,
so this is for looking closely at a band rather than searching a wide one.
"""

import threading
import time

import numpy as np

from . import dsp
from ._libhackrf import RX_CALLBACK, HackRFError, check, check_layout, load

#: Widest span a HackRF can digitise in one tune
MAX_SAMPLE_RATE = 20e6
MIN_SAMPLE_RATE = 2e6

_lib = None
_lib_lock = threading.Lock()
_open_sources = 0


def _library():
    """Load libhackrf once, and call hackrf_init once"""
    global _lib, _open_sources
    with _lib_lock:
        if _lib is None:
            check_layout()
            _lib = load()
            check(_lib, _lib.hackrf_init(), "hackrf_init")
        _open_sources += 1
        return _lib


def _release():
    """Call hackrf_exit once the last source has closed"""
    global _lib, _open_sources
    with _lib_lock:
        _open_sources -= 1
        if _open_sources <= 0 and _lib is not None:
            _lib.hackrf_exit()
            _lib, _open_sources = None, 0


def library_version():
    """Version string of the libhackrf actually loaded"""
    lib = _library()
    try:
        return lib.hackrf_library_release().decode()
    finally:
        _release()


class SpectrumSource:
    """Averaged power spectra from a HackRF held on one frequency

    Call start() with a callable; it is invoked as
    callback(frequencies, powers_db, timestamp) once per completed average,
    on libhackrf's own thread. Keep it short — anything slow belongs on a
    queue, because the sample stream does not pause for it.
    """

    def __init__(self, center_freq, sample_rate=MAX_SAMPLE_RATE, bin_size=None,
                 fft_size=None, average=26, gain=40, amp=False, antenna_power=False,
                 window="hann", dc_bins=2, serial=None):
        if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
            raise ValueError("sample_rate must be between {:g} and {:g} Hz".format(
                MIN_SAMPLE_RATE, MAX_SAMPLE_RATE))
        if fft_size is None:
            if bin_size is None:
                raise ValueError("give either bin_size or fft_size")
            fft_size = dsp.fast_fft_size(sample_rate, bin_size)

        self.center_freq = float(center_freq)
        self.sample_rate = float(sample_rate)
        self.fft_size = int(fft_size)
        self.average = int(average)
        self.gain = gain
        self.amp = amp
        self.antenna_power = antenna_power
        self.serial = serial

        self.frequencies = dsp.frequencies(self.center_freq, self.sample_rate, self.fft_size)
        self.bin_size = self.sample_rate / self.fft_size
        self._accumulator = dsp.SpectrumAccumulator(self.fft_size, self.average, window, dc_bins)

        self._lib = None
        self._device = None
        self._callback = None
        # libhackrf keeps only the raw pointer, so the CFUNCTYPE must outlive
        # the transfer or it is collected and the callback jumps into freed memory
        self._trampoline = None
        self._error = None
        self._samples = 0
        self._spectra = 0
        self._started_at = None

    # -- lifecycle ----------------------------------------------------

    def open(self):
        """Open the device and apply the radio settings"""
        if self._device is not None:
            return

        import ctypes
        self._lib = _library()
        device = ctypes.c_void_p()
        try:
            if self.serial:
                check(self._lib, self._lib.hackrf_open_by_serial(
                    self.serial.encode(), ctypes.byref(device)), "hackrf_open_by_serial")
            else:
                check(self._lib, self._lib.hackrf_open(ctypes.byref(device)), "hackrf_open")
            self._device = device
            self._configure()
        except Exception:
            if self._device is not None:
                self._lib.hackrf_close(self._device)
                self._device = None
            _release()
            self._lib = None
            raise

    def _configure(self):
        """Push sample rate, filter, frequency and gains to the radio"""
        lib, device = self._lib, self._device
        check(lib, lib.hackrf_set_sample_rate(device, self.sample_rate), "set_sample_rate")
        bandwidth = lib.hackrf_compute_baseband_filter_bw(int(0.75 * self.sample_rate))
        check(lib, lib.hackrf_set_baseband_filter_bandwidth(device, bandwidth),
              "set_baseband_filter_bandwidth")
        check(lib, lib.hackrf_set_freq(device, int(self.center_freq)), "set_freq")

        lna, vga = dsp.split_gain(self.gain)
        check(lib, lib.hackrf_set_lna_gain(device, lna), "set_lna_gain")
        check(lib, lib.hackrf_set_vga_gain(device, vga), "set_vga_gain")
        check(lib, lib.hackrf_set_amp_enable(device, 1 if self.amp else 0), "set_amp_enable")
        check(lib, lib.hackrf_set_antenna_enable(device, 1 if self.antenna_power else 0),
              "set_antenna_enable")

    def start(self, callback):
        """Begin receiving, delivering spectra to callback"""
        self.open()
        self._error = None
        self._samples = 0
        self._spectra = 0
        self._accumulator.reset()
        self._callback = callback
        self._started_at = time.monotonic()
        self._trampoline = RX_CALLBACK(self._on_transfer)
        check(self._lib, self._lib.hackrf_start_rx(self._device, self._trampoline, None),
              "hackrf_start_rx")

    def stop(self):
        """Stop receiving, leaving the device open"""
        if self._device is not None and self._lib is not None:
            self._lib.hackrf_stop_rx(self._device)
        self._trampoline = None
        self._callback = None

    def close(self):
        """Stop and release the device"""
        self.stop()
        if self._device is not None:
            self._lib.hackrf_close(self._device)
            self._device = None
            _release()
            self._lib = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    # -- the hot path -------------------------------------------------

    def _on_transfer(self, transfer_pointer):
        """Called by libhackrf on its own thread for every block of samples

        Anything raised here would unwind into C, so failures are recorded and
        the stream is asked to stop instead."""
        try:
            transfer = transfer_pointer.contents
            length = transfer.valid_length
            if length <= 0:
                return 0
            self._samples += length
            block = np.ctypeslib.as_array(transfer.buffer, shape=(length,)).view(np.int8)
            for spectrum in self._accumulator.feed(block):
                self._spectra += 1
                if self._callback is not None:
                    self._callback(self.frequencies, spectrum, time.time())
        except Exception as error:          # noqa: BLE001 - must not escape into C
            self._error = error
            return 1                        # non-zero asks libhackrf to stop
        return 0

    # -- state --------------------------------------------------------

    @property
    def streaming(self):
        """True while libhackrf is still delivering samples"""
        if self._device is None or self._lib is None:
            return False
        return self._lib.hackrf_is_streaming(self._device) == 1

    @property
    def error(self):
        """Whatever the callback raised, if it stopped that way"""
        return self._error

    def statistics(self):
        """Throughput since start(), for checking nothing is being dropped"""
        if not self._started_at:
            return {}
        elapsed = max(time.monotonic() - self._started_at, 1e-9)
        return {
            "seconds": elapsed,
            "bytes_per_second": self._samples / elapsed,
            "spectra_per_second": self._spectra / elapsed,
            "ffts_per_second": self._accumulator.ffts / elapsed,
            "stream_fraction": (self._samples / elapsed) / (self.sample_rate * 2),
        }

"""A continuously tuned HackRF spectrum source.

The HackRF can sweep, retuning as it goes, and that is what hackrf_sweep does.
It is also capped by the retune: about 405 tuning steps a second whatever the
span or bin size, which is 8 GHz/s of coverage but only 6.6 MB/s of the 40 MB/s
the USB link can carry.

Staying on one frequency instead removes the retune entirely and uses the whole
stream. The cost is that one tune only ever sees one sample rate of spectrum,
so this is for looking closely at a band rather than searching a wide one.
"""

import collections
import ctypes
import queue
import threading
import time

import numpy as np

from . import dsp
from ._libhackrf import RX_CALLBACK, HackRFError, check, check_layout, load

#: Widest span a HackRF can digitise in one tune
MAX_SAMPLE_RATE = 20e6
MIN_SAMPLE_RATE = 2e6

#: Transfers the receive thread may run ahead of the FFT thread. At 20 MSPS a
#: transfer is 256 kB and arrives every 6.5 ms, so this is about 800 ms of
#: slack (33 MB): enough to ride out a slow repaint, a window resize or a
#: garbage collection without the radio noticing, and bounded so that a machine
#: that simply cannot keep up drops samples instead of growing a backlog it
#: will never work off. The FFT thread only needs a tenth of real time, so a
#: drop is nearly always a stall somewhere else rather than a shortage of CPU,
#: and a stall is exactly what a deep queue rides out.
QUEUE_DEPTH = 128

#: Bytes in one libhackrf transfer, for turning a count of dropped ones into
#: how much signal that actually was
TRANSFER_BYTES = 262144

#: Blocks of band readings held for a reader that has not come back for them.
#: One block per transfer, so at 20 MSPS this is about ten seconds of slack; a
#: reader that stops reading loses the oldest rather than the newest, and never
#: grows the process.
BAND_BACKLOG = 1536

_lib = None
_lib_lock = threading.Lock()
_open_sources = 0


#: The baseband filters a HackRF has, in hertz. libhackrf picks the widest one
#: no wider than what is asked for, and the radio is asked for three quarters
#: of the sample rate - so at 20 MSPS the passband is 15 MHz, not the 20 MHz
#: that Nyquist would suggest. What falls outside it is filter skirt rather
#: than spectrum, which is why offset_tune() needs to be told: a span placed
#: by Nyquist alone can sit partly outside the filter and be attenuated there,
#: which on a plot looks exactly like the band going quiet at one end.
FILTER_BANDWIDTHS = (1750000.0, 2500000.0, 3500000.0, 5000000.0, 5500000.0,
                     6000000.0, 7000000.0, 8000000.0, 9000000.0, 10000000.0,
                     12000000.0, 14000000.0, 15000000.0, 20000000.0,
                     24000000.0, 28000000.0)


def baseband_filter_bw(requested):
    """The widest filter no wider than `requested`, the way libhackrf chooses

    Pure arithmetic, so it can be known before the radio is opened - which is
    when the tune has to be settled. The device itself is still configured
    from libhackrf's own answer; this is for planning the span."""
    fits = [bw for bw in FILTER_BANDWIDTHS if bw <= requested]
    return fits[-1] if fits else FILTER_BANDWIDTHS[0]


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


def devices():
    """Every HackRF attached to this machine, as dicts

    Each has a "serial" — what SpectrumSource(serial=...) and hackrf_sweep's
    -d both want — and a "board" naming the hardware. Enumerating does not
    open anything, so it is safe to call while a device is streaming, but a
    board already in use by another program still appears here.

    Serials can be None: libhackrf lists a device it could not read the
    descriptor of, which is usually one that something else has claimed."""
    lib = _library()
    try:
        listing = lib.hackrf_device_list()
        if not listing:
            return []
        try:
            found = []
            for i in range(listing.contents.devicecount):
                serial = listing.contents.serial_numbers[i]
                board = lib.hackrf_usb_board_id_name(listing.contents.usb_board_ids[i])
                found.append({
                    "serial": serial.decode(errors="replace") if serial else None,
                    "board": board.decode(errors="replace") if board else "HackRF",
                })
            return found
        finally:
            lib.hackrf_device_list_free(listing)
    finally:
        _release()


class SpectrumSource:
    """Averaged power spectra from a HackRF held on one frequency

    Call start() with a callable; it is invoked as
    callback(frequencies, powers_db, timestamp) once per completed average,
    on this object's own FFT thread. Keep it short — anything slow belongs on
    a queue, because the sample stream does not pause for it.
    """

    def __init__(self, center_freq, sample_rate=MAX_SAMPLE_RATE, bin_size=None,
                 fft_size=None, average=26, gain=40, lna=None, vga=None, amp=False,
                 antenna_power=False, window="hann", dc_bins=None, serial=None, mode="mean"):
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
        #: The two analogue stages, settled here rather than at the radio, so
        #: that a caller can read back what it is actually about to get
        self.lna, self.vga = dsp.stage_gains(gain, lna, vga)
        self.gain = self.lna + self.vga
        self.amp = amp
        self.antenna_power = antenna_power
        self.serial = serial

        self.frequencies = dsp.frequencies(self.center_freq, self.sample_rate, self.fft_size)
        self.bin_size = self.sample_rate / self.fft_size
        #: Seconds of signal behind each finished spectrum. Every spectrum is
        #: exactly this long, so counting them times the stream far better
        #: than a clock read can: the FFT thread works through a whole
        #: transfer at once, and stamping there would bunch a transfer's worth
        #: of spectra onto one instant and leave a gap until the next.
        self.spectrum_duration = self.fft_size * self.average / self.sample_rate
        self.mode = mode
        self._accumulator = dsp.SpectrumAccumulator(self.fft_size, self.average, window,
                                                    dc_bins, mode)

        #: The baseband filter the radio was actually put to, once it is open.
        #: Chosen from the sample rate, and not the same as it: what reaches
        #: the FFT is limited by this, not by the span the bins cover.
        self.filter_bandwidth = None

        self._lib = None
        self._device = None
        self._callback = None
        # libhackrf keeps only the raw pointer, so the CFUNCTYPE must outlive
        # the transfer or it is collected and the callback jumps into freed memory
        self._trampoline = None
        self._error = None
        self._samples = 0
        self._spectra = 0
        self._dropped = 0
        self._started_at = None
        self._blocks = None
        self._worker = None
        # What the FFT thread is doing with its time, so that a drop can say
        # whether the DSP is too slow or is being starved. Both are needed:
        # wall time alone cannot tell working from waiting, and in Python a
        # thread waiting for the interpreter lock is doing exactly that.
        self._busy_seconds = 0.0
        self._cpu_seconds = 0.0
        self._queue_peak = 0
        # Band power at the frame rate; see set_band()
        self._band_range = None
        self._band_resolution = None
        self._band_detector = None
        #: True while the tap reads samples rather than bins; see set_band()
        self._band_magnitude = False
        self._band_readings = 0
        self._band_error = None
        self._band_samples = collections.deque(maxlen=BAND_BACKLOG)
        self._stream_start = None
        self._lost_seconds = 0.0

    # -- lifecycle ----------------------------------------------------

    def open(self):
        """Open the device and apply the radio settings"""
        if self._device is not None:
            return

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
        self.filter_bandwidth = float(bandwidth)
        check(lib, lib.hackrf_set_freq(device, int(self.center_freq)), "set_freq")

        check(lib, lib.hackrf_set_lna_gain(device, self.lna), "set_lna_gain")
        check(lib, lib.hackrf_set_vga_gain(device, self.vga), "set_vga_gain")
        check(lib, lib.hackrf_set_amp_enable(device, 1 if self.amp else 0), "set_amp_enable")
        check(lib, lib.hackrf_set_antenna_enable(device, 1 if self.antenna_power else 0),
              "set_antenna_enable")

    def start(self, callback):
        """Begin receiving, delivering spectra to callback"""
        self.open()
        self._error = None
        self._samples = 0
        self._spectra = 0
        self._dropped = 0
        self._accumulator.reset()
        self._band_samples.clear()
        self._band_readings = 0
        self._band_error = None
        self._busy_seconds = 0.0
        self._cpu_seconds = 0.0
        self._queue_peak = 0
        self._callback = callback
        self._started_at = time.monotonic()
        self._stream_start = time.time()
        self._lost_seconds = 0.0

        # The FFTs run here rather than on libhackrf's thread. They are a
        # sixth of a core at 20 MSPS, and in Python they hold the GIL for the
        # whole of it, so leaving them in the receive callback let anything
        # else holding the GIL — a plot being repainted, say — stall the
        # callback and cost the radio the samples that arrived meanwhile.
        self._blocks = queue.Queue(maxsize=QUEUE_DEPTH)
        self._worker = threading.Thread(target=self._process, name="hackrf-fft",
                                        daemon=True)
        self._worker.start()

        self._trampoline = RX_CALLBACK(self._on_transfer)
        try:
            check(self._lib, self._lib.hackrf_start_rx(self._device, self._trampoline, None),
                  "hackrf_start_rx")
        except Exception:
            self._shutdown_worker()
            self._trampoline = None
            raise

    def stop(self):
        """Stop receiving, leaving the device open"""
        if self._device is not None and self._lib is not None:
            self._lib.hackrf_stop_rx(self._device)
        # Only now that no more transfers can arrive is it safe to retire the
        # trampoline libhackrf holds a raw pointer to
        self._trampoline = None
        self._shutdown_worker()
        self._callback = None

    def _shutdown_worker(self):
        """Ask the FFT thread to finish and wait for it"""
        worker, blocks = self._worker, self._blocks
        self._worker = None
        if worker is None:
            return
        if blocks is not None:
            try:
                blocks.put_nowait(None)
            except queue.Full:
                # Full of samples nobody will look at now; clear a slot for
                # the sentinel rather than waiting for the worker to drain
                try:
                    blocks.get_nowait()
                    blocks.put_nowait(None)
                except (queue.Empty, queue.Full):
                    pass
        worker.join(timeout=5.0)
        self._blocks = None

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

    # -- watching one band --------------------------------------------

    def set_band(self, low_hz, high_hz, resolution=None, detector="peak"):
        """Follow one band's power in the time domain — a zero span view

        The radio cannot sit on a single frequency; it digitises a whole
        sample rate at once. But one bin of an FFT is a filter, so reducing
        every frame to the peak across a few bins is the same measurement a
        spectrum analyser makes in zero span, with a resolution bandwidth of
        the bin size.

        This reads the frames, not the spectra that are delivered. Frames are
        produced every fft_size/sample_rate — 25.6 us at 20 MSPS with 40 kHz
        bins — where delivered spectra are averaged down to a hundred a
        second, so a burst too short to survive the averaging is still
        measured at full height. `resolution` asks for seconds per reading and
        is rounded to whole frames; None gives one reading per frame, which is
        as fine as the radio can be read. `detector` decides how the frames in
        one reading are combined, which is the video bandwidth choice a
        spectrum analyser makes: "peak" keeps a pulse shorter than the reading
        at its own height, "mean" smooths as the square root of the count.

        What it cannot do is look backwards: it measures the band asked for
        here, so changing the band starts again. Pass None for either edge to
        stop watching."""
        if low_hz is None or high_hz is None:
            self._accumulator.clear_band()
            self._accumulator.clear_magnitude()
            self._band_range = self._band_resolution = None
            self._band_detector = None
            self._band_magnitude = False
            self._band_samples.clear()
            self._band_readings = 0
            return

        low, high = sorted((float(low_hz), float(high_hz)))
        first = int(np.searchsorted(self.frequencies, low, side="left"))
        last = int(np.searchsorted(self.frequencies, high, side="right"))
        # A band narrower than one bin, or off the end of the tune, still has
        # to name a bin rather than an empty slice
        first = min(max(first, 0), self.fft_size - 1)
        last = min(max(last, first + 1), self.fft_size)

        frame = self.fft_size / self.sample_rate

        if resolution and float(resolution) < frame:
            # Finer than one frame, which no bin size can deliver: the frame
            # is fft_size/sample_rate and the FFT bottoms out at 16 points.
            # Read the samples instead. This ignores the band it was given -
            # there are no bins to select with - so the caller is told through
            # band_magnitude and band(), which then name the whole passband.
            group = max(1, int(round(float(resolution) * self.sample_rate)))
            # "total" is a question about bins, and there are none here
            detector = "peak" if detector == "total" else detector
            self._band_samples.clear()
            self._band_readings = 0
            self._accumulator.set_magnitude(group, detector)
            self._band_range = (float(self.frequencies[0]),
                                float(self.frequencies[-1]))
            self._band_resolution = group / self.sample_rate
            self._band_detector = detector
            self._band_magnitude = True
            return

        group = 1 if not resolution else max(1, int(round(float(resolution) / frame)))

        self._band_samples.clear()
        self._band_readings = 0
        self._accumulator.set_band(first, last, group, detector)
        self._band_range = (float(self.frequencies[first]),
                            float(self.frequencies[last - 1]))
        self._band_resolution = group * frame
        self._band_detector = detector
        self._band_magnitude = False

    def set_gain(self, lna=None, vga=None, amp=None):
        """Change the analogue gain while the radio is running

        All three are control transfers the radio accepts mid-stream, so this
        takes effect on the next samples rather than at the next restart.
        That matters more than it sounds: gain is the control you reach for
        when a signal will not show, and a control that silently does nothing
        until the run is restarted reads exactly like a signal that is not
        there. Whatever is left None is not touched.

        Returns the (lna, vga, amp) actually in force, with the stages rounded
        down to steps the radio has."""
        if lna is not None or vga is not None:
            self.lna, self.vga = dsp.stage_gains(
                -1, self.lna if lna is None else lna,
                self.vga if vga is None else vga)
            self.gain = self.lna + self.vga
        if amp is not None:
            self.amp = bool(amp)

        device, lib = self._device, self._lib
        if device is not None and lib is not None:
            if lna is not None or vga is not None:
                check(lib, lib.hackrf_set_lna_gain(device, self.lna), "set_lna_gain")
                check(lib, lib.hackrf_set_vga_gain(device, self.vga), "set_vga_gain")
            if amp is not None:
                check(lib, lib.hackrf_set_amp_enable(device, 1 if self.amp else 0),
                      "set_amp_enable")
        return self.lna, self.vga, self.amp

    @property
    def dc_bins(self):
        """Bins either side of the centre being flattened, as settled on"""
        return self._accumulator.dc_bins

    @property
    def dc_band(self):
        """Where the receiver's own carrier is, as (low, high) in hertz

        Always there and always at the centre of the tune, whether or not the
        delivered spectra have it flattened. Worth asking for separately from
        dc_bins because the band tap reads raw bins: a zero span watch pointed
        here is measuring the radio rather than the air, and nothing about the
        reading says so."""
        half = (max(self.dc_bins, 0) + 0.5) * self.bin_size
        return (self.center_freq - half, self.center_freq + half)

    @property
    def band(self):
        """The band actually being watched, snapped to bin edges"""
        return self._band_range

    @property
    def band_error(self):
        """Whatever stopped the band tap, if anything did"""
        return self._band_error

    @property
    def band_resolution(self):
        """Seconds each band reading covers, or None when not watching"""
        return self._band_resolution

    @property
    def band_magnitude(self):
        """True while the tap is reading samples rather than bins

        Worth asking before believing band(): a magnitude tap measures the
        whole passband whatever band was requested, so a caller that reports
        the requested band would be describing a filter that is not there."""
        return self._band_magnitude

    @property
    def band_detector(self):
        """How the frames in one band reading are combined"""
        return self._band_detector

    @property
    def frame_duration(self):
        """Seconds of signal in one FFT frame — the finest band resolution"""
        return self.fft_size / self.sample_rate

    @property
    def stream_start(self):
        """The clock time the stream began, or None before it has

        Band readings are stamped in seconds from here rather than in seconds
        from 1970, so that their spacing survives being written down. This is
        what puts them back on a wall clock."""
        return self._stream_start

    def take_band_power(self):
        """Take the band samples gathered since the last call

        Returns an (N, 2) array of time and power in dB, oldest first, or None
        if there are none. Times are seconds since stream_start. Draining in
        bulk rather than calling back per sample keeps a thousand-odd readings
        a second off the reader's thread."""
        samples = self._band_samples
        taken = []
        try:
            while True:
                taken.append(samples.popleft())
        except IndexError:
            pass                            # emptied it
        if not taken:
            return None
        return np.concatenate(taken) if len(taken) > 1 else taken[0]

    # -- the hot path -------------------------------------------------

    def _on_transfer(self, transfer_pointer):
        """Called by libhackrf on its own thread for every block of samples

        libhackrf reuses the transfer buffer as soon as this returns, and it
        cannot resubmit the transfer until it does, so this copies the bytes
        and hands them straight to the FFT thread. That copy is tens of
        microseconds where doing the work here was over a millisecond.

        Anything raised here would unwind into C, so failures are recorded and
        the stream is asked to stop instead."""
        try:
            transfer = transfer_pointer.contents
            length = transfer.valid_length
            if length <= 0:
                return 0
            blocks = self._blocks
            if blocks is None:
                # stop() got here first; libhackrf may still have a transfer
                # in flight, and there is nowhere left to put it
                return 0
            self._samples += length
            depth = blocks.qsize()
            if depth > self._queue_peak:
                self._queue_peak = depth
            try:
                blocks.put_nowait(ctypes.string_at(transfer.buffer, length))
            except queue.Full:
                # The FFT thread is behind. Dropping this transfer loses the
                # samples in it; blocking here would lose them anyway, and
                # every transfer after it as well.
                self._dropped += 1
                # The signal in them still happened, so the stream clock has
                # to step over the hole or everything after it reads early
                self._lost_seconds += (length / 2) / self.sample_rate
        except Exception as error:          # noqa: BLE001 - must not escape into C
            self._error = error
            return 1                        # non-zero asks libhackrf to stop
        return 0

    def _process(self):
        """FFT thread: turn queued sample blocks into spectra"""
        blocks = self._blocks
        while True:
            block = blocks.get()
            if block is None:
                return
            started, started_cpu = time.monotonic(), time.thread_time()
            try:
                samples = np.frombuffer(block, dtype=np.int8)
                for spectrum in self._accumulator.feed(samples):
                    self._spectra += 1
                    # When the signal in this spectrum reached the antenna,
                    # counted in samples rather than read off a clock
                    when = (self._stream_start + self._lost_seconds
                            + self._spectra * self.spectrum_duration)
                    callback = self._callback
                    if callback is not None:
                        callback(self.frequencies, spectrum, when)
            except Exception as error:      # noqa: BLE001 - report, do not die silently
                self._error = error
                return

            try:
                self._collect_band()
            except Exception as error:      # noqa: BLE001 - the tap is optional
                # The band tap is a second reading of spectra that are being
                # delivered anyway. Losing it should cost the tap, not the
                # radio, so it is switched off and reported rather than
                # stopping the stream.
                self._band_error = error
                self._accumulator.clear_band()
                self._band_range = self._band_resolution = None
            self._busy_seconds += time.monotonic() - started
            self._cpu_seconds += time.thread_time() - started_cpu

    def _collect_band(self):
        """Stamp this block's band readings and queue them for the reader

        Kept as whole blocks rather than one entry per reading: at a reading
        every 25 us that is forty thousand tuples a second built and torn down
        on the thread doing the FFTs, against one array per transfer here."""
        # Read the resolution once, and before taking the readings. The band
        # is set and cleared from whichever thread owns the display, so it can
        # go away between the frames being reduced and the readings being
        # collected; readings with no resolution to stamp them belong to a
        # band nobody is watching any more, and are dropped rather than
        # stopping the radio.
        step = self._band_resolution
        readings = self._accumulator.take_band()
        begin = self._accumulator.band_start_frame
        if readings is None or step is None or begin is None:
            return
        # From where this band began, not from where the stream did. Pointing
        # the tap somewhere else starts its readings again from zero, and
        # counting those from the start of the stream stamped the whole trace
        # as far in the past as the radio had been running - so it fell
        # outside every window drawn and the pane went back to the delivered
        # sweeps without saying so.
        # Counted from the start of the stream, not from the epoch. Absolute
        # Unix time in a float is quantised to 2**-22 s in this decade — 0.24
        # us, a third of a reading at 20 MSPS with 16-point frames — so
        # stamping readings on that clock loses the very spacing the tap
        # exists to measure. stream_start says where this zero is; adding it
        # back is the caller's business, and it is one addition per window
        # rather than one per reading.
        start = (self._lost_seconds
                 + begin * self.frame_duration
                 + self._band_readings * step)
        # Stamped at the end of the frames behind it, as spectra are
        times = start + np.arange(1, len(readings) + 1) * step
        self._band_readings += len(readings)
        self._band_samples.append(
            np.column_stack((times, readings)).astype(np.float64))

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
            "dropped_transfers": self._dropped,
            "busy_fraction": self._busy_seconds / elapsed,
            "cpu_fraction": self._cpu_seconds / elapsed,
            "queue_peak": self._queue_peak,
            "queue_depth": QUEUE_DEPTH,
        }

    @property
    def busy_fraction(self):
        """How much of real time the FFT thread spends inside its work

        Not the same as how much work it does: a thread waiting for the
        interpreter lock is inside the work and doing none of it. Compare
        with cpu_fraction to tell those apart."""
        if not self._started_at:
            return 0.0
        return self._busy_seconds / max(time.monotonic() - self._started_at, 1e-9)

    @property
    def cpu_fraction(self):
        """How much of real time the FFT thread spends on the CPU

        This is the one that says whether the DSP is too slow. Near one and it
        genuinely cannot keep up; well under one while busy_fraction is near
        one and it is being starved by something else holding the lock."""
        if not self._started_at:
            return 0.0
        return self._cpu_seconds / max(time.monotonic() - self._started_at, 1e-9)

    @property
    def queue_peak(self):
        """The deepest the block queue has been, out of QUEUE_DEPTH"""
        return self._queue_peak

    def take_queue_peak(self):
        """The deepest the queue has been since this was last asked

        Reset on reading. A peak that is never forgotten reports the first
        seconds of a run — when the display is building a waterfall, filling
        curves and warming its caches — for as long as the radio stays open,
        so every message minutes later blames a stall that happened once at
        the start and has not happened since."""
        peak = self._queue_peak
        self._queue_peak = 0
        return peak

    @property
    def transfer_seconds(self):
        """Seconds of signal in one dropped transfer, for reporting"""
        return (TRANSFER_BYTES / 2) / self.sample_rate

    @property
    def dropped(self):
        """Transfers thrown away because the FFT thread could not keep up"""
        return self._dropped

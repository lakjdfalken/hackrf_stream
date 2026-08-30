"""Turning a stream of 8 bit I/Q samples into averaged power spectra."""

import numpy as np

#: HackRF sends interleaved signed 8 bit I and Q
SAMPLE_DTYPE = np.int8


def fast_fft_size(sample_rate, bin_size):
    """FFT length closest to the requested bin size that numpy is quick at

    numpy is fastest on powers of two and slow on large prime lengths, so the
    requested bin size is treated as a request. The caller should read the bin
    size actually used back out of frequencies()."""
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    requested = sample_rate / bin_size
    return max(16, int(2 ** round(np.log2(requested))))


def frequencies(center_freq, sample_rate, fft_size):
    """Frequency of every bin, in ascending order"""
    return np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / sample_rate)) + center_freq


def split_gain(gain_db):
    """Split one gain figure across the HackRF's two analogue stages

    The LNA moves in 8 dB steps to 40, the VGA in 2 dB steps to 62. Negative
    means leave both at zero.

    The LNA comes first, and is filled first. It is the stage that sets what
    the receiver can hear: gain there lifts the signal above the noise of
    everything after it. The VGA is at baseband and amplifies whatever the LNA
    already let through, noise included, so gain spent there makes the trace
    bigger without making the radio any more sensitive. Spreading a modest
    request across both — 24 dB as 8 in the LNA and 16 in the VGA — throws
    away most of what asking for it was supposed to buy."""
    if gain_db < 0:
        return 0, 0
    gain_db = min(gain_db, 102)
    lna = min(40, 8 * (int(gain_db) // 8))
    vga = min(62, 2 * ((int(gain_db) - lna) // 2))
    return lna, vga


def remove_dc_spike(spectrum, bins=2):
    """Flatten the receiver's own carrier at the centre of the band, in place

    Tuning to one frequency puts the radio's DC offset exactly in the middle of
    the spectrum, where it shows as a peak that is not on the air and is usually
    the strongest thing on screen. Interpolate across it from its neighbours.

    A wider fix is to tune off to one side and shift back, which is what
    hackrf_sweep does; this keeps the whole band usable instead of half of it,
    at the cost of a few bins that are guessed rather than measured."""
    if bins <= 0:
        return spectrum

    centre = spectrum.size // 2
    low, high = centre - bins, centre + bins
    if low - 1 < 0 or high + 1 >= spectrum.size:
        return spectrum

    left, right = spectrum[low - 1], spectrum[high + 1]
    spectrum[low:high + 1] = np.linspace(left, right, high - low + 1)
    return spectrum


class SpectrumAccumulator:
    """Combines FFTs of incoming samples into finished spectra

    Sample blocks do not arrive aligned to anything, so leftover samples are
    carried into the next block rather than dropped: at 20 MSPS every discarded
    sample is measurement thrown away.

    How the frames are combined decides what the result is good for:

    "mean" averages them, which lowers the noise floor by the square root of
    the count and is what you want for a signal that is always there.

    "peak" keeps the loudest frame instead, which is what you want for one
    that is not. A radar pulse a microsecond long inside a 26 frame average
    is one frame of signal and 25 of noise, and averaging spreads it over all
    26 — around 12 dB thrown away on a 0.1% duty cycle. Peak keeps the frame
    it actually arrived in at full height, while still handing over spectra at
    the same slow rate, so a long average becomes a wider net to catch a pulse
    in rather than a deeper hole to lose it down."""

    #: How the frames making up one spectrum may be combined
    MODES = ("mean", "peak")

    def __init__(self, fft_size, average, window="hann", dc_bins=2, mode="mean"):
        if average < 1:
            raise ValueError("average must be at least 1")
        if mode not in self.MODES:
            raise ValueError("unknown mode {!r}, expected one of {}".format(
                mode, ", ".join(self.MODES)))
        self.fft_size = fft_size
        self.average = average
        self.dc_bins = dc_bins
        self.mode = mode
        self.window = _window(window, fft_size)
        # Coherent gain, so a full scale sine reads 0 dBFS whatever the window.
        # (Summing the squares instead would normalise for noise density, and
        # would put a full scale tone 25 dB high with a Hann window.)
        self._window_gain = float(np.sum(self.window))
        self._tail = np.empty(0, dtype=np.float32)
        self._acc = None
        self._count = 0
        self.ffts = 0
        # One reading per group of frames, for the band tap; see set_band()
        self._band_index = None
        self._band_group = 1
        self._band_detector = "peak"
        self._band_carry = None
        self._band_out = []
        # What one frame's power has to be divided by for 0 dBFS to be a full
        # scale sine, the single frame case of the scale _finish() works out
        self._frame_scale = (self._window_gain * 127.0) ** 2

    #: How the frames making up one band reading may be combined
    DETECTORS = ("peak", "mean")

    def set_band(self, first, last, group=1, detector="peak"):
        """Watch one band of bins, one reading per `group` frames

        `first` and `last` number the bins the way frequencies() does, lowest
        frequency first, which is not the order the FFT produces them in; the
        mapping is worked out once here so that the hot path is a plain take.

        The group is *not* the same as `average`. A display wants frames
        averaged, which pulls the noise floor down by the square root of the
        count and smears anything short across the whole average. Watching for
        a burst wants few enough frames per reading to see how long the burst
        lasted. So a reading is noisier than a delivered spectrum by design,
        and that is the price of seeing microseconds instead of milliseconds.

        `detector` decides how a group is combined, which is the same choice a
        spectrum analyser makes with its video bandwidth:

        "peak" keeps the loudest frame, so a pulse shorter than the group
        still reads at its own height. It does not smooth: one frame of noise
        wobbles by about 3.3 dB and a group of 39 still wobbles by 1.1.

        "mean" averages them, which smooths as the square root of the count —
        3.3 dB at one frame, 0.5 dB at 39 — and is what makes the shape of a
        signal legible when it is only a few dB out of the noise."""
        first = min(max(int(first), 0), self.fft_size - 1)
        last = min(max(int(last), first + 1), self.fft_size)
        # fftshift moves the upper half of the spectrum to the front, so bin i
        # counted from the lowest frequency lives at (i + n/2) % n unshifted
        half = self.fft_size // 2
        if detector not in self.DETECTORS:
            raise ValueError("unknown detector {!r}, expected one of {}".format(
                detector, ", ".join(self.DETECTORS)))
        self._band_index = (np.arange(first, last) + half) % self.fft_size
        self._band_group = max(1, int(group))
        self._band_detector = detector
        self._band_carry = None
        self._band_out = []

    def clear_band(self):
        """Stop watching a band"""
        self._band_index = None
        self._band_carry = None
        self._band_out = []

    def take_band(self):
        """Band readings finished since the last call, in dB, oldest first"""
        if not self._band_out:
            return None
        taken = np.concatenate(self._band_out) if len(self._band_out) > 1 else self._band_out[0]
        self._band_out = []
        return taken

    def _watch_band(self, power):
        """Reduce this block's frames to band readings

        Vectorised over the whole block, so the cost does not depend on how
        many readings come out of it: at 20 MSPS a transfer is a couple of
        hundred frames and the band is a handful of bins wide."""
        values = power[:, self._band_index].max(axis=1)
        if self._band_carry is not None:
            values = np.concatenate((self._band_carry, values))
            self._band_carry = None

        group = self._band_group
        complete = values.shape[0] // group
        if complete:
            frames = values[:complete * group].reshape(complete, group)
            done = frames.max(axis=1) if self._band_detector == "peak" else frames.mean(axis=1)
            self._band_out.append(10.0 * np.log10(done / self._frame_scale + 1e-20))
        rest = values[complete * group:]
        if rest.size:
            # Frames left over start the next reading rather than being
            # dropped: at 25 us a frame, throwing one away is throwing away
            # the burst it might have been
            self._band_carry = rest

    def feed(self, samples):
        """Add raw interleaved I/Q and yield every spectrum it completes

        Yields power in dB, one array per completed average, lowest frequency
        first.

        This is the whole cost of the backend, and it runs while the radio is
        still filling its next buffer, so it is written to touch the data as
        few times as possible: no per-frame Python, and no temporary that only
        exists to be thrown away."""
        block = samples.astype(np.float32)
        if self._tail.size:
            block = np.concatenate((self._tail, block))

        pairs = block.size // 2
        usable = (pairs // self.fft_size) * self.fft_size
        if usable == 0:
            self._tail = block
            return

        self._tail = block[usable * 2:].copy()
        # Interleaved float32 I, Q is already the memory layout of complex64,
        # so this reinterprets the buffer instead of building a second one
        frames = block[:usable * 2].view(np.complex64).reshape(-1, self.fft_size)
        spectra = np.fft.fft(frames * self.window, axis=1)
        # abs() takes a square root that squaring immediately undoes
        power = spectra.real ** 2 + spectra.imag ** 2
        self.ffts += power.shape[0]

        if self._band_index is not None:
            self._watch_band(power)

        yield from self._accumulate(power)

    def _combine(self, frames, axis):
        """Reduce several frames of power to one, however this mode does it"""
        return frames.max(axis=axis) if self.mode == "peak" else frames.sum(axis=axis)

    def _fold(self, running, addition):
        """Fold a partial result into the one being built"""
        return np.maximum(running, addition) if self.mode == "peak" else running + addition

    def _accumulate(self, power):
        """Combine FFTs into spectra, yielding each one as it completes

        Reducing whole spectra at a time in NumPy, rather than a row at a time
        in Python, is what keeps this off the critical path: one transfer at
        20 MSPS is a couple of hundred FFTs."""
        rows = power.shape[0]
        taken = 0

        # Finish the spectrum that the previous block left part way through
        if self._count:
            take = min(self.average - self._count, rows)
            self._acc = self._fold(self._acc, self._combine(power[:take], 0))
            self._count += take
            taken = take
            if self._count >= self.average:
                yield self._finish()

        complete = (rows - taken) // self.average
        if complete:
            end = taken + complete * self.average
            groups = self._combine(
                power[taken:end].reshape(complete, self.average, -1), 1)
            taken = end
            for group in groups:
                self._acc, self._count = group, self.average
                yield self._finish()

        # Whatever is left starts the next spectrum
        if taken < rows:
            self._acc = self._combine(power[taken:], 0)
            self._count = rows - taken

    def _finish(self):
        """Convert what has accumulated to dB and start the next spectrum"""
        # 127 is full scale for signed 8 bit, so 0 dBFS is a full scale sine.
        # A mean is the sum of the frames behind it and has to be divided by
        # how many; a peak is one frame's power already.
        frames = self._count if self.mode == "mean" else 1
        scale = frames * (self._window_gain * 127.0) ** 2
        spectrum = 10.0 * np.log10(np.fft.fftshift(self._acc) / scale + 1e-20)
        remove_dc_spike(spectrum, self.dc_bins)
        self._acc, self._count = None, 0
        return spectrum

    def reset(self):
        """Drop anything part-way through"""
        self._tail = np.empty(0, dtype=np.float32)
        self._acc, self._count = None, 0
        self._band_carry = None
        self._band_out = []


def _window(name, size):
    """Analysis window by name"""
    windows = {
        "hann": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
        "bartlett": np.bartlett,
        "boxcar": np.ones,
    }
    if name not in windows:
        raise ValueError("unknown window {!r}, expected one of {}".format(
            name, ", ".join(sorted(windows))))
    return windows[name](size).astype(np.float32)

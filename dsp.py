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
    means leave both at zero."""
    if gain_db < 0:
        return 0, 0
    gain_db = min(gain_db, 102)
    lna = min(40, 8 * (int(gain_db) // 18))
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
    """Averages FFTs of incoming samples into finished spectra

    Sample blocks do not arrive aligned to anything, so leftover samples are
    carried into the next block rather than dropped: at 20 MSPS every discarded
    sample is measurement thrown away."""

    def __init__(self, fft_size, average, window="hann", dc_bins=2):
        if average < 1:
            raise ValueError("average must be at least 1")
        self.fft_size = fft_size
        self.average = average
        self.dc_bins = dc_bins
        self.window = _window(window, fft_size)
        # Coherent gain, so a full scale sine reads 0 dBFS whatever the window.
        # (Summing the squares instead would normalise for noise density, and
        # would put a full scale tone 25 dB high with a Hann window.)
        self._window_gain = float(np.sum(self.window))
        self._tail = np.empty(0, dtype=np.float32)
        self._sum = None
        self._count = 0
        self.ffts = 0

    def feed(self, samples):
        """Add raw interleaved I/Q and yield every spectrum it completes

        Yields power in dB, one array per completed average, lowest frequency
        first."""
        block = samples.astype(np.float32)
        if self._tail.size:
            block = np.concatenate((self._tail, block))

        pairs = block.size // 2
        usable = (pairs // self.fft_size) * self.fft_size
        if usable == 0:
            self._tail = block
            return

        self._tail = block[usable * 2:].copy()
        iq = block[:usable * 2]
        frames = (iq[0::2] + 1j * iq[1::2]).reshape(-1, self.fft_size) * self.window
        power = np.abs(np.fft.fft(frames, axis=1)) ** 2
        self.ffts += power.shape[0]

        for row in power:
            self._sum = row if self._sum is None else self._sum + row
            self._count += 1
            if self._count >= self.average:
                yield self._finish()

    def _finish(self):
        """Convert the running sum to dB and start the next average"""
        # 127 is full scale for signed 8 bit, so 0 dBFS is a full scale sine
        scale = self._count * (self._window_gain * 127.0) ** 2
        spectrum = 10.0 * np.log10(np.fft.fftshift(self._sum) / scale + 1e-20)
        remove_dc_spike(spectrum, self.dc_bins)
        self._sum, self._count = None, 0
        return spectrum

    def reset(self):
        """Drop anything part-way through"""
        self._tail = np.empty(0, dtype=np.float32)
        self._sum, self._count = None, 0


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

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


def offset_tune(start_freq, stop_freq, sample_rate, guard=0.0, usable=None):
    """Where to tune so a requested span misses the receiver's DC spike

    The spike sits at the centre of the tune because that is what the centre
    of the tune is: zero hertz at baseband, where the receiver's own DC offset
    lands. The only way to keep it out of a span is to put the whole span on
    one side of the centre, so this works when the span is less than half the
    sample rate and not otherwise — no offset helps a wider one, since half of
    it is always on the far side of the centre.

    When it fits, the span goes in the middle of one half of the passband.
    That leaves it the same clearance from the spike as from the anti-alias
    filter's roll-off at the edge, which is the most that both can be given at
    once. `guard` is the least clearance worth having, in hertz: below it the
    offset is not worth the half of the tune it spends.

    `usable` is how much of the tune the receiver really passes, in hertz -
    the baseband filter, which is narrower than the sample rate. Left out, the
    passband is taken to reach Nyquist, which is what a span calculation
    assumes and what no radio actually does.

    Returns the centre frequency to tune to, or None when the span cannot be
    moved clear and the spike has to be flattened where it falls instead."""
    span = abs(stop_freq - start_freq)
    middle = (start_freq + stop_freq) / 2.0
    # Half of what the receiver actually passes. The anti-alias filter is
    # narrower than the sample rate - a HackRF at 20 MSPS passes 15 MHz - so
    # taking Nyquist for the edge puts the top of a wide span in the roll-off,
    # where it is attenuated and reads as the band going quiet at one end.
    half = (sample_rate if usable is None else min(usable, sample_rate)) / 2.0
    clearance = (half - span) / 2.0
    # Zero clearance is not a fit: it puts the spike exactly on the span's edge,
    # which is inside it. A span of half the sample rate is therefore the first
    # one that cannot be helped, not the last one that can.
    if clearance <= 0.0 or clearance < guard:
        return None
    # Tuning below the span puts the span in the upper half, which also folds
    # the IQ image of everything in it into the lower half — the half that is
    # being thrown away regardless
    return middle - half / 2.0


#: The RF amp ahead of the receiver, when it is switched on
AMP_GAIN_DB = 14

#: What each analogue stage can be turned up to
LNA_MAX_DB = 40
VGA_MAX_DB = 62


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
    gain_db = min(gain_db, LNA_MAX_DB + VGA_MAX_DB)
    lna = min(LNA_MAX_DB, 8 * (int(gain_db) // 8))
    vga = min(VGA_MAX_DB, 2 * ((int(gain_db) - lna) // 2))
    return lna, vga


def stage_gains(gain_db=-1, lna=None, vga=None):
    """The two stage settings to use, from a total or from the stages themselves

    A stage given explicitly is used as it stands, rounded down to a step the
    radio actually has; whatever is left None comes from splitting the single
    gain figure with split_gain(). So a caller that only knows "40 dB" gets the
    LNA filled first, and one that has been told what each stage should be gets
    exactly that."""
    split_lna, split_vga = split_gain(gain_db)
    if lna is not None:
        split_lna = min(LNA_MAX_DB, 8 * (max(0, int(lna)) // 8))
    if vga is not None:
        split_vga = min(VGA_MAX_DB, 2 * (max(0, int(vga)) // 2))
    return split_lna, split_vga


def describe_gain(lna, vga, amp=False):
    """Say in words what the gain settings do to the radio, as lines of text

    One gain figure in an interface hides three stages that are not
    interchangeable, so "40 dB" can mean the LNA is already full and every
    further dB would be spent at baseband. Report the split, and say where
    anything left has to come from."""
    front = (AMP_GAIN_DB if amp else 0) + lna

    lines = ['RF amp {} + LNA {} of {} dB (IF) + VGA {} of {} dB (baseband) '
             '= {} dB in all'
             .format('on (+{} dB)'.format(AMP_GAIN_DB) if amp else 'off',
                     lna, LNA_MAX_DB, vga, VGA_MAX_DB, front + vga)]

    # What is still unspent, in the order worth spending it: the amp and the
    # LNA both decide what the receiver can hear, the VGA only makes the trace
    # bigger.
    headroom = []
    if not amp:
        headroom.append('the RF amp is off, and its {} dB sit in front of everything else'
                        .format(AMP_GAIN_DB))
    if lna < LNA_MAX_DB:
        headroom.append('{} dB of LNA is unspent, which is the gain that makes the '
                        'radio hear more'.format(LNA_MAX_DB - lna))
    elif vga < VGA_MAX_DB:
        # Baseband gain does not make the radio hear anything it could not
        # hear already — but at VGA 0 a quiet band can land so low in the
        # 8 bit ADC that quantisation, not the air, sets the floor. The test
        # is whether the floor follows the gain: raise it 6 dB, and if the
        # noise floor rises 6 dB there was nothing to gain.
        headroom.append('the LNA is full, so more gain only turns up the VGA, which '
                        'lifts the noise with the signal'
                        + (' — worth a try only if the trace sits so low that '
                           'the 8 bit ADC is what limits it' if vga == 0 else ''))
    else:
        headroom.append('every stage is at maximum')
    lines.append('  ' + '; '.join(headroom))
    return lines


#: How far a windowed DC offset reaches either side of the centre bin
#:
#: The receiver's DC offset is a constant added to I and Q, so in the FFT it
#: is a delta at bin zero and everything either side of it is the analysis
#: window's own shape. That makes the spike's width a property of the window
#: counted in *bins*, and not a bandwidth in hertz: a Hann window spreads it
#: over three bins whether those bins are 40 kHz or 1250 kHz wide.
#:
#: Measured as the last bin still within 30 dB of the centre, which is where
#: it stops being tellable from a signal. Hann is 6 dB down at one bin and 32
#: down at two, so one either side is the whole of it.
WINDOW_MAINLOBE_BINS = {
    "boxcar": 0,
    "hann": 1,
    "hamming": 1,
    "bartlett": 1,
    "blackman": 2,
}


def dc_spike_bins(window="hann"):
    """How many bins either side of the centre the DC spike actually reaches

    Flattening fewer leaves shoulders of the receiver's own carrier standing
    on either side of the hole. Flattening more throws away measurement for
    nothing, and at coarse bins it throws away a great deal: one bin too many
    either side is 2.5 MHz of a 20 MHz span at 1250 kHz bins, so a default
    that is generous at 40 kHz bins quietly eats an eighth of the band."""
    return WINDOW_MAINLOBE_BINS.get(window, 1)


def remove_dc_spike(spectrum, bins=2):
    """Flatten the receiver's own carrier at the centre of the band, in place

    Tuning to one frequency puts the radio's DC offset exactly in the middle of
    the spectrum, where it shows as a peak that is not on the air and is usually
    the strongest thing on screen. Interpolate across it from its neighbours.

    This is the fallback. offset_tune() moves the spike out of the requested
    span altogether and measures every bin that is shown; it needs half the
    tune to spend and so only works for a span under half the sample rate.
    Where that does not fit, these bins are guessed rather than measured, and
    the caller should say so on screen rather than draw a straight line and
    leave it looking like quiet.

    A 2D array is taken as one spectrum per row, so a whole block of them
    costs one call rather than one each."""
    if bins <= 0:
        return spectrum

    size = spectrum.shape[-1]
    centre = size // 2
    low, high = centre - bins, centre + bins
    if low - 1 < 0 or high + 1 >= size:
        return spectrum

    left, right = spectrum[..., low - 1], spectrum[..., high + 1]
    spectrum[..., low:high + 1] = np.linspace(left, right, high - low + 1, axis=-1)
    return spectrum


#: Candidate frame counts for noise_ceiling(), and the quantile ratio each
#: one produces. Built once: it is the same table for every call.
_CEILING_GRID = np.exp(np.linspace(0.0, np.log(4096.0), 200))
_CEILING_RATIO = (np.log1p(-0.5 ** (1.0 / _CEILING_GRID))
                  / np.log1p(-0.16 ** (1.0 / _CEILING_GRID)))


def noise_ceiling(body_db, count):
    """How high the noise on its own gets in `count` readings, in dB

    A trigger level has to clear the noise's own maximum, and that maximum is
    not a fixed number of decibels above the noise floor: it depends both on
    how many readings there are to get lucky in and on how many frames were
    reduced into each one. One frame of bin noise wobbles by about 6 dB from
    its median to its 16th percentile; the peak of 25 frames wobbles by 1.3,
    because taking the largest of 25 draws throws away most of the spread.
    A level set at a fixed multiple of that spread is therefore wrong by tens
    of decibels at one end of the range or the other.

    So measure the shape instead of assuming it. A reading is the peak of some
    number of exponentially distributed frames — the frames in the reading
    times the bins in the band, and it does not matter which — and that number
    fixes the ratio between any two of its quantiles. Read the ratio off the
    16th and the 50th, both of which are below anything a burst does to the
    trace, recover the frame count and the mean frame power from it, and put
    the ceiling where the distribution says one reading in `count` will reach.

    Checked against simulation for 1 to 125 frames a reading and 12,500 to
    312,500 readings: within about a decibel throughout. A burst in the window
    lifts it, since it lifts the median, but slowly — 4.7 dB at a duty cycle
    of 60%, which is far more than a radar dwell ever is, and still 15 dB
    under the burst itself."""
    low = 10.0 ** (float(np.percentile(body_db, 16.0)) / 10.0)
    mid = 10.0 ** (float(np.percentile(body_db, 50.0)) / 10.0)
    if not low > 0.0 or not mid >= low:
        return float(np.max(body_db))
    # The ratio falls monotonically with the frame count, so interpolate on a
    # reversed table
    frames = float(np.interp(mid / low, _CEILING_RATIO[::-1], _CEILING_GRID[::-1]))
    mean_frame = -low / np.log1p(-0.16 ** (1.0 / frames))
    reach = -mean_frame * np.log1p(-(1.0 - 1.0 / max(count, 2)) ** (1.0 / frames))
    return 10.0 * np.log10(max(reach, 1e-30))


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

    def __init__(self, fft_size, average, window="hann", dc_bins=None, mode="mean"):
        if average < 1:
            raise ValueError("average must be at least 1")
        if mode not in self.MODES:
            raise ValueError("unknown mode {!r}, expected one of {}".format(
                mode, ", ".join(self.MODES)))
        self.fft_size = fft_size
        self.average = average
        #: Bins either side of the centre to flatten. None means let the
        #: window decide, which is the only thing that knows the answer
        self.dc_bins = dc_spike_bins(window) if dc_bins is None else dc_bins
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
        self._band_start_frame = None
        # The magnitude tap: the same output queue, read from the samples
        # instead of the bins. None means it is not running; see set_magnitude
        self._mag_group = None
        self._mag_detector = "peak"
        self._mag_carry = None
        self._mag_start_pair = None
        #: I/Q pairs this accumulator has been fed, counted whether or not
        #: anything is tapping them, so a tap switched on mid-stream knows
        #: where it began
        self.pairs_seen = 0
        # What one frame's power has to be divided by for 0 dBFS to be a full
        # scale sine, the single frame case of the scale _finish() works out
        self._frame_scale = (self._window_gain * 127.0) ** 2
        # The same reference for a single sample, where there is no window and
        # no FFT: a full scale complex exponential has I*I + Q*Q = 127*127
        self._sample_scale = 127.0 ** 2

    #: How a band reading may be reduced to one number
    DETECTORS = ("peak", "mean", "total")

    def set_band(self, first, last, group=1, detector="peak", skip=()):
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
        signal legible when it is only a few dB out of the noise.

        "total" is a different question, and it is about the bins rather than
        the frames. Both of the above take the *loudest bin* in the band and
        throw the others away, which is right for a carrier sitting in one bin
        and wrong for a pulse spread across many: it keeps a fraction of the
        signal and pays a noise floor that rises with every bin added. "total"
        adds the band's bins up instead, so a pulse that fills the band adds
        coherently while the noise across the bins averages. The gain over
        "peak" goes as sqrt(n)*ln(n) — about 10 dB across sixteen bins, which
        is roughly what a radar detector in a Wi-Fi chipset is doing when it
        watches a whole channel. Frames are still combined by their peak,
        because a wideband pulse is usually a short one.

        `skip` names bins inside the band to leave out, numbered the way
        `first` and `last` are. A band that would be left with nothing keeps
        them all, since a reading of something is better than no reading."""
        first = min(max(int(first), 0), self.fft_size - 1)
        last = min(max(int(last), first + 1), self.fft_size)
        # fftshift moves the upper half of the spectrum to the front, so bin i
        # counted from the lowest frequency lives at (i + n/2) % n unshifted
        half = self.fft_size // 2
        if detector not in self.DETECTORS:
            raise ValueError("unknown detector {!r}, expected one of {}".format(
                detector, ", ".join(self.DETECTORS)))
        # One tap at a time: both write into _band_out, and a reader has no
        # way to tell a bin reading from a sample reading once they are mixed
        self.clear_magnitude()
        bins = np.arange(first, last)
        kept = bins[~np.isin(bins, np.asarray(skip, dtype=int))]
        if kept.size:
            bins = kept
        self._band_index = (bins + half) % self.fft_size
        self._band_group = max(1, int(group))
        self._band_detector = detector
        self._band_carry = None
        self._band_out = []
        self._band_start_frame = None

    def clear_band(self):
        """Stop watching a band"""
        self._band_index = None
        self._band_carry = None
        self._band_out = []
        self._band_start_frame = None

    #: How a magnitude reading may be reduced from the samples behind it.
    #: "total" is missing on purpose: it is a question about bins, and this
    #: tap has none
    MAGNITUDE_DETECTORS = ("peak", "mean")

    def set_magnitude(self, group=1, detector="peak"):
        """Watch the whole passband in the time domain, `group` samples a reading

        The band tap reduces each *frame* to the loudest of a few bins, so its
        step can never be shorter than one FFT frame - 0.8 us at 20 MSPS even
        with the 16 point FFT that is the shortest this library will build.
        Nothing about the radio requires that: samples arrive every 50 ns, and
        the frame is a property of the transform rather than of the signal.
        This reads I*I + Q*Q straight off the samples and skips the transform,
        which reaches one sample a reading.

        What it gives up is the whole point of the FFT. There are no bins
        here, so there is no filter: a reading is the power of everything
        inside the baseband filter at once. That is not an oversight to be
        fixed later, it is the uncertainty relation - a reading covering t
        seconds cannot separate frequencies closer than about 1/t, so 50 ns of
        time resolution and 20 MHz of bandwidth are the same statement. Against
        a band tap on one 1.25 MHz bin the noise floor rises by about 12 dB,
        and every other signal in the passband arrives with it.

        So this is the tap for the shape of a pulse in a band already known to
        be quiet, and the band tap is the one for finding a signal in a band
        that is not. `detector` combines the samples behind one reading:
        "peak" keeps a spike shorter than the group, "mean" smooths it."""
        if detector not in self.MAGNITUDE_DETECTORS:
            raise ValueError("unknown detector {!r}, expected one of {}".format(
                detector, ", ".join(self.MAGNITUDE_DETECTORS)))
        self.clear_band()
        self._mag_group = max(1, int(group))
        self._mag_detector = detector
        self._mag_carry = None
        self._mag_start_pair = None
        self._band_out = []

    def clear_magnitude(self):
        """Stop watching the passband"""
        self._mag_group = None
        self._mag_carry = None
        self._mag_start_pair = None

    @property
    def magnitude_group(self):
        """Samples behind one magnitude reading, or None if it is not running"""
        return self._mag_group

    @property
    def band_start_frame(self):
        """Frame of the stream the current band began at, None before it has

        A band can be pointed somewhere else long after the radio started, and
        its readings are counted from where it began rather than from the
        start of the stream. Without this the caller has no way to say when
        the first of them was measured, and stamping them from the stream
        start puts the whole trace as far into the past as the radio has been
        running."""
        if self._mag_group is not None:
            # Counted in samples, reported in frames, so that a caller
            # multiplying by the frame duration gets seconds from either tap
            # without having to know which one it is reading
            if self._mag_start_pair is None:
                return None
            return self._mag_start_pair / self.fft_size
        return self._band_start_frame

    def take_band(self):
        """Band readings finished since the last call, in dB, oldest first"""
        if not self._band_out:
            return None
        taken = np.concatenate(self._band_out) if len(self._band_out) > 1 else self._band_out[0]
        self._band_out = []
        return taken

    def _watch_magnitude(self, fresh):
        """Reduce this block's samples to magnitude readings

        Vectorised over the whole block for the reason _watch_band() is: at
        20 MSPS one transfer is 65536 samples, and touching them a row at a
        time in Python would cost more than the FFTs do."""
        if self._mag_start_pair is None:
            self._mag_start_pair = self.pairs_seen

        data = fresh if self._mag_carry is None else \
            np.concatenate((self._mag_carry, fresh))
        self._mag_carry = None

        group = self._mag_group
        complete = (data.size // 2) // group
        used = complete * group * 2
        if complete:
            iq = data[:used].reshape(-1, 2)
            power = iq[:, 0] * iq[:, 0] + iq[:, 1] * iq[:, 1]
            groups = power.reshape(complete, group)
            done = (groups.mean(axis=1) if self._mag_detector == "mean"
                    else groups.max(axis=1))
            self._band_out.append(
                10.0 * np.log10(done / self._sample_scale + 1e-20))

        rest = data[used:]
        if rest.size:
            # Samples that did not fill a reading start the next one rather
            # than being dropped. Carrying also keeps I and Q interleaved: a
            # block with an odd number of values would otherwise swap the two
            # for every reading after it, silently and forever.
            self._mag_carry = rest

    def _watch_band(self, power):
        """Reduce this block's frames to band readings

        Vectorised over the whole block, so the cost does not depend on how
        many readings come out of it: at 20 MSPS a transfer is a couple of
        hundred frames and the band is a handful of bins wide."""
        if self._band_start_frame is None:
            # This block is the first the band has seen: ffts has already
            # counted it, so the band begins where the block does
            self._band_start_frame = self.ffts - power.shape[0]

        band = power[:, self._band_index]
        # "total" adds the bins, everything else keeps the loudest one
        values = band.sum(axis=1) if self._band_detector == "total" else band.max(axis=1)
        if self._band_carry is not None:
            values = np.concatenate((self._band_carry, values))
            self._band_carry = None

        group = self._band_group
        complete = values.shape[0] // group
        if complete:
            frames = values[:complete * group].reshape(complete, group)
            # peak and total both keep the loudest frame; only mean smooths
            done = (frames.mean(axis=1) if self._band_detector == "mean"
                    else frames.max(axis=1))
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
        fresh = samples.astype(np.float32)
        # Before the tail is joined on, so that every sample is measured
        # exactly once: the tail is samples an earlier call already saw and is
        # holding back until they can fill a frame
        if self._mag_group is not None:
            self._watch_magnitude(fresh)
        self.pairs_seen += fresh.size // 2

        block = fresh
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
            # Converted to dB as one array rather than a row at a time. At a
            # short average a block finishes thousands of spectra, and the
            # shift, log and DC fill per row cost several times the FFTs
            yield from self._to_db(groups, self.average)

        # Whatever is left starts the next spectrum
        if taken < rows:
            self._acc = self._combine(power[taken:], 0)
            self._count = rows - taken

    def _finish(self):
        """Convert what has accumulated to dB and start the next spectrum"""
        spectrum = self._to_db(self._acc, self._count)
        self._acc, self._count = None, 0
        return spectrum

    def _to_db(self, power, count):
        """Spectra in dB, lowest frequency first, from `count` frames of power

        Works on one spectrum or on a row per spectrum alike."""
        # 127 is full scale for signed 8 bit, so 0 dBFS is a full scale sine.
        # A mean is the sum of the frames behind it and has to be divided by
        # how many; a peak is one frame's power already.
        frames = count if self.mode == "mean" else 1
        scale = frames * (self._window_gain * 127.0) ** 2
        spectra = 10.0 * np.log10(np.fft.fftshift(power, axes=-1) / scale + 1e-20)
        return remove_dc_spike(spectra, self.dc_bins)

    def reset(self):
        """Drop anything part-way through"""
        self._tail = np.empty(0, dtype=np.float32)
        self._acc, self._count = None, 0
        self._band_carry = None
        self._mag_carry = None
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

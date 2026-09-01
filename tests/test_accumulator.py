"""Turning samples into spectra, and reading a band at the frame rate.

The accumulator is where a measurement is actually made, so these check what
comes out means what it says: that a full scale sine reads 0 dBFS, that no
sample is thrown away between blocks, and that peak and mean differ in the way
that decides whether a short pulse survives at all.
"""

import numpy as np

from .. import dsp


def interleave(signal):
    """Complex samples as the radio sends them: interleaved signed 8 bit"""
    block = np.empty(2 * len(signal), dtype=np.int8)
    block[0::2] = np.clip(np.round(signal.real), -127, 127).astype(np.int8)
    block[1::2] = np.clip(np.round(signal.imag), -127, 127).astype(np.int8)
    return block


def tone(n, cycles_per_frame, fft_size, amplitude=127.0):
    """A complex sine sitting exactly in the middle of one bin"""
    t = np.arange(n)
    return amplitude * np.exp(2j * np.pi * cycles_per_frame * t / fft_size)


# -- what a reading means -------------------------------------------------

def test_a_full_scale_sine_reads_zero_dbfs():
    # Whatever the window. If this drifts, every power in the application is
    # wrong by a constant nobody will notice.
    fft = 256
    for window in ("hann", "hamming", "blackman", "boxcar"):
        acc = dsp.SpectrumAccumulator(fft, 4, window=window, dc_bins=0)
        spectrum = list(acc.feed(interleave(tone(fft * 4, 40, fft))))[0]
        assert abs(spectrum.max()) < 0.5, (window, spectrum.max())


def test_the_tone_lands_in_the_bin_it_was_asked_for():
    fft = 256
    acc = dsp.SpectrumAccumulator(fft, 4, dc_bins=0)
    spectrum = list(acc.feed(interleave(tone(fft * 4, 40, fft))))[0]
    assert int(np.argmax(spectrum)) == fft // 2 + 40


def test_a_quieter_tone_reads_lower_by_what_it_was_reduced_by():
    fft = 256
    out = []
    for amplitude in (127.0, 12.7):
        acc = dsp.SpectrumAccumulator(fft, 4, dc_bins=0)
        out.append(list(acc.feed(interleave(tone(fft * 4, 40, fft, amplitude))))[0].max())
    assert abs((out[0] - out[1]) - 20.0) < 0.5, out


# -- keeping every sample -------------------------------------------------

def test_samples_left_over_are_carried_into_the_next_block():
    # At 20 MSPS a discarded sample is measurement thrown away, and blocks do
    # not arrive aligned to the FFT length
    fft, average = 64, 2
    acc = dsp.SpectrumAccumulator(fft, average, dc_bins=0)
    signal = tone(fft * average * 5, 10, fft)
    whole = list(acc.feed(interleave(signal)))

    acc2 = dsp.SpectrumAccumulator(fft, average, dc_bins=0)
    pieces = []
    cut = 0
    for size in (37, 91, 5, 200, 1000):                 # deliberately ragged
        piece = signal[cut:cut + size]
        if len(piece):
            pieces.extend(acc2.feed(interleave(piece)))
        cut += len(piece)
    if cut < len(signal):
        pieces.extend(acc2.feed(interleave(signal[cut:])))
    assert len(pieces) == len(whole), (len(pieces), len(whole))
    for a, b in zip(whole, pieces):
        assert np.allclose(a, b, atol=1e-6)


def test_the_frame_count_is_what_was_asked_for():
    fft, average = 32, 7
    acc = dsp.SpectrumAccumulator(fft, average, dc_bins=0)
    made = list(acc.feed(interleave(tone(fft * average * 3, 5, fft))))
    assert len(made) == 3
    assert acc.ffts == fft * average * 3 // fft


def test_reset_drops_what_was_part_way_through():
    fft = 32
    acc = dsp.SpectrumAccumulator(fft, 4, dc_bins=0)
    list(acc.feed(interleave(tone(fft * 2, 5, fft))))   # half of one spectrum
    acc.reset()
    made = list(acc.feed(interleave(tone(fft * 4, 5, fft))))
    assert len(made) == 1, "the half-finished one should not have completed this"


# -- peak against mean ----------------------------------------------------

def test_peak_keeps_a_short_pulse_that_the_mean_spreads():
    # The reason the detector exists. One frame of signal in a 32 frame
    # average is 15 dB of the pulse thrown away; peak keeps it whole.
    fft, average = 64, 32
    quiet = np.zeros(fft * average, dtype=complex)
    pulse = quiet.copy()
    pulse[:fft] = tone(fft, 10, fft)                    # exactly one frame of it

    heights = {}
    for mode in ("mean", "peak"):
        acc = dsp.SpectrumAccumulator(fft, average, dc_bins=0, mode=mode)
        heights[mode] = list(acc.feed(interleave(pulse)))[0].max()
    assert heights["peak"] > heights["mean"] + 12.0, heights
    assert abs(heights["peak"]) < 0.5, "peak should read it at its own height"
    assert abs(heights["mean"] - (-10 * np.log10(average))) < 0.5


def test_an_unknown_mode_or_average_is_refused():
    for kwargs in ({"mode": "median"},):
        try:
            dsp.SpectrumAccumulator(64, 4, **kwargs)
        except ValueError:
            continue
        raise AssertionError("should not accept {}".format(kwargs))
    try:
        dsp.SpectrumAccumulator(64, 0)
    except ValueError:
        return
    raise AssertionError("an average of 0 should not be accepted")


def test_an_unknown_window_is_refused():
    try:
        dsp.SpectrumAccumulator(64, 4, window="triangular-ish")
    except ValueError:
        return
    raise AssertionError("an unknown window should not be accepted")


# -- the DC spike, through the whole path ---------------------------------

def test_the_dc_offset_is_flattened_by_default():
    fft = 256
    signal = tone(fft * 4, 40, fft, 40.0) + (20 + 20j)   # a tone and an offset
    acc = dsp.SpectrumAccumulator(fft, 4, window="hann")
    assert acc.dc_bins == 1, "the window should decide it"
    spectrum = list(acc.feed(interleave(signal)))[0]
    centre = fft // 2
    # interpolated across, so the middle three are a straight line
    middle = spectrum[centre - 1:centre + 2]
    assert np.allclose(np.diff(middle), np.diff(middle)[0], atol=0.01)
    # and the tone is untouched
    assert int(np.argmax(spectrum)) == centre + 40


# -- the band tap ---------------------------------------------------------

def test_the_band_reads_the_bins_it_was_given_lowest_first():
    fft = 64
    acc = dsp.SpectrumAccumulator(fft, 8, dc_bins=0)
    acc.set_band(fft // 2 + 10, fft // 2 + 11)          # one bin, 10 above centre
    acc.feed(interleave(tone(fft * 8, 10, fft)))
    list(acc.feed(interleave(tone(fft * 8, 10, fft))))
    readings = acc.take_band()
    assert readings is not None and len(readings)
    assert abs(readings.max()) < 0.5, "a full scale tone in the band reads 0 dBFS"


def test_a_band_somewhere_else_hears_nothing():
    fft = 64
    acc = dsp.SpectrumAccumulator(fft, 8, dc_bins=0)
    acc.set_band(fft // 2 + 20, fft // 2 + 21)          # not where the tone is
    list(acc.feed(interleave(tone(fft * 8, 10, fft))))
    readings = acc.take_band()
    assert readings.max() < -40.0, readings.max()


def test_one_reading_per_group_of_frames():
    fft, frames = 32, 60
    for group in (1, 5, 20):
        acc = dsp.SpectrumAccumulator(fft, 10, dc_bins=0)
        acc.set_band(fft // 2, fft // 2 + 1, group=group)
        list(acc.feed(interleave(tone(fft * frames, 5, fft))))
        readings = acc.take_band()
        assert len(readings) == frames // group, (group, len(readings))


def test_the_band_detector_decides_whether_a_pulse_survives():
    fft, group = 32, 16
    signal = np.zeros(fft * group, dtype=complex)
    signal[:fft] = tone(fft, 5, fft)                    # one frame in sixteen
    heights = {}
    for detector in ("peak", "mean"):
        acc = dsp.SpectrumAccumulator(fft, group, dc_bins=0)
        acc.set_band(fft // 2, fft // 2 + 1, group=group, detector=detector)
        list(acc.feed(interleave(signal)))
        heights[detector] = acc.take_band().max()
    assert heights["peak"] > heights["mean"] + 9.0, heights


def test_total_beats_peak_on_a_chirp_that_fills_the_band():
    """What "total" is for, and how much it is actually worth

    Peak keeps the loudest bin. For a chirp - a pulse-compression radar's
    waveform - every bin in the sweep holds about the same power, so the
    loudest bin is no better than any other and peak gains nothing from the
    band being wide: it measures the same level whether one bin is watched or
    thirty. Adding the bins instead lets the signal grow while the noise
    across them averages, which is worth a few decibels and no more. Measured
    here at about 3 dB with the band matched to the pulse, and it turns
    negative once the band is much wider than the signal, which is why this
    checks both.
    """
    fft, blocks, sig_bins = 64, 400, 16

    def margin(detector, watch):
        def run(amp):
            rng = np.random.default_rng(7)
            acc = dsp.SpectrumAccumulator(fft, 1, dc_bins=0)
            acc.set_band(fft // 2 - watch // 2, fft // 2 + max(1, watch // 2),
                         group=1, detector=detector)
            x = (rng.normal(0, 4, fft * blocks) + 1j * rng.normal(0, 4, fft * blocks))
            if amp:
                t = np.arange(fft)
                x[:fft] += amp * np.exp(1j * np.pi * (sig_bins / float(fft))
                                        * (t - fft / 2.0) ** 2 / fft)
            list(acc.feed(interleave(x)))
            return np.asarray(acc.take_band(), dtype=float)
        return run(30.0)[0] - run(0.0)[1:].max()

    matched = margin("total", 8) - margin("peak", 8)
    assert matched > 2.0, matched
    # and the other way round when the band is much wider than the pulse
    assert margin("total", 32) - margin("peak", 32) < matched, "wide band should not help"


def test_an_unknown_band_detector_is_still_refused_after_total_was_added():
    acc = dsp.SpectrumAccumulator(32, 4, dc_bins=0)
    assert "total" in acc.DETECTORS
    try:
        acc.set_band(10, 12, detector="sum")
    except ValueError:
        return
    raise AssertionError("'sum' is not a detector and should not be accepted")


def test_clearing_the_band_stops_the_readings():
    fft = 32
    acc = dsp.SpectrumAccumulator(fft, 4, dc_bins=0)
    acc.set_band(fft // 2, fft // 2 + 1)
    list(acc.feed(interleave(tone(fft * 8, 5, fft))))
    assert acc.take_band() is not None
    acc.clear_band()
    list(acc.feed(interleave(tone(fft * 8, 5, fft))))
    assert acc.take_band() is None


def test_the_band_says_which_frame_it_began_at():
    fft = 32
    acc = dsp.SpectrumAccumulator(fft, 4, dc_bins=0)
    assert acc.band_start_frame is None
    list(acc.feed(interleave(tone(fft * 8, 5, fft))))   # eight frames before it starts
    acc.set_band(fft // 2, fft // 2 + 1)
    list(acc.feed(interleave(tone(fft * 8, 5, fft))))
    assert acc.band_start_frame == 8, acc.band_start_frame


def test_an_unknown_band_detector_is_refused():
    acc = dsp.SpectrumAccumulator(32, 4)
    try:
        acc.set_band(16, 17, detector="quasi-peak")
    except ValueError:
        return
    raise AssertionError("an unknown detector should not be accepted")

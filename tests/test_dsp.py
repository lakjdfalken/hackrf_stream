"""The maths that decides what a measurement means.

These are the numbers that cost an evening each when they were wrong, so the
tests say what the right answer is and why, not just that it has not changed.
"""

import numpy as np

from .. import dsp


# -- sizing and axes ------------------------------------------------------

def test_fast_fft_size_is_a_power_of_two_near_the_request():
    for rate, bin_size, expected in ((20e6, 1250e3, 16), (20e6, 40e3, 512),
                                     (20e6, 625e3, 32), (8e6, 40e3, 256)):
        got = dsp.fast_fft_size(rate, bin_size)
        assert got == expected, (rate, bin_size, got, expected)
        assert got & (got - 1) == 0, "not a power of two: {}".format(got)


def test_fast_fft_size_refuses_a_nonsense_bin():
    for bad in (0, -1):
        try:
            dsp.fast_fft_size(20e6, bad)
        except ValueError:
            continue
        raise AssertionError("bin_size {} should not be accepted".format(bad))


def test_frequencies_are_ascending_and_centred():
    f = dsp.frequencies(2770e6, 20e6, 16)
    assert len(f) == 16
    assert np.all(np.diff(f) > 0), "must come back lowest first"
    assert np.isclose(np.diff(f)[0], 20e6 / 16)
    # An even-length FFT has no bin exactly at the centre: the centre falls on
    # the bin fftshift puts at n/2, which is where the DC spike lands
    assert np.isclose(f[16 // 2], 2770e6)


# -- the two analogue stages ----------------------------------------------

def test_split_gain_fills_the_lna_first():
    # The LNA is what decides what the radio can hear; spreading a modest
    # request across both stages throws away most of what asking for it buys
    assert dsp.split_gain(24) == (24, 0)
    assert dsp.split_gain(40) == (40, 0)
    assert dsp.split_gain(64) == (40, 24)
    assert dsp.split_gain(-1) == (0, 0)


def test_split_gain_keeps_to_steps_the_radio_has():
    for total in range(0, 103):
        lna, vga = dsp.split_gain(total)
        assert lna % 8 == 0 and 0 <= lna <= dsp.LNA_MAX_DB, (total, lna)
        assert vga % 2 == 0 and 0 <= vga <= dsp.VGA_MAX_DB, (total, vga)
        assert lna + vga <= total, (total, lna, vga)


def test_split_gain_clamps_at_the_top():
    assert dsp.split_gain(1000) == (dsp.LNA_MAX_DB, dsp.VGA_MAX_DB)


def test_stage_gains_takes_a_stage_given_explicitly():
    assert dsp.stage_gains(64) == (40, 24)
    assert dsp.stage_gains(64, lna=8) == (8, 24)
    assert dsp.stage_gains(64, vga=40) == (40, 40)
    # rounded down to a step that exists, not rejected
    assert dsp.stage_gains(-1, lna=30, vga=25) == (24, 24)
    assert dsp.stage_gains(-1, lna=-5, vga=-5) == (0, 0)


def test_describe_gain_says_where_more_gain_would_go():
    full = "\n".join(dsp.describe_gain(40, 24, amp=True))
    assert "40 of 40" in full and "24 of 62" in full
    assert "78" in full, "should total amp + LNA + VGA"
    assert "LNA is full" in full
    room = "\n".join(dsp.describe_gain(8, 0, amp=False))
    assert "LNA is full" not in room


# -- the DC spike ---------------------------------------------------------

def test_dc_spike_width_comes_from_the_window():
    # A DC offset is a delta at bin zero, so what is left either side is the
    # window's own shape counted in bins - not a bandwidth in hertz
    assert dsp.dc_spike_bins("hann") == 1
    assert dsp.dc_spike_bins("hamming") == 1
    assert dsp.dc_spike_bins("blackman") == 2
    assert dsp.dc_spike_bins("boxcar") == 0
    assert dsp.dc_spike_bins("something else") == 1, "an unknown window gets the usual"


def test_hann_really_does_leak_one_bin_and_not_two():
    # The measurement dc_spike_bins() is built on. If this changes, the
    # default flattening width is wrong.
    for n in (16, 512):
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(np.ones(n) * np.hanning(n))) ** 2)
        db = 10 * np.log10(spectrum / spectrum.max() + 1e-30)
        centre = n // 2
        assert -7.0 < db[centre + 1] < -5.0, "one bin out should be about 6 dB down"
        assert db[centre + 2] < -30.0, "two bins out should be gone"


def test_remove_dc_spike_interpolates_across_the_middle():
    spectrum = np.full(32, -80.0)
    spectrum[16] = 0.0                        # the receiver's own carrier
    spectrum[15] = spectrum[17] = -6.0        # and its leakage
    dsp.remove_dc_spike(spectrum, bins=1)
    assert np.allclose(spectrum[15:18], -80.0), "the spike should be gone"
    assert spectrum[14] == -80.0 and spectrum[18] == -80.0, "neighbours untouched"


def test_remove_dc_spike_leaves_it_alone_when_asked_to():
    spectrum = np.full(32, -80.0)
    spectrum[16] = 0.0
    before = spectrum.copy()
    dsp.remove_dc_spike(spectrum, bins=0)
    assert np.array_equal(spectrum, before)


def test_remove_dc_spike_will_not_run_off_the_ends():
    spectrum = np.zeros(4)
    dsp.remove_dc_spike(spectrum, bins=3)     # wider than the array has room for
    assert np.array_equal(spectrum, np.zeros(4))


# -- tuning off to one side -----------------------------------------------

def test_offset_tune_needs_the_span_to_fit_in_half_the_tune():
    rate = 20e6
    # The spike is at the centre of the tune, so the span has to sit entirely
    # on one side of it. Anything wider than half the rate cannot.
    assert dsp.offset_tune(2760e6, 2780e6, rate) is None      # 20 MHz, the whole tune
    assert dsp.offset_tune(2770e6, 2780e6, rate) is None      # 10 MHz, exactly half
    assert dsp.offset_tune(2775e6, 2780e6, rate) is not None  # 5 MHz, fits


def test_offset_tune_puts_the_span_clear_of_the_centre_and_the_edge():
    rate, low, high = 20e6, 2775e6, 2780e6
    centre = dsp.offset_tune(low, high, rate)
    assert centre is not None
    # the whole span is inside the tune
    assert centre - rate / 2 <= low and high <= centre + rate / 2
    # the spike is outside the span
    assert not low <= centre <= high
    # and the clearance either side is the same, which is the most both can get
    to_spike = low - centre
    to_edge = (centre + rate / 2) - high
    assert np.isclose(to_spike, to_edge), (to_spike, to_edge)
    assert np.isclose(to_spike, (rate / 2 - (high - low)) / 2)


def test_offset_tune_respects_the_guard():
    rate, low, high = 20e6, 2775e6, 2780e6
    clearance = (rate / 2 - (high - low)) / 2          # 2.5 MHz
    assert dsp.offset_tune(low, high, rate, guard=clearance - 1) is not None
    assert dsp.offset_tune(low, high, rate, guard=clearance + 1) is None


def test_offset_tune_does_not_care_which_way_round_the_span_is_given():
    assert dsp.offset_tune(2780e6, 2775e6, 20e6) == dsp.offset_tune(2775e6, 2780e6, 20e6)


# -- how high the noise gets ----------------------------------------------

def _readings(rng, frames, count):
    """`count` readings, each the peak of `frames` frames of bin noise"""
    return rng.exponential(1.0, (count, frames)).max(axis=1)


def test_noise_ceiling_tracks_the_real_maximum():
    # The whole point: the ceiling is not a fixed number of decibels above the
    # floor. It depends on how many readings there are and on how many frames
    # went into each, and a level set at a multiple of the spread is wrong by
    # tens of decibels at one end of the range.
    rng = np.random.default_rng(4)
    for frames in (1, 2, 5, 25, 125):
        for count in (12500, 200000):
            r = 10 * np.log10(_readings(rng, frames, count))
            body = r[:: max(1, len(r) // 4096)]
            got = dsp.noise_ceiling(body, count)
            real = float(np.max(r))
            assert abs(got - real) < 2.0, (frames, count, got, real)


def test_noise_ceiling_is_not_fooled_by_a_burst_in_the_window():
    # It reads the 16th and 50th percentiles, both of which sit below anything
    # a burst does to the trace, so a high duty cycle lifts it only slowly
    rng = np.random.default_rng(5)
    count = 200000
    clean = _readings(rng, 1, count)
    quiet = dsp.noise_ceiling(10 * np.log10(clean[::48]), count)
    for duty in (0.05, 0.2, 0.4):
        loud = clean.copy()
        loud[: int(count * duty)] *= 100.0             # a burst 20 dB up
        rng.shuffle(loud)
        got = dsp.noise_ceiling(10 * np.log10(loud[::48]), count)
        assert got - quiet < 5.0, (duty, got, quiet)
        assert got < 10 * np.log10(loud.max()) - 10.0, "must stay well under the burst"


def test_noise_ceiling_rises_with_the_number_of_readings():
    rng = np.random.default_rng(6)
    body = 10 * np.log10(_readings(rng, 1, 4096))
    ceilings = [dsp.noise_ceiling(body, n) for n in (1000, 10000, 100000, 1000000)]
    assert ceilings == sorted(ceilings), ceilings
    # but slowly: an extreme value grows like the log of the log
    assert ceilings[-1] - ceilings[0] < 6.0, ceilings


def test_noise_ceiling_survives_a_degenerate_window():
    flat = np.full(256, -40.0)                 # every reading identical
    assert np.isfinite(dsp.noise_ceiling(flat, 1000))

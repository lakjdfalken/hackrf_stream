"""Fast spectra from a HackRF held on one frequency.

hackrf_sweep is limited by retuning to about 405 tuning steps a second. Staying
on one tune instead removes that limit and uses the whole 20 MSPS stream, which
is worth roughly 1500 averaged spectra a second over a 20 MHz band.

    from hackrf_stream import SpectrumSource

    def show(frequencies, powers_db, timestamp):
        print(powers_db.max())

    with SpectrumSource(center_freq=128e6, bin_size=40e3) as source:
        source.start(show)
        ...
        source.stop()

Requires libhackrf, the same system library hackrf_sweep uses. Nothing here
bundles it.
"""

from ._libhackrf import HackRFError
from .dsp import (AMP_GAIN_DB, LNA_MAX_DB, VGA_MAX_DB, WINDOW_MAINLOBE_BINS,
                  SpectrumAccumulator, dc_spike_bins, describe_gain,
                  fast_fft_size, frequencies, offset_tune, remove_dc_spike,
                  split_gain, stage_gains)
from .source import (BAND_BACKLOG, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE,
                     SpectrumSource, devices, library_version)

__version__ = "0.1.0"
__all__ = [
    "HackRFError", "SpectrumAccumulator", "SpectrumSource",
    "AMP_GAIN_DB", "BAND_BACKLOG", "LNA_MAX_DB", "MAX_SAMPLE_RATE", "MIN_SAMPLE_RATE",
    "VGA_MAX_DB", "WINDOW_MAINLOBE_BINS",
    "dc_spike_bins", "describe_gain", "devices", "fast_fft_size", "frequencies",
    "library_version", "offset_tune", "remove_dc_spike", "split_gain", "stage_gains",
]

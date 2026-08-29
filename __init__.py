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
from .dsp import (SpectrumAccumulator, fast_fft_size, frequencies,
                  remove_dc_spike, split_gain)
from .source import MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, SpectrumSource, library_version

__version__ = "0.1.0"
__all__ = [
    "HackRFError", "SpectrumAccumulator", "SpectrumSource",
    "MAX_SAMPLE_RATE", "MIN_SAMPLE_RATE",
    "fast_fft_size", "frequencies", "library_version", "remove_dc_spike",
    "split_gain",
]

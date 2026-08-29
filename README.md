# hackrf_stream

Fast power spectra from a HackRF held on one frequency.

`hackrf_sweep` retunes as it goes, and the retune is what limits it: about
**405 tuning steps per second** whatever the span or the bin size. That is
8 GHz/s of coverage, but only 6.6 MB/s of the 40 MB/s the USB link carries —
it spends most of its time changing frequency rather than measuring.

Staying on one frequency removes the retune. Measured on a HackRF One:

| | hackrf_sweep | hackrf_stream |
| --- | --- | --- |
| spectra per second | 405 | **1501** |
| stream used | 16% | **100%** |
| FFTs per spectrum | 1 | 26 |

Every sample is used, so the extra rate comes with a *better* noise floor
rather than a worse one.

The trade is span: one tune only sees one sample rate of spectrum, so at most
20 MHz. Use `hackrf_sweep` for anything wider.

## Use

```python
from hackrf_stream import SpectrumSource

def show(frequencies, powers_db, timestamp):
    print("%.1f dBFS" % powers_db.max())

with SpectrumSource(center_freq=128e6, bin_size=40e3, average=26) as source:
    source.start(show)
    ...
    source.stop()
```

The callback runs on libhackrf's own receive thread. Keep it short: while it
is running the radio has nowhere to put samples and starts dropping them. Hand
the spectrum off and do the work elsewhere.

Powers are dBFS — a full scale sine reads 0 dB, whichever window is chosen.

The receiver's own carrier lands in the middle of the band when tuned this way,
so the centre bins are interpolated across rather than shown as a peak that is
not on the air. Set `dc_bins=0` to see it.

## Requirements

- Python 3.9+
- numpy
- libhackrf, the same system library `hackrf_sweep` uses
  (`brew install hackrf`, `apt install libhackrf0`)

Nothing is bundled: libhackrf is loaded from wherever the platform put it.

## Licence

MIT — see LICENSE.

libhackrf itself is GPL-2.0-or-later and is *not* distributed with this
package; it has to be installed separately.

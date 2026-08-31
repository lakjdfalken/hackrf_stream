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

## Testing

```
python -m hackrf_stream.tests
```

No radio, no libhackrf and no test runner needed — they check the maths that
decides what a measurement means, and they are shipped in the wheel so that a
copy can always be checked where it is installed. pytest finds them too.

## Requirements

- Python 3.9+
- numpy
- libhackrf, the same system library `hackrf_sweep` uses
  (`brew install hackrf`, `apt install libhackrf0`)

Nothing is bundled: libhackrf is loaded from wherever the platform put it.

## Exporting this package

It is developed inside [QSpectrumAnalyzer](https://github.com/lakjdfalken/qspectrumanalyzer)
and exported from there, so this repository is a product rather than a place
to work: **commit to the analyser, not here.** Anything committed here is lost
at the next export.

From a clone of the analyser:

```sh
git branch -D export-hackrf-stream 2>/dev/null   # subtree split will not reuse it
git subtree split --prefix=hackrf_stream -b export-hackrf-stream
git clone -b export-hackrf-stream --single-branch . ../hackrf_stream
cd ../hackrf_stream
mkdir hackrf_stream
git mv __init__.py _libhackrf.py dsp.py source.py tests hackrf_stream/
git commit -m "Put the package in its own directory"
python -m hackrf_stream.tests
git remote add origin https://github.com/lakjdfalken/hackrf_stream.git
```

The split keeps every commit that touched the package and nothing else, and
puts `pyproject.toml`, `README.md` and `LICENSE` at the root where a build
expects them; the one commit after it moves the modules into the package
directory. `git subtree split` is deterministic, so a later export produces the
same history again with the new commits on the end.

## Licence

MIT — see LICENSE.

libhackrf itself is GPL-2.0-or-later and is *not* distributed with this
package; it has to be installed separately.

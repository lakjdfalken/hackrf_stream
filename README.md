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

## Install

```sh
pip install git+https://github.com/lakjdfalken/hackrf_stream.git@v0.2.0
```

libhackrf has to be installed separately; see Requirements.

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

The callback runs on the source's own FFT thread, not on libhackrf's: the
receive callback only copies each transfer onto a queue. Keep it short all the
same — spectra arrive about 1500 times a second with 512 bins averaging 26, and a slow
callback backs the queue up until transfers are dropped. Hand the spectrum off
and do the work elsewhere.

The receive callback is Python, so it needs the interpreter lock, and libhackrf
holds only about 20 ms of transfers while it waits. Any thread in the same
process holding the lock longer than that - a long numpy call, a repaint in a
GUI toolkit that does not release it - loses the samples that arrive
meanwhile, below the point where `dropped_transfers` can count them. Two
figures show it: `statistics()["stream_fraction"]` falls short of 1, and
`take_gap_peak()` reports a wait between transfers far past their period
(6.55 ms at 20 MSPS). A short stream with ordinary gaps was lost in USB or the
radio instead.

Powers are dBFS — a full scale sine reads 0 dB, whichever window is chosen.

The receiver's own carrier lands in the middle of the band when tuned this way,
so the centre bins are interpolated across rather than shown as a peak that is
not on the air. Set `dc_bins=0` to see it.

## Testing

```
python3 -m hackrf_stream.tests
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

## Development

This repository is where the library is developed: issues and pull requests
belong here. [QSpectrumAnalyzer](https://github.com/lakjdfalken/qspectrumanalyzer),
where it began, is one program using it, and depends on its released
versions like any other.

To work on it beside a program that uses it, install the clone editable into
that program's environment, so that changes here are picked up without
reinstalling:

```sh
pip install -e path/to/hackrf_stream
```

A release is a version bump in both `pyproject.toml` and `__init__.py` — a
test checks that they agree — and a tag:

```sh
git tag -a v0.3.0 -m "hackrf_stream 0.3.0"
git push origin main v0.3.0
```

## Licence

MIT — see LICENSE.

libhackrf itself is GPL-2.0-or-later and is *not* distributed with this
package; it has to be installed separately.

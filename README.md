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

Powers are dBFS — a full scale sine reads 0 dB, whichever window is chosen.

The receiver's own carrier lands in the middle of the band when tuned this way,
so the centre bins are interpolated across rather than shown as a peak that is
not on the air. Set `dc_bins=0` to see it, or keep it out of the span
altogether with `offset_tune()`, below.

`devices()` lists the HackRFs attached, and `serial=` picks one.

## Watching one band in time

A spectrum is an average, so a burst shorter than it is spread across all of
it. The band tap reads the FFT frames themselves instead - one bin of an FFT is
a filter, so following a few bins frame by frame is what a spectrum analyser
does in zero span:

```python
source.set_band(1089e6, 1091e6, resolution=None, detector="peak")
...
readings = source.take_band_power()   # (N, 2): seconds since stream_start, dB
```

A frame is `fft_size / sample_rate` long - 25.6 us at 40 kHz bins - and with
`resolution=None` there is one reading per frame. A longer `resolution` groups
frames, combined by `detector`: `"peak"` keeps a pulse shorter than the
reading at its own height, `"mean"` smooths, and `"total"` adds the band's bins
rather than keeping the loudest, which suits a pulse that fills the band.

The receiver's own carrier sits at the centre of every tune, and the tap would
keep it as the loudest bin in every reading. So a band with the centre in it
is read without the spike's bins (`dc_band` says where they are), and
`band_skips_dc` says when that has happened. A band that is nothing but the
spike still reads it. `skip_dc=False` reads the raw bins regardless.

Ask for a `resolution` shorter than one frame and the tap reads I*I + Q*Q off
the samples instead, down to one sample (50 ns at 20 MSPS). That has no bins,
so it measures the whole passband, not the band asked for; `band_magnitude`
says when that is what is running. Readings are stamped from `stream_start`
rather than from 1970, so their spacing survives being written down.
`set_band(None, None)` stops the tap.

## Peak instead of mean

`mode="peak"` keeps the loudest of the frames behind each spectrum instead of
averaging them. Averaging lowers the noise floor for a signal that is always
there; for one that is not, it spreads the burst over the frames it missed and
buries it. With `"peak"` a long `average` becomes a wider net rather than a
deeper hole.

## Gain

`gain=40` is split across the two stages with the LNA filled first, because
it is the stage that decides what the radio can hear; `lna=` and `vga=` set
them directly, and `amp=True` adds the 14 dB RF amplifier in front.
`set_gain()` changes them while running. `describe_gain(lna, vga, amp)` says in
words what a setting adds up to and where any more gain would come from.

## Keeping the DC spike out

```python
from hackrf_stream import baseband_filter_bw, offset_tune
centre = offset_tune(start_freq, stop_freq, sample_rate,
                     usable=baseband_filter_bw(0.75 * sample_rate))
```

returns a centre frequency that puts the whole span on one side of the
receiver's own carrier, so that every bin shown is measured. `usable` is what
the receiver actually passes - the source sets its baseband filter to three
quarters of the sample rate, 15 MHz at 20 MSPS - and without it the span is
fitted to the full sample rate and can land on the filter's roll-off, where it
reads as the band going quiet at one end. It returns None when the span is too
wide for any offset to clear the spike, and the centre bins are then
interpolated instead.

## Is anything being lost?

`statistics()` reports the rate, how much of the stream arrived
(`stream_fraction`), transfers dropped because the FFT thread fell behind
(`dropped_transfers`), and how busy that thread is. `take_queue_peak()` and
`take_gap_peak()` give the deepest the queue got and the longest wait between
transfers since they were last asked.

The receive callback is Python, so it needs the interpreter lock, and libhackrf
holds only about 20 ms of transfers while it waits. Any thread in the same
process holding the lock longer than that - a long numpy call, a repaint in a
GUI toolkit that does not release it - loses the samples that arrive
meanwhile, below the point where `dropped_transfers` can count them. Two
figures show it: `statistics()["stream_fraction"]` falls short of 1, and
`take_gap_peak()` reports a wait between transfers far past their period
(6.55 ms at 20 MSPS). A short stream with ordinary gaps was lost in USB or the
radio instead.

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

"""Tests for hackrf_stream.

Everything here runs without a radio and without libhackrf: the DSP is plain
numpy and the parts that talk to hardware are not exercised. Run them with
pytest, or directly with `python -m hackrf_stream.tests` when pytest is not
installed.
"""

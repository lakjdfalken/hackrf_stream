"""Run every test in this package without pytest.

The library is meant to be usable by anyone with numpy and a HackRF, and
asking them to install a test runner before they can check that their copy
works is a barrier for no reason."""

import sys
import traceback


def main():
    from . import test_dsp, test_accumulator
    failed = []
    for module in (test_dsp, test_accumulator):
        for name in sorted(n for n in dir(module) if n.startswith("test_")):
            try:
                getattr(module, name)()
                print("  ok    {}.{}".format(module.__name__.rsplit(".", 1)[-1], name))
            except Exception:                    # noqa: BLE001 - report them all
                failed.append((module.__name__, name, traceback.format_exc()))
                print("  FAIL  {}.{}".format(module.__name__.rsplit(".", 1)[-1], name))
    for where, name, why in failed:
        print("\n{}.{}\n{}".format(where, name, why))
    print("\n{} failed".format(len(failed)) if failed else "\nall passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

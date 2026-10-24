#!/usr/bin/env python3
"""Compatibility entrypoint for older FROSS Brain scripts.

The current Python implementation and test suite live in fross_brain_v03.py.
Run: python fross_brain_v03.py
This compatibility file forwards imports and test execution to version 0.3.
"""
from fross_brain_v03 import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("fross_brain_v03", run_name="__main__")

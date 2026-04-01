#!/usr/bin/env python
"""
Setup script for NICERsoft.

This setup.py exists primarily to handle installation of executable scripts
from the scripts/ directory. The main configuration is in pyproject.toml.
"""

from setuptools import setup
from glob import glob

# Get all Python scripts from the scripts directory
scripts = glob("scripts/*.py")

# Filter out __init__.py and any backup files
scripts = [s for s in scripts if not s.endswith("__init__.py") and not s.endswith("~")]

setup(
    scripts=scripts,
)

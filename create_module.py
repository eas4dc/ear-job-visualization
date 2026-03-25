#!/usr/bin/env python3
"""
Generate a Lmod modulefile for ear-job-visualizer.

Usage:
    python create_modulefile.py [OPTIONS]

Options:
    --prefix PATH       Installation prefix
    --version VERSION   Package version (default: read from installed package metadata)
    --output FILE       Output path (default: ear-job-visualizer/<version>.lua)
    --python-version    Override Python version (default: sys.version_info)
"""

import sys
import os
import argparse
from pathlib import Path


def get_package_version():
    """Read version from installed package metadata (preferred) or pyproject fallback."""
    try:
        from importlib.metadata import version
        return version("ear-job-visualization")
    except Exception:
        return None


def get_python_version(override=None):
    if override:
        return override
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def render_modulefile(prefix: str, python_version: str, ejv_version: str) -> str:
    return f"""\
-- -*- lua -*-
-- Lmod modulefile for ear-job-visualizer {ejv_version}

whatis("Name:        ear-job-visualizer")
whatis("Version:     {ejv_version}")
whatis("Description: Visualisation tool for performance metrics collected by EAR.")

local prefix      = "{prefix}"
local python_ver  = "{python_version}"

prepend_path("PATH",       pathJoin(prefix, "bin"))
prepend_path("PYTHONPATH", pathJoin(prefix, "lib", "python" .. python_ver, "site-packages"))
"""


def main():
    parser = argparse.ArgumentParser(description="Generate Lmod modulefile for ear-job-visualizer.")
    parser.add_argument("--prefix",         default=None)
    parser.add_argument("--version",        default=None)
    parser.add_argument("--output",         default=None)
    parser.add_argument("--python-version", default=None, dest="python_version")
    args = parser.parse_args()

    # Fail explicitly rather than produce a broken module
    if not args.prefix:
        sys.exit(
            "Error: installation prefix required.\n"
            "Pass --prefix."
        )

    ejv_version = args.version or get_package_version()
    if not ejv_version:
        sys.exit(
            "Error: could not determine package version.\n"
            "Pass --version or install the package first."
        )

    python_version = get_python_version(args.python_version)

    output = Path(args.output) if args.output else Path(f"ear-job-visualizer/{ejv_version}.lua")
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(render_modulefile(args.prefix, python_version, ejv_version))
    print(f"Modulefile written to: {output}")


if __name__ == "__main__":
    main()
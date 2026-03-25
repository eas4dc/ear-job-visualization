"""
Normalize old-format EAR CSV files (pre-EAR6, missing APPID) to the format
expected by the current version of ear_analytics_core / ear-job-visualizer.

Installed as the ``ear-normalize-csv`` command.
"""

import sys
import os
import argparse

import pandas as pd


# ---------------------------------------------------------------------------
# Column rename maps (old name → new name)
# ---------------------------------------------------------------------------
_APPS_RENAME = {
    'USER_ACC':   'ACCOUNTID',
    'CPU-GFLOPS': 'CPU_GFLOPS',
    'DP_256':     'DPOPS_256',
}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _is_old_format(df: pd.DataFrame) -> bool:
    return 'APPID' not in df.columns


def normalize_loops(df: pd.DataFrame) -> pd.DataFrame:
    """Insert APPID = 1 before NODENAME."""
    df.insert(df.columns.get_loc('NODENAME'), 'APPID', 1)
    return df


def _add_job_start_end_times(df: pd.DataFrame, df_loops: pd.DataFrame) -> pd.DataFrame:
    """
    Derive JOB_EARL_START_TIME and JOB_EARL_END_TIME (Unix timestamps) from the loops
    TIMESTAMP and ELAPSED columns and merge them into the apps DataFrame.

    JOB_EARL_START_TIME = first_TIMESTAMP - first_ELAPSED
        Places the window start before the first measurement, matching the
        convention used by the new EAR format.  This is required for the
        library's bfill() call to fill the full time range: with the data
        point at the end of the window, bfill fills every earlier timestamp.

    JOB_EARL_END_TIME = last_TIMESTAMP
        Ends the window at the last recorded measurement.  Extending by
        ELAPSED would add trailing NaN rows that bfill cannot fill.
    """
    grp = df_loops.sort_values('TIMESTAMP').groupby(['JOBID', 'STEPID', 'NODENAME'])
    timing = pd.concat([
        (grp['TIMESTAMP'].first() - grp['ELAPSED'].first()).rename('JOB_EARL_START_TIME'),
        grp['TIMESTAMP'].last().rename('JOB_EARL_END_TIME'),
    ], axis=1).reset_index()
    return df.merge(timing, on=['JOBID', 'STEPID', 'NODENAME'], how='left')


def normalize_apps(df: pd.DataFrame, df_loops: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, add APPID and derive JOB_EARL_START/END_TIME."""
    df = df.rename(columns=_APPS_RENAME).pipe(_add_job_start_end_times, df_loops)
    df.insert(df.columns.get_loc('STEPID') + 1, 'APPID', 1)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description='Normalize old-format EAR CSV files for ear-job-visualizer.')
    p.add_argument('--apps-file', '-a', required=True, metavar='PATH',
                   help='Path to the apps CSV file.')
    p.add_argument('--loops-file', '-l', required=True, metavar='PATH',
                   help='Path to the loops CSV file.')
    p.add_argument('--output-dir', '-o', default='.', metavar='DIR',
                   help='Directory for output files (default: current directory).')
    args = p.parse_args()

    df_loops = pd.read_csv(args.loops_file, sep=';')
    df_apps  = pd.read_csv(args.apps_file,  sep=';')

    if not _is_old_format(df_loops) and not _is_old_format(df_apps):
        print('Both files are already in new format — nothing to do.', file=sys.stderr)
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    in_loops = os.path.basename(args.loops_file)
    in_apps  = os.path.basename(args.apps_file)
    out_loops = "normalized_loops_" + in_loops
    out_apps  = "normalized_apps_" + in_apps
    out_loops = os.path.join(args.output_dir, out_loops)
    out_apps  = os.path.join(args.output_dir, out_apps)

    if _is_old_format(df_loops):
        df_loops = normalize_loops(df_loops)
        print(f'[normalize] loops → {out_loops}', file=sys.stderr)
    else:
        print('[normalize] loops already in new format, copying unchanged.', file=sys.stderr)

    if _is_old_format(df_apps):
        df_apps = normalize_apps(df_apps, df_loops)
        print(f'[normalize] apps  → {out_apps}', file=sys.stderr)
    else:
        print('[normalize] apps already in new format, copying unchanged.', file=sys.stderr)

    df_loops.to_csv(out_loops, sep=';', index=False)
    df_apps.to_csv(out_apps,  sep=';', index=False)

    print(f'\near-job-visualizer --apps-file {out_apps} --loops-file {out_loops}')

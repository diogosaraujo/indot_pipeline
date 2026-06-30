"""08b_smooth_rp_counts.py

Smooth (continuous) return-period version of the event-overlap confusion matrix.

This is a thin wrapper over 08_trigger_analysis.py.  It runs the EXACT same
event-overlap scoring but, instead of the 10 discrete Atlas 14 precipitation
return periods, it fits a GEV to each station/duration's 10 published DDF
quantiles and resamples the depth threshold on a dense 40-point log-spaced
return-period grid (1..1000 yr).  Every count is still an integer recomputed at
each return period — only the depth THRESHOLD is interpolated, on a published
parametric DDF family (see fit_gev_quantiles in 08).  The result is a smooth
POD / FAR / CSI / precision-recall curve along the precipitation return-period
axis (citations and method in 08_trigger_analysis.py).

Output:
    s3://<bucket>/<prefix>analysis/event_confusion_matrix_smooth.parquet

Usage:
    python scripts/08b_smooth_rp_counts.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# Import the digit-prefixed module 08_trigger_analysis.py by file path.
_spec = importlib.util.spec_from_file_location(
    "trigger_analysis_08", Path(__file__).with_name("08_trigger_analysis.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def main() -> None:
    _mod.main(smooth=True, output_key=_mod.SMOOTH_OUTPUT_KEY)


if __name__ == "__main__":
    main()

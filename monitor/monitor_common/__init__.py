"""monitor_common — shared runtime for the INDOT bridge flood-alert monitor.

This package is baked into the Lambda container image and is deliberately
self-contained (it does not import the numbered pipeline scripts) so the image
stays lean.  The heavy research machinery (LP3, Atlas-14 fetch, Tc) lives only
in ``monitor/precompute/`` and runs once on EC2.
"""

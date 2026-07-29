"""MRMS GRIB2 helpers — copied verbatim (behaviour-preserving) from
scripts/utils.py so the Lambda image needs only monitor_common.

Keep in sync with scripts/utils.py if the grid conventions ever change.
"""
from __future__ import annotations

import gzip


def decompress_gz(data: bytes) -> bytes:
    return gzip.decompress(data)


def open_mrms_grib(path):
    """Open an MRMS GRIB2 file with the pipeline's cfgrib options."""
    import xarray as xr
    return xr.open_dataset(
        path,
        engine="cfgrib",
        decode_timedelta=False,
        backend_kwargs={"indexpath": ""},
    )


def canonicalize_mrms_grid(da):
    """Return (data_2d, lats_desc, lons_asc_-180_180)."""
    import numpy as np
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"
    lats = da[lat_name].values.copy()
    lons = da[lon_name].values.copy()
    arr = da.values

    if lons.max() > 180.0:
        lons = np.where(lons > 180, lons - 360.0, lons)
        order_lon = np.argsort(lons)
        if not np.array_equal(order_lon, np.arange(lons.size)):
            lons = lons[order_lon]
            arr = arr[..., order_lon]

    if lats[0] < lats[-1]:
        lats = lats[::-1]
        arr = arr[::-1, :]

    return arr, lats, lons


def apply_units(arr, kind: str, units: str):
    """Negatives -> NaN (missing sentinel); QPE mm -> inches when units=='in'."""
    import numpy as np
    arr = np.where(arr < 0, np.nan, arr)
    if kind == "qpe" and units == "in":
        arr = arr / 25.4
    return arr

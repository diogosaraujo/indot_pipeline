"""Thin S3 helpers for the monitor.

Own output bucket -> authenticated boto3 (ambient Lambda role).
Public NOAA buckets -> anonymous s3fs (mirrors scripts/utils.py + script 05/10).
"""
from __future__ import annotations

import io
from functools import lru_cache

import boto3
import pyarrow as pa
import pyarrow.parquet as pq

from . import config


@lru_cache(maxsize=1)
def s3():
    return boto3.client("s3", region_name=config.REGION)


@lru_cache(maxsize=1)
def anon_fs():
    import s3fs
    return s3fs.S3FileSystem(anon=True)


def read_parquet(bucket: str, key: str, columns=None):
    obj = s3().get_object(Bucket=bucket, Key=key)
    return pq.read_table(io.BytesIO(obj["Body"].read()), columns=columns).to_pandas()


def write_parquet(df, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="zstd")
    s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def write_bytes(data: bytes, bucket: str, key: str,
                content_type: str = "application/octet-stream") -> None:
    s3().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def object_exists(bucket: str, key: str) -> bool:
    resp = s3().list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return any(o["Key"] == key for o in resp.get("Contents", []))


def list_keys(bucket: str, prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3().list_objects_v2(**kw)
        keys.extend(o["Key"] for o in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            return keys
        token = resp["NextContinuationToken"]


def delete_keys(bucket: str, keys: list[str]) -> None:
    for i in range(0, len(keys), 1000):
        batch = [{"Key": k} for k in keys[i:i + 1000]]
        if batch:
            s3().delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})

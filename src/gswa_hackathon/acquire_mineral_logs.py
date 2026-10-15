"""Acquire validated GSWA mineral logs without replacing existing files."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http.client import HTTPException
from pathlib import Path
from threading import Lock
from urllib.error import URLError
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd

TRUSTED_HOST = "dasc.dmirs.wa.gov.au"
TRUSTED_BLOB_HOST = "warsydprdstadasc.blob.core.windows.net"
TRUSTED_BLOB_PATH = "/downloads/Hylogger/Mineral_Logs/"
REQUEST_HEADERS = {"Accept": "text/csv", "User-Agent": "GSWA-Hackathon/0.1"}
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
SOURCE_COLUMNS = ["Dataset_ID", "CollarId", "Project", "source_url"]
MANIFEST_COLUMNS = [
    "local_path",
    "status",
    "sha256",
    "file_size_bytes",
    "error",
    "validated_at_utc",
]
DEPTH_COLUMNS = ("Depth_from_(m)", "Depth_to_(m)")


def _trusted_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != TRUSTED_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 80 if parsed.scheme == "http" else 443}
    ):
        raise ValueError(f"Only HTTP(S) URLs on {TRUSTED_HOST} are allowed")
    return urlunsplit(("https", TRUSTED_HOST, parsed.path, parsed.query, ""))


def _trusted_download_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.hostname == TRUSTED_HOST:
        return _trusted_url(url)
    path = unquote(parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.hostname != TRUSTED_BLOB_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or not path.casefold().startswith(TRUSTED_BLOB_PATH.casefold())
        or any(segment in {".", ".."} for segment in path.split("/"))
        or "\\" in path
    ):
        raise ValueError("Untrusted mineral-log download URL")
    return url


class _TrustedRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(
            req, fp, code, msg, headers, _trusted_download_url(newurl)
        )


# Validate redirects before following them; retain urllib's default TLS checks.
urlopen = build_opener(_TrustedRedirectHandler()).open


def source_filename(url: str) -> str:
    """Return the safe SM_*.csv basename from a trusted URL's fileName query."""
    parsed = urlsplit(_trusted_url(url))
    filenames = [
        value
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() == "filename"
    ]
    if len(filenames) != 1:
        raise ValueError("URL must contain exactly one fileName query parameter")
    filename = filenames[0].rsplit("/", 1)[-1]
    if (
        not re.fullmatch(r"SM_[A-Za-z0-9][A-Za-z0-9_.-]*\.csv", filename)
        or ".." in filename
    ):
        raise ValueError("fileName must end in a safe SM_*.csv basename")
    return filename


def validate_mineral_log(content: bytes, expected_id: str) -> None:
    """Require matching Dataset metadata, unique named headers and depth data.

    Individual malformed depth rows remain available for the builder to report
    and filter, but at least one row must contain two finite numeric depths.
    """
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig")), strict=True))
    datasets = [row[1].strip() for row in rows if len(row) > 1 and row[0] == "Dataset"]
    if datasets != [expected_id]:
        raise ValueError(f"Declared Dataset {datasets!r}, expected {expected_id!r}")
    header_index = next(
        (index for index, row in enumerate(rows) if row and row[0] == DEPTH_COLUMNS[0]),
        None,
    )
    if header_index is None or any(
        rows[header_index].count(column) != 1 for column in DEPTH_COLUMNS
    ):
        raise ValueError("Missing or ambiguous depth header")
    header_counts = Counter(column for column in rows[header_index] if column.strip())
    duplicate_headers = sorted(
        column for column, count in header_counts.items() if count > 1
    )
    if duplicate_headers:
        raise ValueError(f"Duplicate measurement header names: {duplicate_headers}")
    depth_indices = [rows[header_index].index(column) for column in DEPTH_COLUMNS]
    has_numeric_depths = False
    for row in rows[header_index + 1 :]:
        if len(row) != len(rows[header_index]):
            if row:
                raise ValueError("Measurement row does not match depth header")
            continue
        try:
            depths = [float(row[index]) for index in depth_indices]
        except ValueError:
            continue
        if all(math.isfinite(depth) for depth in depths):
            has_numeric_depths = True
    if not has_numeric_depths:
        raise ValueError("No finite numeric depth rows")


def acquire_logs(
    sources: pd.DataFrame,
    output_dir: Path,
    *,
    max_workers: int = 4,
    timeout_seconds: float = 30,
) -> pd.DataFrame:
    """Download unambiguous sources and return a manifest in source row order.

    Makes HTTPS requests (including upgrades of HTTP URLs) and creates the output
    directory and validated CSV files. Responses are limited to 20 MiB; requests
    use the supplied socket timeout, default certificate verification and no
    retries. Sources must be on DASC; redirects may also use its specific Azure
    blob host under /downloads/Hylogger/Mineral_Logs/. Existing files are
    read, validated and hashed without network access, never overwritten. New
    files are staged as .part files and atomically linked into place without
    clobbering files created concurrently. Temporary files are removed afterward.

    Expected network, filesystem and CSV errors become per-source failures.
    Failed rows have an error but no checksum, size or validation timestamp.
    Requires Dataset_ID, CollarId, Project and source_url columns; extra columns
    and the input index are preserved. The caller resolves source ambiguity.
    """
    missing = set(SOURCE_COLUMNS) - set(sources.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    if set(MANIFEST_COLUMNS) & set(sources.columns):
        raise ValueError("Sources already contain manifest columns")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")

    output_dir = Path(output_dir).absolute()
    # Serialise identical filenames even when distinct sources refer to them.
    path_locks: dict[str, Lock] = {}
    locks_lock = Lock()

    def acquire_source(source: dict) -> dict:
        record = dict.fromkeys(MANIFEST_COLUMNS)
        record.update(status="failed", error="")
        temporary_path = None
        try:
            url = source["source_url"]
            dataset_id = source["Dataset_ID"]
            if not isinstance(url, str) or not isinstance(dataset_id, str):
                raise ValueError("source_url and Dataset_ID must be strings")
            filename = source_filename(url)
            local_path = output_dir / filename
            record["local_path"] = str(local_path)
            with locks_lock:
                path_lock = path_locks.setdefault(filename, Lock())

            with path_lock:
                output_dir.mkdir(parents=True, exist_ok=True)
                if local_path.exists() or local_path.is_symlink():
                    with local_path.open("rb") as handle:
                        content = handle.read(MAX_RESPONSE_BYTES + 1)
                    status = "already_present"
                else:
                    request = Request(_trusted_url(url), headers=REQUEST_HEADERS)
                    with urlopen(request, timeout=timeout_seconds) as response:
                        _trusted_download_url(response.geturl())
                        content = response.read(MAX_RESPONSE_BYTES + 1)
                    status = "downloaded"

                if len(content) > MAX_RESPONSE_BYTES:
                    raise ValueError("Mineral log exceeds the 20 MiB size limit")
                validate_mineral_log(content, dataset_id)

                if status == "downloaded":
                    with tempfile.NamedTemporaryFile(
                        dir=output_dir,
                        prefix=f".{filename}.",
                        suffix=".part",
                        delete=False,
                    ) as handle:
                        temporary_path = Path(handle.name)
                        handle.write(content)
                    # Unlike rename/replace, linking cannot overwrite a raced-in cache.
                    os.link(temporary_path, local_path)
                    temporary_path.unlink()
                    temporary_path = None

                record.update(
                    status=status,
                    sha256=hashlib.sha256(content).hexdigest(),
                    file_size_bytes=len(content),
                    validated_at_utc=datetime.now(UTC).isoformat(),
                )
        except (OSError, URLError, HTTPException, ValueError, csv.Error) as error:
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as error:
                    record.update(
                        status="failed",
                        sha256=None,
                        file_size_bytes=None,
                        validated_at_utc=None,
                    )
                    record["error"] += f"; temporary file cleanup failed: {error}"
        return record

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(executor.map(acquire_source, sources.to_dict("records")))
    manifest = sources.copy()
    for column in MANIFEST_COLUMNS:
        manifest[column] = [record[column] for record in records]
    return manifest

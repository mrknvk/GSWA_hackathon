import hashlib
import io
from datetime import datetime
from threading import Barrier, Lock, get_ident
from urllib.error import URLError
from urllib.request import Request

import pandas as pd
import pytest

from gswa_hackathon import acquire_mineral_logs as acquisition

BLOB_URL = (
    "https://warsydprdstadasc.blob.core.windows.net/"
    "downloads/Hylogger/Mineral_Logs/SM_UNRD331.csv?ts=20261015212749"
)
UNTRUSTED_REDIRECTS = [
    "https://evil.example/log.csv",
    BLOB_URL.replace("warsydprdstadasc", "otheraccount"),
    BLOB_URL.replace(".net/", ".net.evil.example/"),
    BLOB_URL.replace("https:", "http:"),
    BLOB_URL.replace("https://", "https://user:password@"),
    BLOB_URL.replace(".net/", ".net:444/"),
    BLOB_URL.replace("/Mineral_Logs/", "/Other/"),
    BLOB_URL.replace("/Mineral_Logs/", "/Mineral_Logs_evil/"),
    BLOB_URL.replace("/SM_UNRD331", "/../SM_UNRD331"),
    BLOB_URL.replace("/SM_UNRD331", "/%2e%2e/SM_UNRD331"),
]


def mineral_log(dataset_id="TEST"):
    return (
        "GSWA TSG Summary Mineral Log,,\n"
        f"Dataset,{dataset_id}\n"
        "Depth_from_(m),Depth_to_(m),Feature\n"
        "0.0,1.0,0.5\n"
        "1.0,2.0,N.A.\n"
    ).encode()


def source_url(filename="SM_REMOTE.csv"):
    return (
        "https://dasc.dmirs.wa.gov.au/download?"
        f"fileName=Hylogger/Mineral_Logs/{filename}"
    )


def sources(*dataset_ids):
    return pd.DataFrame(
        [
            {
                "Dataset_ID": dataset_id,
                "CollarId": index,
                "Project": "Yilgarn",
                "source_url": source_url(f"SM_{dataset_id}.csv"),
            }
            for index, dataset_id in enumerate(dataset_ids)
        ],
        columns=acquisition.SOURCE_COLUMNS,
    )


class Response(io.BytesIO):
    def geturl(self):
        return source_url()


@pytest.fixture(autouse=True)
def prevent_network(monkeypatch):
    def unexpected_network(*args, **kwargs):
        pytest.fail("Unexpected network request")

    monkeypatch.setattr(acquisition, "urlopen", unexpected_network)


def test_valid_cache_is_hashed_without_network_or_filename_id_match(tmp_path):
    content = mineral_log()
    path = tmp_path / "SM_REMOTE.csv"
    path.write_bytes(content)
    original_mtime = path.stat().st_mtime_ns
    source = sources("TEST")
    source["source_url"] = source_url()
    source.index = [17]
    source["extra"] = "preserved"

    manifest = acquisition.acquire_logs(source, tmp_path)

    assert manifest.index.tolist() == [17]
    pd.testing.assert_frame_equal(manifest[source.columns], source)
    row = manifest.iloc[0]
    assert row.status == "already_present"
    assert row.local_path == str(path.absolute())
    assert row.sha256 == hashlib.sha256(content).hexdigest()
    assert row.file_size_bytes == len(content)
    assert row.error == ""
    assert datetime.fromisoformat(row.validated_at_utc).utcoffset().total_seconds() == 0
    assert path.stat().st_mtime_ns == original_mtime


def test_downloads_then_resumes_from_cache(tmp_path, monkeypatch):
    content = mineral_log()
    calls = []

    def open_log(request, *, timeout):
        assert isinstance(request, Request)
        assert dict(request.header_items()) == {
            "Accept": "text/csv",
            "User-agent": "GSWA-Hackathon/0.1",
        }
        calls.append((request.full_url, timeout))
        return Response(content)

    monkeypatch.setattr(acquisition, "urlopen", open_log)
    source = sources("TEST")
    source["source_url"] = source_url().replace("https:", "http:")
    output_dir = tmp_path / "logs"

    manifest = acquisition.acquire_logs(source, output_dir, timeout_seconds=12)
    resumed = acquisition.acquire_logs(source, output_dir)

    assert calls == [(source_url(), 12)]
    assert manifest.iloc[0].status == "downloaded"
    assert resumed.iloc[0].status == "already_present"
    assert manifest.iloc[0].sha256 == hashlib.sha256(content).hexdigest()
    assert manifest.iloc[0].file_size_bytes == len(content)
    assert (output_dir / "SM_REMOTE.csv").read_bytes() == content
    assert not list(output_dir.glob("*.part"))


@pytest.mark.parametrize(
    "content",
    [
        b"<html>upstream failure</html>",
        mineral_log("OTHER"),
        b"Dataset,TEST\nFeature\n1\n",
        b"Dataset,TEST\nDepth_from_(m),Depth_to_(m)\n",
        b"Dataset,TEST\nDepth_from_(m),Depth_to_(m)\ninvalid,inf\n",
        b"Dataset,TEST\nDepth_from_(m),Depth_to_(m)\n\xff,1\n",
        mineral_log() + b"malformed,row\n",
    ],
)
def test_malformed_download_is_not_installed(tmp_path, monkeypatch, content):
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **k: Response(content))

    row = acquisition.acquire_logs(sources("TEST"), tmp_path).iloc[0]

    assert row.status == "failed"
    assert row.error
    assert row.sha256 is None
    assert row.validated_at_utc is None
    assert not list(tmp_path.iterdir())


@pytest.fixture
def duplicate_feature_log():
    return (
        b"GSWA TSG Summary Mineral Log,,\n"
        b"Dataset,TEST\n"
        b"Depth_from_(m),Depth_to_(m),Feature,Feature,,\n"
        b"0.0,1.0,0.5,0.6,,\n"
        b"1.0,2.0,N.A.,0.7,,\n"
    )


@pytest.mark.parametrize("is_cached", [False, True])
def test_duplicate_feature_headers_fail_without_install_or_overwrite(
    tmp_path, monkeypatch, duplicate_feature_log, is_cached
):
    path = tmp_path / "SM_TEST.csv"
    if is_cached:
        path.write_bytes(duplicate_feature_log)
    else:
        monkeypatch.setattr(
            acquisition, "urlopen", lambda *a, **k: Response(duplicate_feature_log)
        )

    row = acquisition.acquire_logs(sources("TEST"), tmp_path).iloc[0]

    assert row.status == "failed"
    assert "Duplicate measurement header names: ['Feature']" in row.error
    assert row.sha256 is None
    assert row.validated_at_utc is None
    if is_cached:
        assert path.read_bytes() == duplicate_feature_log
    else:
        assert not list(tmp_path.iterdir())


def test_empty_trailing_header_fields_are_allowed(tmp_path, monkeypatch):
    content = (
        b"GSWA TSG Summary Mineral Log,,\n"
        b"Dataset,TEST\n"
        b"Depth_from_(m),Depth_to_(m),Feature,,\n"
        b"0.0,1.0,0.5,,\n"
        b"1.0,2.0,N.A.,,\n"
    )
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **k: Response(content))

    row = acquisition.acquire_logs(sources("TEST"), tmp_path).iloc[0]

    assert row.status == "downloaded"
    assert (tmp_path / "SM_TEST.csv").read_bytes() == content


@pytest.mark.parametrize("content", [mineral_log("OTHER"), b"broken cached CSV"])
def test_invalid_cache_is_not_overwritten(tmp_path, content):
    path = tmp_path / "SM_TEST.csv"
    path.write_bytes(content)

    row = acquisition.acquire_logs(sources("TEST"), tmp_path).iloc[0]

    assert row.status == "failed"
    assert row.error
    assert path.read_bytes() == content


def test_remote_failure_does_not_block_other_holes_or_change_order(
    tmp_path, monkeypatch
):
    def open_log(request, **kwargs):
        if "SM_BAD.csv" in request.full_url:
            raise URLError("connection failed")
        return Response(mineral_log("GOOD"))

    monkeypatch.setattr(acquisition, "urlopen", open_log)
    manifest = acquisition.acquire_logs(sources("BAD", "GOOD"), tmp_path)

    assert manifest.Dataset_ID.tolist() == ["BAD", "GOOD"]
    assert manifest.status.tolist() == ["failed", "downloaded"]
    assert "connection failed" in manifest.iloc[0].error
    assert (tmp_path / "SM_GOOD.csv").exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/download?fileName=SM_TEST.csv",
        "https://dasc.dmirs.wa.gov.au.evil.example/?fileName=SM_TEST.csv",
        "https://user@dasc.dmirs.wa.gov.au/?fileName=SM_TEST.csv",
        "https://dasc.dmirs.wa.gov.au:444/?fileName=SM_TEST.csv",
        "ftp://dasc.dmirs.wa.gov.au/?fileName=SM_TEST.csv",
        "not a URL",
        source_url("SM_..csv"),
        source_url("SM_TEST.exe"),
        source_url("SM_TEST%5Cbad.csv"),
        source_url("SM_TEST%00.csv"),
        source_url("SM_TEST.csv") + "&FILENAME=SM_OTHER.csv",
        "https://dasc.dmirs.wa.gov.au/download",
        BLOB_URL + "&fileName=SM_TEST.csv",
    ],
)
def test_invalid_urls_and_unsafe_filenames_fail_without_network(tmp_path, url):
    source = sources("TEST")
    source["source_url"] = url

    row = acquisition.acquire_logs(source, tmp_path).iloc[0]

    assert row.status == "failed"
    assert row.error
    assert not list(tmp_path.iterdir())


def test_filename_ignores_remote_directory_and_query_key_case():
    url = source_url().replace("fileName=", "FILENAME=")
    assert acquisition.source_filename(url) == "SM_REMOTE.csv"


@pytest.mark.parametrize("url", UNTRUSTED_REDIRECTS)
def test_untrusted_redirect_is_rejected_before_following(url):
    handler = acquisition._TrustedRedirectHandler()
    request = Request(source_url())

    with pytest.raises(ValueError, match="Untrusted"):
        handler.redirect_request(request, None, 302, "Found", {}, url)


@pytest.mark.parametrize(
    "url",
    [BLOB_URL, BLOB_URL.replace("/Hylogger/Mineral_Logs/", "/hylogger/mineral_logs/")],
)
def test_official_blob_redirect_is_allowed_and_preserves_headers(url):
    request = Request(source_url(), headers=acquisition.REQUEST_HEADERS)
    redirected = acquisition._TrustedRedirectHandler().redirect_request(
        request, None, 302, "Found", {}, url
    )
    assert redirected.full_url == url
    assert dict(redirected.header_items()) == dict(request.header_items())


def test_official_blob_final_response_is_accepted(tmp_path, monkeypatch):
    class BlobResponse(Response):
        def geturl(self):
            return BLOB_URL

    monkeypatch.setattr(
        acquisition, "urlopen", lambda *a, **k: BlobResponse(mineral_log())
    )
    row = acquisition.acquire_logs(sources("TEST"), tmp_path).iloc[0]
    assert row.status == "downloaded"
    assert (tmp_path / "SM_TEST.csv").read_bytes() == mineral_log()


@pytest.mark.parametrize("url", UNTRUSTED_REDIRECTS)
def test_untrusted_final_response_is_rejected_before_read(tmp_path, monkeypatch, url):
    class UntrustedResponse(Response):
        def geturl(self):
            return url

        def read(self, size=-1):
            pytest.fail("Untrusted response must not be read")

    monkeypatch.setattr(
        acquisition, "urlopen", lambda *a, **k: UntrustedResponse(mineral_log())
    )
    row = acquisition.acquire_logs(sources("TEST"), tmp_path).iloc[0]
    assert row.status == "failed"
    assert "Untrusted" in row.error
    assert not list(tmp_path.iterdir())


def test_oversized_response_is_bounded_and_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "MAX_RESPONSE_BYTES", 10)
    read_sizes = []

    class OversizedResponse(Response):
        def read(self, size=-1):
            read_sizes.append(size)
            return super().read(size)

    monkeypatch.setattr(
        acquisition, "urlopen", lambda *a, **k: OversizedResponse(mineral_log())
    )

    row = acquisition.acquire_logs(sources("TEST"), tmp_path).iloc[0]

    assert row.status == "failed"
    assert "size limit" in row.error
    assert read_sizes == [11]
    assert not list(tmp_path.iterdir())


def test_file_created_during_download_is_never_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "SM_TEST.csv"

    def open_log(*args, **kwargs):
        path.write_bytes(b"concurrently created cache")
        return Response(mineral_log())

    monkeypatch.setattr(acquisition, "urlopen", open_log)

    row = acquisition.acquire_logs(sources("TEST"), tmp_path).iloc[0]

    assert row.status == "failed"
    assert "FileExistsError" in row.error
    assert path.read_bytes() == b"concurrently created cache"
    assert not list(tmp_path.glob("*.part"))


def test_filesystem_failure_is_reported(tmp_path):
    output = tmp_path / "not_a_directory"
    output.write_text("preserve me")

    row = acquisition.acquire_logs(sources("TEST"), output).iloc[0]

    assert row.status == "failed"
    assert row.error
    assert output.read_text() == "preserve me"


def test_empty_sources_return_manifest_schema(tmp_path):
    source = sources()
    manifest = acquisition.acquire_logs(source, tmp_path)
    assert manifest.empty
    assert manifest.columns.tolist() == [
        *acquisition.SOURCE_COLUMNS,
        *acquisition.MANIFEST_COLUMNS,
    ]


@pytest.mark.parametrize("kwargs", [{"max_workers": 0}, {"timeout_seconds": 0}])
def test_invalid_configuration_is_not_a_per_source_failure(tmp_path, kwargs):
    with pytest.raises(ValueError):
        acquisition.acquire_logs(sources("TEST"), tmp_path, **kwargs)


def test_programmer_errors_are_not_swallowed(tmp_path, monkeypatch):
    def broken_opener(*args, **kwargs):
        raise RuntimeError("programmer error")

    monkeypatch.setattr(acquisition, "urlopen", broken_opener)
    with pytest.raises(RuntimeError, match="programmer error"):
        acquisition.acquire_logs(sources("TEST"), tmp_path)


def test_default_four_workers_download_concurrently(tmp_path, monkeypatch):
    barrier = Barrier(4, timeout=5)
    lock = Lock()
    thread_ids = set()
    source = sources("A", "B", "C", "D")
    contents = {
        row.source_url: mineral_log(row.Dataset_ID) for row in source.itertuples()
    }

    def open_log(request, **kwargs):
        with lock:
            thread_ids.add(get_ident())
        barrier.wait()
        return Response(contents[request.full_url])

    monkeypatch.setattr(acquisition, "urlopen", open_log)

    manifest = acquisition.acquire_logs(source, tmp_path)

    assert len(thread_ids) == 4
    assert manifest.status.tolist() == ["downloaded"] * 4
    assert manifest.Dataset_ID.tolist() == ["A", "B", "C", "D"]


def test_distinct_datasets_with_same_filename_do_not_overwrite(tmp_path, monkeypatch):
    source = sources("FIRST", "SECOND")
    source["source_url"] = source_url()
    monkeypatch.setattr(
        acquisition, "urlopen", lambda *a, **k: Response(mineral_log("FIRST"))
    )

    manifest = acquisition.acquire_logs(source, tmp_path, max_workers=1)

    assert manifest.status.tolist() == ["downloaded", "failed"]
    assert "Declared Dataset" in manifest.iloc[1].error
    assert (tmp_path / "SM_REMOTE.csv").read_bytes() == mineral_log("FIRST")


def test_timeout_becomes_source_failure(tmp_path, monkeypatch):
    def timed_out(*args, **kwargs):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(acquisition, "urlopen", timed_out)

    row = acquisition.acquire_logs(sources("TEST"), tmp_path).iloc[0]

    assert row.status == "failed"
    assert "TimeoutError: read timed out" in row.error
    assert not list(tmp_path.iterdir())


def test_mixed_numeric_depth_issues_remain_for_builder_filtering():
    content = mineral_log() + b"invalid,3,N.A.\ninf,4,N.A.\n"
    acquisition.validate_mineral_log(content, "TEST")

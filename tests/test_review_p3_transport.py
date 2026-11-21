"""P3 review fixes in the transport (review-p3 L-5).

L-5: the ``name`` and ``filename`` parameters of a multipart part follow the WHATWG multipart rule that Node's
``FormData`` uses: only CR becomes ``%0D``, LF ``%0A`` and ``"`` ``%22``. A backslash and every other character are
sent as they are, and a CR, LF or quote still cannot end the parameter or the header line.

The checks read the raw body bytes: a MIME parser would read a backslash inside a quoted string as an escape.
"""

from __future__ import annotations

import re

import anyio
import pytest
from helpers.mock_transport import MockAPI, RecordedRequest, json_response, sequence

from devupai._client import DEFAULT_RETRY, MultipartParts


def send(client_kind: str, parts: MultipartParts) -> RecordedRequest:
    api = MockAPI(sequence(json_response({})))
    client = api.client(client_kind)
    if client_kind == "sync":
        client._request("POST", "/fixture/upload", multipart=parts, retry=DEFAULT_RETRY)
    else:

        async def main() -> None:
            await client._request("POST", "/fixture/upload", multipart=parts, retry=DEFAULT_RETRY)

        anyio.run(main)
    assert len(api.calls) == 1
    return api.calls[0]


def disposition_lines(request: RecordedRequest) -> list[bytes]:
    """The Content-Disposition line of each part, as sent."""
    return re.findall(rb"Content-Disposition: [^\r\n]*", request.content)


@pytest.mark.parametrize(
    ("given", "sent"),
    [
        ("a\\b.png", b"a\\b.png"),
        ("dir\\\\sub\\file.png", b"dir\\\\sub\\file.png"),
        ("tab\tand\x1bescape", b"tab\tand\x1bescape"),
        ("café.png", "café.png".encode()),
    ],
    ids=["backslash", "double_backslash", "tab_and_esc", "non_ascii"],
)
def test_a_filename_keeps_backslashes_and_other_characters(client_kind: str, given: str, sent: bytes) -> None:
    request = send(client_kind, [("image", (given, b"\x89PNG", "image/png"))])
    assert disposition_lines(request) == [b'Content-Disposition: form-data; name="image"; filename="' + sent + b'"']


def test_a_name_keeps_backslashes(client_kind: str) -> None:
    request = send(client_kind, [("field\\one", "value")])
    assert disposition_lines(request) == [b'Content-Disposition: form-data; name="field\\one"']


@pytest.mark.parametrize(
    ("given", "sent"),
    [
        ('na"me', b"na%22me"),
        ("a\rb", b"a%0Db"),
        ("a\nb", b"a%0Ab"),
        ('x"\r\nX-Injected: 1', b"x%22%0D%0AX-Injected: 1"),
    ],
    ids=["quote", "cr", "lf", "crlf_injection"],
)
def test_cr_lf_and_quote_are_percent_encoded_in_names_and_filenames(client_kind: str, given: str, sent: bytes) -> None:
    request = send(client_kind, [(given, "text"), ("file", (given, b"data", None))])
    assert disposition_lines(request) == [
        b'Content-Disposition: form-data; name="' + sent + b'"',
        b'Content-Disposition: form-data; name="file"; filename="' + sent + b'"',
    ]
    # The header line is never broken: no injected header line exists in the body.
    assert not re.search(rb"\r\nX-Injected", request.content)
    assert b"Content-Type: application/octet-stream\r\n\r\ndata\r\n" in request.content
